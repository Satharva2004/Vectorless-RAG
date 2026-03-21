import json
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None

try:
    from groq import Groq
except ImportError:
    Groq = None


CHAPTER_PATTERN = re.compile(
    r"^(chapter|book|part)\s+([a-z0-9ivxlcdm\-']+)\b",
    re.IGNORECASE,
)
LEADING_DECORATION_PATTERN = re.compile(r"^[^A-Za-z]+")
TRAILING_DECORATION_PATTERN = re.compile(r"[^A-Za-z0-9]+$")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


@dataclass
class PageData:
    page_number: int
    raw_text: str
    text: str


@dataclass
class SummaryConfig:
    provider: str = "heuristic"
    model_name: str | None = None
    timeout_seconds: int = 30
    google_api_key: str | None = None
    groq_api_key: str | None = None
    local_model_id: str | None = None


def build_pageindex_tree(
    pdf_paths: list[str],
    corpus_title: str = "Vectorless RAG Corpus",
    summary_config: SummaryConfig | None = None,
) -> dict[str, Any]:
    summary_config = summary_config or SummaryConfig()
    _log(f"[TREE] Building corpus tree for {len(pdf_paths)} PDF file(s)")
    documents = []

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        document_node = _build_document_node(
            pdf_path=path,
            summary_config=summary_config,
        )
        documents.append(document_node)

    root_summary_source = "\n\n".join(
        node.get("summary_source", "") for node in documents if node.get("summary_source")
    )

    root = {
        "id": _corpus_id(pdf_paths, corpus_title),
        "type": "corpus",
        "title": corpus_title,
        "children": documents,
        "document_count": len(documents),
        "summary_provider": summary_config.provider,
        "summary_model": summary_config.model_name or summary_config.local_model_id,
        "summary": _summarize_text(
            root_summary_source,
            summary_config=summary_config,
            fallback_label=corpus_title,
        ),
    }
    _log(f"[TREE] Built corpus tree with {len(documents)} document node(s)")
    return root


def discover_pdf_inputs(
    input_mode: str,
    combined_pdf: str | None,
    input_dir: str | None,
) -> list[str]:
    if input_mode == "combined":
        if not combined_pdf:
            raise ValueError("combined mode requires --combined-pdf")
        path = Path(combined_pdf)
        if not path.exists():
            raise FileNotFoundError(f"Combined PDF not found: {path}")
        return [str(path)]

    if not input_dir:
        raise ValueError(f"{input_mode} mode requires --input-dir")

    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    pdf_paths = sorted(str(path) for path in directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    if input_mode == "individual":
        return [path for path in pdf_paths if Path(path).name.lower() != "combined.pdf"]

    if input_mode == "auto":
        if combined_pdf:
            combined_path = Path(combined_pdf)
            if combined_path.exists():
                return [str(combined_path)]

        non_combined = [path for path in pdf_paths if Path(path).name.lower() != "combined.pdf"]
        return non_combined or pdf_paths[:1]

    raise ValueError(f"Unsupported input mode: {input_mode}")


def save_tree(tree: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_tree = _strip_internal_fields(tree)
    path.write_text(json.dumps(cleaned_tree, indent=2, ensure_ascii=True), encoding="utf-8")
    _log(f"[TREE] Saved tree JSON to {path.resolve()}")


def _build_document_node(
    pdf_path: Path,
    summary_config: SummaryConfig,
) -> dict[str, Any]:
    _log(f"[TREE] Reading document: {pdf_path}")
    pages = _extract_pages(pdf_path)
    page_nodes = [_build_page_node(pdf_path, page) for page in pages]
    chapter_nodes = _group_pages_into_chapters(
        pdf_path=pdf_path,
        pages=pages,
        page_nodes=page_nodes,
        summary_config=summary_config,
    )

    document_text = "\n\n".join(page.text for page in pages if page.text)
    title = _humanize_title(pdf_path.stem)

    return {
        "id": _document_id(pdf_path),
        "type": "document",
        "title": title,
        "source": str(pdf_path),
        "page_count": len(pages),
        "children": chapter_nodes,
        "summary": _summarize_text(
            document_text,
            summary_config=summary_config,
            fallback_label=title,
        ),
        "summary_source": document_text[:6000],
    }


def _extract_pages(pdf_path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            raw_text = page.get_text("text") or ""
            pages.append(
                PageData(
                    page_number=index + 1,
                    raw_text=raw_text,
                    text=_normalize_text(raw_text),
                )
            )
    _log(f"[TREE] Extracted {len(pages)} page(s) from {pdf_path.name}")
    return pages


def _build_page_node(pdf_path: Path, page: PageData) -> dict[str, Any]:
    snippet = _first_sentences(page.text, sentence_count=2, char_limit=280)
    return {
        "id": _page_id(page.page_number),
        "type": "page",
        "title": f"Page {page.page_number}",
        "page_number": page.page_number,
        "source": str(pdf_path),
        "text_preview": snippet,
        "text_length": len(page.text),
    }


def _group_pages_into_chapters(
    pdf_path: Path,
    pages: list[PageData],
    page_nodes: list[dict[str, Any]],
    summary_config: SummaryConfig,
) -> list[dict[str, Any]]:
    chapter_starts: list[tuple[int, str]] = []

    for index, page in enumerate(pages):
        title = _detect_chapter_title(page.text)
        if not title:
            title = _detect_chapter_title(page.raw_text)
        if title:
            chapter_starts.append((index, title))

    if not chapter_starts:
        _log(f"[TREE] No explicit chapter boundaries found in {pdf_path.name}; creating overview node")
        return [_build_single_chapter(pdf_path, pages, page_nodes, summary_config)]

    chapters: list[dict[str, Any]] = []
    _log(f"[TREE] Detected {len(chapter_starts)} chapter start(s) in {pdf_path.name}")

    for start_index, (page_index, title) in enumerate(chapter_starts, start=1):
        next_page_index = (
            chapter_starts[start_index][0]
            if start_index < len(chapter_starts)
            else len(pages)
        )
        chapter_pages = pages[page_index:next_page_index]
        chapter_page_nodes = page_nodes[page_index:next_page_index]
        chapter_text = "\n\n".join(page.text for page in chapter_pages if page.text)

        chapters.append(
            {
                "id": _chapter_id(start_index),
                "type": "chapter",
                "title": title,
                "page_start": chapter_pages[0].page_number,
                "page_end": chapter_pages[-1].page_number,
                "full_text": chapter_text,
                "children": chapter_page_nodes,
                "summary": _summarize_text(
                    chapter_text,
                    summary_config=summary_config,
                    fallback_label=title,
                ),
                "summary_source": chapter_text[:4000],
            }
        )
        _log(
            f"[TREE] Chapter node created: {title} "
            f"(pages {chapter_pages[0].page_number}-{chapter_pages[-1].page_number})"
        )

    return chapters


def _build_single_chapter(
    pdf_path: Path,
    pages: list[PageData],
    page_nodes: list[dict[str, Any]],
    summary_config: SummaryConfig,
) -> dict[str, Any]:
    document_title = _humanize_title(pdf_path.stem)
    full_text = "\n\n".join(page.text for page in pages if page.text)
    return {
        "id": _chapter_id(1),
        "type": "chapter",
        "title": f"{document_title} Overview",
        "page_start": 1 if pages else 0,
        "page_end": pages[-1].page_number if pages else 0,
        "full_text": full_text,
        "children": page_nodes,
        "summary": _summarize_text(
            full_text,
            summary_config=summary_config,
            fallback_label=document_title,
        ),
        "summary_source": full_text[:4000],
    }


def _detect_chapter_title(text: str) -> str | None:
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_lines = lines[:6]

    for index, line in enumerate(candidate_lines):
        normalized = WHITESPACE_PATTERN.sub(" ", line).strip()
        normalized = LEADING_DECORATION_PATTERN.sub("", normalized)
        normalized = TRAILING_DECORATION_PATTERN.sub("", normalized)
        if not normalized:
            continue

        if CHAPTER_PATTERN.match(normalized):
            normalized = _compact_heading_line(normalized)
            if index + 1 < len(candidate_lines):
                next_line = candidate_lines[index + 1].strip()
                next_line = _compact_heading_line(next_line)
                if next_line and len(next_line.split()) <= 10:
                    return f"{normalized}: {next_line}"
            return normalized

    return None


def _summarize_text(
    text: str,
    summary_config: SummaryConfig,
    fallback_label: str,
) -> str:
    source = text.strip()
    if not source:
        return f"No extracted text was available for {fallback_label}."

    clipped = source[:5000]
    provider = summary_config.provider.lower()

    if provider in {"gemini", "auto"}:
        gemini_summary = _summarize_with_gemini(clipped, summary_config)
        if gemini_summary:
            return gemini_summary

    if provider in {"groq", "auto"}:
        groq_summary = _summarize_with_groq(clipped, summary_config)
        if groq_summary:
            return groq_summary

    if provider == "local":
        local_summary = _summarize_with_local_model(clipped, summary_config)
        if local_summary:
            return local_summary

    if provider == "auto":
        local_summary = _summarize_with_local_model(clipped, summary_config)
        if local_summary:
            return local_summary

    return _first_sentences(clipped, sentence_count=2, char_limit=360)


def _summarize_with_gemini(text: str, summary_config: SummaryConfig) -> str | None:
    api_key = summary_config.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None

    model_name = summary_config.model_name or DEFAULT_GEMINI_MODEL
    prompt = (
        "Create a concise retrieval-tree summary for this document section. "
        "Return exactly 2 sentences focusing on major characters, events, and themes. "
        "Avoid markdown and bullet points.\n\n"
        f"{text}"
    )

    try:
        modern_summary = _summarize_with_google_genai_sdk(
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=summary_config.timeout_seconds,
        )
        if modern_summary:
            return modern_summary

        legacy_summary = _summarize_with_legacy_google_sdk(
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=summary_config.timeout_seconds,
        )
        if legacy_summary:
            return legacy_summary
    except Exception:
        return None
    return None


def _summarize_with_groq(text: str, summary_config: SummaryConfig) -> str | None:
    api_key = summary_config.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key or Groq is None:
        return None

    model_name = summary_config.model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    prompt = (
        "Create a concise retrieval-tree summary for this document section. "
        "Return exactly 2 sentences focusing on major characters, events, and themes. "
        "Avoid markdown and bullet points.\n\n"
        f"{text}"
    )

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=float(summary_config.timeout_seconds),
        )
        response_text = completion.choices[0].message.content or ""
        cleaned = response_text.strip()
        return cleaned or None
    except Exception:
        return None


def _summarize_with_local_model(text: str, summary_config: SummaryConfig) -> str | None:
    local_model_id = summary_config.local_model_id
    if not local_model_id or not shutil.which("llama"):
        return None

    prompt = (
        "Summarize this document section for a retrieval tree. "
        "Return 2 concise sentences focusing on characters, events, and themes.\n\n"
        f"{text}"
    )
    try:
        result = subprocess.run(
            ["llama", "model", "prompt", "--model-id", local_model_id, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=summary_config.timeout_seconds,
        )
        response = result.stdout.strip()
        return response or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _summarize_with_google_genai_sdk(
    prompt: str,
    model_name: str,
    api_key: str,
    timeout_seconds: int,
) -> str | None:
    if google_genai is None or google_genai_types is None:
        return None

    with google_genai.Client(
        api_key=api_key,
        http_options=google_genai_types.HttpOptions(
            api_version="v1alpha",
            client_args={"timeout": timeout_seconds},
        ),
    ) as client:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        response_text = getattr(response, "text", "") or ""
        cleaned = response_text.strip()
        return cleaned or None


def _summarize_with_legacy_google_sdk(
    prompt: str,
    model_name: str,
    api_key: str,
    timeout_seconds: int,
) -> str | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            import google.generativeai as deprecated_google_genai
        except ImportError:
            return None
        deprecated_google_genai.configure(api_key=api_key)
        model = deprecated_google_genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout_seconds},
        )
    response_text = getattr(response, "text", "") or ""
    cleaned = response_text.strip()
    return cleaned or None


def _first_sentences(text: str, sentence_count: int, char_limit: int) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(normalized) if part.strip()]
    joined = " ".join(sentences[:sentence_count]).strip()
    if not joined:
        joined = normalized[:char_limit]

    if len(joined) > char_limit:
        return joined[: char_limit - 3].rstrip() + "..."
    return joined


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def _humanize_title(raw_title: str) -> str:
    title = raw_title.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _compact_heading_line(text: str) -> str:
    compact = WHITESPACE_PATTERN.sub(" ", text).strip()
    compact = LEADING_DECORATION_PATTERN.sub("", compact)
    compact = TRAILING_DECORATION_PATTERN.sub("", compact)
    separators = [" . ", " ? ", " ! ", " — ", " – ", " - "]
    for separator in separators:
        if separator in compact:
            compact = compact.split(separator, 1)[0].strip()
    return compact[:120].strip()


def _corpus_id(pdf_paths: list[str], corpus_title: str) -> str:
    if len(pdf_paths) == 1:
        stem = Path(pdf_paths[0]).stem.lower()
        if "hp 1" in stem or "philosopher" in stem or "sorcerer" in stem:
            return "corpus_hp1"
    return f"corpus_{_slugify(corpus_title)}"


def _document_id(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    if "hp 1" in stem or "philosopher" in stem or "sorcerer" in stem:
        return "document_hp1"
    return f"document_{_slugify(pdf_path.stem)}"


def _chapter_id(index: int) -> str:
    return f"chapter_{index:03d}"


def _page_id(page_number: int) -> str:
    return f"page_{page_number:03d}"


def _slugify(value: str) -> str:
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _strip_internal_fields(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_internal_fields(item) for item in node]

    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key == "summary_source":
                continue
            cleaned[key] = _strip_internal_fields(value)
        return cleaned

    return node


def _log(message: str) -> None:
    safe_message = message.encode("ascii", errors="replace").decode("ascii")
    print(safe_message)
