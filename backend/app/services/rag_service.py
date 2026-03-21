import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger("vectorless_rag.rag")

CHAPTER_SUMMARY_LIMIT = 180
MAX_BATCH_CHARS = 12000
MAX_BATCH_CHAPTERS = 24
MAX_LLM_RETRIES = 3
MAX_ANSWER_CONTEXT_CHARS = 500000
MAX_PREFILTER_CHAPTERS = 8
MAX_SNIPPET_CHARS = 500


class TreeRoutingError(Exception):
    pass


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def select_relevant_chapters(
    question: str,
    tree: dict[str, Any],
    max_chapters: int = 3,
) -> dict[str, Any]:
    logger.info("[ROUTE] Question received: %s", question)
    chapters = _extract_chapter_candidates(tree)
    if not chapters:
        raise TreeRoutingError("No chapter nodes with summaries were found in the tree.")
    logger.info("[ROUTE] Found %s chapter candidate(s)", len(chapters))

    prefiltered_chapters = _prefilter_chapters(question, chapters)
    logger.info(
        "[ROUTE] Prefiltered to %s chapter candidate(s) before model selection",
        len(prefiltered_chapters),
    )
    for chapter in prefiltered_chapters:
        logger.info(
            "[ROUTE][SUMMARY] %s | %s | %s",
            chapter["node_id"],
            chapter["title"],
            chapter["summary"][:220],
        )

    compact_chapters = [_compact_chapter(chapter) for chapter in prefiltered_chapters]
    shortlisted = _shortlist_chapters(question, compact_chapters, max_chapters=max_chapters)
    logger.info("[ROUTE] Shortlisted %s chapter candidate(s) for final selection", len(shortlisted))
    for chapter in shortlisted:
        logger.info(
            "[ROUTE][SHORTLIST] %s | %s | %s",
            chapter["node_id"],
            chapter["title"],
            chapter["summary"][:220],
        )

    final_selection = _request_node_selection(
        question=question,
        chapters=shortlisted,
        max_chapters=max_chapters,
    )

    selected_ids = final_selection["node_ids"]
    selected_chapters = [chapter for chapter in chapters if chapter["node_id"] in selected_ids]

    if not selected_chapters:
        selected_chapters = chapters[:max_chapters]
        selected_ids = [chapter["node_id"] for chapter in selected_chapters]
        logger.warning(
            "[ROUTE] Falling back to first chapters because the model did not return valid node IDs"
        )

    logger.info("[ROUTE] Selected node_ids: %s", selected_ids)

    return {
        "question": question,
        "selected_node_ids": selected_ids,
        "selected_chapters": selected_chapters,
        "model": settings.OPENROUTER_MODEL,
        "provider": "openrouter",
        "reasoning": final_selection.get("reasoning"),
        "assistant_content": final_selection.get("assistant_content"),
        "reasoning_details": final_selection.get("reasoning_details"),
    }


def select_nodes_stream_from_tree(
    question: str,
    tree: dict[str, Any],
    max_nodes: int = 3,
):
    logger.info("[ROUTE] Streaming node selection started for question: %s", question)
    yield _sse("log", {"stage": "route", "message": "starting", "question": question})

    chapters = _extract_chapter_candidates(tree)
    if not chapters:
        logger.error("[ROUTE] No chapter nodes with summaries were found in the tree")
        yield _sse("error", {"message": "No chapter nodes with summaries were found in the tree."})
        return

    yield _sse("log", {"stage": "route", "message": "chapter_candidates", "count": len(chapters)})

    prefiltered_chapters = _prefilter_chapters(question, chapters)
    logger.info(
        "[ROUTE] Prefiltered to %s chapter candidate(s) before final selection",
        len(prefiltered_chapters),
    )
    yield _sse("log", {"stage": "route", "message": "prefiltered", "count": len(prefiltered_chapters)})
    for chapter in prefiltered_chapters:
        logger.info(
            "[ROUTE][SUMMARY] %s | %s | %s",
            chapter["node_id"],
            chapter["title"],
            chapter["summary"][:220],
        )
        yield _sse(
            "candidate",
            {
                "node_id": chapter["node_id"],
                "title": chapter["title"],
                "page_start": chapter["page_start"],
                "page_end": chapter["page_end"],
                "summary": chapter["summary"],
            },
        )

    compact_chapters = [_compact_chapter(chapter) for chapter in prefiltered_chapters]
    shortlisted = _shortlist_chapters(question, compact_chapters, max_chapters=max_nodes)
    logger.info("[ROUTE] Shortlisted %s chapter candidate(s)", len(shortlisted))
    yield _sse("log", {"stage": "route", "message": "shortlisted", "count": len(shortlisted)})
    for chapter in shortlisted:
        logger.info(
            "[ROUTE][SHORTLIST] %s | %s | %s",
            chapter["node_id"],
            chapter["title"],
            chapter["summary"][:220],
        )
        yield _sse(
            "shortlist",
            {
                "node_id": chapter["node_id"],
                "title": chapter["title"],
                "page_start": chapter["page_start"],
                "page_end": chapter["page_end"],
                "summary": chapter["summary"],
            },
        )

    final_selection = None
    for chunk in _stream_request_node_selection(
        question=question,
        chapters=shortlisted,
        max_chapters=max_nodes,
    ):
        if chunk["type"] == "reasoning":
            yield _sse(
                "reasoning",
                {
                    "provider": "openrouter",
                    "model": settings.OPENROUTER_MODEL,
                    "text": chunk["text"],
                },
            )
        elif chunk["type"] == "error":
            yield _sse("error", {"message": chunk["message"]})
            return
        elif chunk["type"] == "result":
            final_selection = chunk

    if not final_selection:
        yield _sse("error", {"message": "Processing failed to yield a final selection."})
        return

    logger.info("[ROUTE] Streaming selection picked node_ids: %s", final_selection["node_ids"])

    yield _sse(
        "selection",
        {
            "question": question,
            "selected_node_ids": final_selection["node_ids"],
            "model": settings.OPENROUTER_MODEL,
            "provider": "openrouter",
            "reasoning": final_selection.get("reasoning"),
            "assistant_content": final_selection.get("assistant_content"),
            "reasoning_details": final_selection.get("reasoning_details"),
        },
    )
    yield _sse("done", {"message": "complete"})


def load_tree_from_path(tree_path: str) -> dict[str, Any]:
    path = Path(tree_path)
    if not path.is_absolute():
        path = Path.cwd() / tree_path
    if not path.exists():
        raise TreeRoutingError(f"Tree file not found: {path}")
    logger.info("[TREE] Loading tree from %s", path)
    return json.loads(path.read_text(encoding="utf-8"))




def generate_answer_from_node_ids(
    question: str,
    tree: dict[str, Any],
    node_ids: list[str],
    assistant_content: str | None = None,
    reasoning_details: Any | None = None,
) -> dict[str, Any]:
    logger.info("[ANSWER] Starting answer generation from provided node_ids: %s", node_ids)
    chapter_contexts = fetch_chapter_contexts(tree, node_ids)
    if not chapter_contexts:
        raise TreeRoutingError("Could not fetch full text for the provided node_ids.")

    logger.info("[FETCH] Loaded %s full chapter context(s)", len(chapter_contexts))
    retrieved_nodes = []
    for chapter in chapter_contexts:
        logger.info(
            "[FETCH] %s | %s | pages %s-%s | chars=%s",
            chapter["node_id"],
            chapter["title"],
            chapter["page_start"],
            chapter["page_end"],
            len(chapter["full_text"]),
        )
        retrieved_nodes.append(
            {
                "title": chapter["title"],
                "node_id": chapter["node_id"],
                "relevant_contents": [
                    {
                        "page_index": chapter["page_start"],
                        "relevant_content": chapter["full_text"],
                    }
                ],
            }
        )

    answer = _generate_grounded_answer_from_retrieval(
        question=question,
        retrieved_nodes=retrieved_nodes,
        assistant_content=assistant_content,
        reasoning_details=reasoning_details,
    )
    citations = [
        {
            "node_id": chapter["node_id"],
            "title": chapter["title"],
            "page_start": chapter["page_start"],
            "page_end": chapter["page_end"],
            "source": chapter.get("source"),
        }
        for chapter in chapter_contexts
    ]
    return {
        "question": question,
        "answer": answer,
        "selected_node_ids": [chapter["node_id"] for chapter in chapter_contexts],
        "citations": citations,
        "model": settings.GEMINI_MODEL,
        "provider": "gemini",
    }


def generate_answer_stream_from_node_ids(
    question: str,
    tree: dict[str, Any],
    node_ids: list[str],
    assistant_content: str | None = None,
    reasoning_details: Any | None = None,
):
    logger.info("[ANSWER] Streaming answer generation started for node_ids: %s", node_ids)
    yield _sse("log", {"stage": "answer", "message": "starting", "question": question})

    chapter_contexts = fetch_chapter_contexts(tree, node_ids)
    if not chapter_contexts:
        logger.error("[FETCH] Could not fetch full text for provided node_ids: %s", node_ids)
        yield _sse("error", {"message": "Could not fetch full text for the provided node_ids."})
        return

    citations = [
        {
            "node_id": chapter["node_id"],
            "title": chapter["title"],
            "page_start": chapter["page_start"],
            "page_end": chapter["page_end"],
            "source": chapter.get("source"),
        }
        for chapter in chapter_contexts
    ]
    yield _sse(
        "citations",
        {
            "selected_node_ids": [chapter["node_id"] for chapter in chapter_contexts],
            "citations": citations,
            "model": settings.GEMINI_MODEL,
            "provider": "gemini",
        },
    )

    retrieved_nodes = []
    for chapter in chapter_contexts:
        logger.info(
            "[FETCH] %s | %s | pages %s-%s | chars=%s",
            chapter["node_id"],
            chapter["title"],
            chapter["page_start"],
            chapter["page_end"],
            len(chapter["full_text"]),
        )
        yield _sse(
            "log",
            {
                "stage": "answer",
                "message": "context_loaded",
                "node_id": chapter["node_id"],
                "title": chapter["title"],
                "page_start": chapter["page_start"],
                "page_end": chapter["page_end"],
                "chars": len(chapter["full_text"]),
            },
        )
        retrieved_nodes.append(
            {
                "title": chapter["title"],
                "node_id": chapter["node_id"],
                "relevant_contents": [
                    {
                        "page_index": chapter["page_start"],
                        "relevant_content": chapter["full_text"],
                    }
                ],
            }
        )

    for chunk in _stream_answer_from_retrieval(
        question=question,
        retrieved_nodes=retrieved_nodes,
        assistant_content=assistant_content,
        reasoning_details=reasoning_details,
    ):
        yield chunk


def fetch_chapter_contexts(tree: dict[str, Any], node_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(node_ids)
    results: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], current_source: str | None = None) -> None:
        node_type = node.get("type")
        source = node.get("source") or current_source

        if node_type == "chapter" and node.get("id") in wanted:
            results.append(
                {
                    "node_id": node["id"],
                    "title": node.get("title", "Untitled Chapter"),
                    "page_start": int(node.get("page_start", 0)),
                    "page_end": int(node.get("page_end", 0)),
                    "full_text": node.get("full_text", ""),
                    "source": source,
                }
            )

        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, source)

    walk(tree)
    ordered = []
    lookup = {item["node_id"]: item for item in results if item.get("full_text")}
    for node_id in node_ids:
        if node_id in lookup:
            ordered.append(lookup[node_id])
    return ordered


def _extract_chapter_candidates(tree: dict[str, Any]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], current_source: str | None = None) -> None:
        node_type = node.get("type")
        source = node.get("source") or current_source

        if node_type == "chapter":
            chapters.append(
                {
                    "node_id": node.get("id", ""),
                    "title": node.get("title", "Untitled Chapter"),
                    "page_start": int(node.get("page_start", 0)),
                    "page_end": int(node.get("page_end", 0)),
                    "summary": node.get("summary", ""),
                    "source": source,
                }
            )

        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, source)

    walk(tree)
    return [chapter for chapter in chapters if chapter["node_id"] and chapter["summary"]]


def _compact_chapter(chapter: dict[str, Any]) -> dict[str, Any]:
    compact = dict(chapter)
    compact["summary"] = chapter["summary"][:CHAPTER_SUMMARY_LIMIT]
    return compact


def _prefilter_chapters(question: str, chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(chapters) <= MAX_PREFILTER_CHAPTERS:
        return chapters

    query_terms = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    if not query_terms:
        return chapters[:MAX_PREFILTER_CHAPTERS]

    scored: list[tuple[int, dict[str, Any]]] = []
    for index, chapter in enumerate(chapters):
        haystack = f"{chapter['title']} {chapter['summary']}".lower()
        score = 0
        for term in query_terms:
            if term in haystack:
                score += 3 if term in chapter["title"].lower() else 1
        scored.append((score, chapters[index]))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_scored = [chapter for score, chapter in scored[:MAX_PREFILTER_CHAPTERS] if score > 0]
    if top_scored:
        return top_scored
    return chapters[:MAX_PREFILTER_CHAPTERS]


def _shortlist_chapters(
    question: str,
    chapters: list[dict[str, Any]],
    max_chapters: int,
) -> list[dict[str, Any]]:
    batches = _chunk_chapters(chapters)
    logger.info("[ROUTE] Split chapters into %s selection batch(es)", len(batches))
    if len(batches) == 1:
        return chapters

    shortlisted_ids: list[str] = []
    shortlist_target = min(max_chapters, 2)

    for batch in batches:
        logger.info("[ROUTE] Sending shortlist batch with %s chapter(s)", len(batch))
        batch_result = _request_node_selection(
            question=question,
            chapters=batch,
            max_chapters=shortlist_target,
        )
        shortlisted_ids.extend(batch_result["node_ids"])

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    chapter_lookup = {chapter["node_id"]: chapter for chapter in chapters}

    for node_id in shortlisted_ids:
        if node_id in chapter_lookup and node_id not in seen_ids:
            deduped.append(chapter_lookup[node_id])
            seen_ids.add(node_id)

    if not deduped:
        return chapters[: max_chapters * 4]

    return deduped


def _chapter_line(chapter: dict[str, Any]) -> str:
    return f"{chapter.get('title', '')} {chapter.get('summary', '')}"


def _chunk_chapters(chapters: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for chapter in chapters:
        chapter_line = _chapter_line(chapter)
        if current_batch and (
            len(current_batch) >= MAX_BATCH_CHAPTERS
            or current_chars + len(chapter_line) > MAX_BATCH_CHARS
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(chapter)
        current_chars += len(chapter_line)

    if current_batch:
        batches.append(current_batch)

    return batches

def _stream_request_node_selection(
    question: str,
    chapters: list[dict[str, Any]],
    max_chapters: int,
):
    api_key = settings.OPENROUTER_API_KEY.strip()
    if not api_key:
        yield {"type": "error", "message": "OPENROUTER_API_KEY is not configured."}
        return

    tree_without_text = [
        {
            "node_id": chapter["node_id"],
            "node_title": chapter["title"],
            "summary": chapter["summary"],
        }
        for chapter in chapters
    ]

    user_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {question}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please output your reasoning first, securely enclosed in <think> and </think> tags.
Then, output your final node selection array as a JSON object EXACTLY in this format on a new line:
{{
    "node_list": ["node_id_1", "node_id_2"]
}}
Do not output anything else but your reasoning block and the JSON block.
"""

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": user_prompt.strip()},
        ],
        "stream": True,
        "include_reasoning": True,
        "extra_body": {
            "reasoning": {
                "enabled": True
            }
        }
    }

    logger.info(
        "[FEATHERLESS] Streaming node selection request: candidates=%s",
        len(chapters),
    )
    
    api_url = "https://api.featherless.ai/v1/chat/completions"
    timeout = settings.OPENROUTER_TIMEOUT_SECONDS
    
    content_buffer = ""
    json_buffer = ""
    inside_think_tag = False
    seen_think_tag = False
    full_content = ""

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    
                    reasoning_chunk = delta.get("reasoning")
                    if reasoning_chunk:
                        yield {"type": "reasoning", "text": reasoning_chunk}
                        
                    content_chunk = delta.get("content")
                    if content_chunk:
                        full_content += content_chunk
                        content_buffer += content_chunk
                        
                        if not seen_think_tag:
                            if "<think>" in content_buffer:
                                seen_think_tag = True
                                inside_think_tag = True
                                idx = content_buffer.find("<think>")
                                content_buffer = content_buffer[idx+7:]
                            elif "{" in content_buffer and not seen_think_tag:
                                json_buffer += content_buffer
                                content_buffer = ""
                                continue
                                
                        if inside_think_tag:
                            if "</think>" in content_buffer:
                                idx = content_buffer.find("</think>")
                                yield {"type": "reasoning", "text": content_buffer[:idx]}
                                inside_think_tag = False
                                json_buffer += content_buffer[idx+8:]
                                content_buffer = ""
                            else:
                                yield {"type": "reasoning", "text": content_buffer}
                                content_buffer = ""
                        else:
                            if seen_think_tag:
                                json_buffer += content_buffer
                                content_buffer = ""

    except Exception as exc:
        logger.exception("[OPENROUTER] Streamed node selection request failed")
        yield {"type": "error", "message": f"Selection stream failed: {exc}"}
        return

    if not json_buffer.strip() and "node_list" in full_content:
        json_buffer = full_content

    try:
        parsed = _parse_model_json(json_buffer)
    except Exception:
        parsed = {}

    node_list = parsed.get("node_list", [])
    valid_ids = {chapter["node_id"] for chapter in chapters}
    node_ids = [n for n in node_list if n in valid_ids][:max_chapters]
    
    if not node_ids:
        node_ids = [chapter["node_id"] for chapter in chapters[:max_chapters]]

    yield {
        "type": "result",
        "node_ids": node_ids,
        "reasoning": "", 
        "assistant_content": full_content,
        "reasoning_details": None,
    }


def _stream_answer_from_retrieval(
    question: str,
    retrieved_nodes: list[dict[str, Any]],
    assistant_content: str | None = None,
    reasoning_details: Any | None = None,
):
    context_blocks: list[str] = []
    total_chars = 0
    for node in retrieved_nodes:
        snippet_parts = []
        for item in node.get("relevant_contents", []):
            snippet_parts.append(f"{item['relevant_content']}")
        chapter_block = "\n".join(snippet_parts) + "\n"
        remaining = MAX_ANSWER_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(chapter_block) > remaining:
            chapter_block = chapter_block[:remaining]
        context_blocks.append(chapter_block)
        total_chars += len(chapter_block)

    system_prompt = (
        "You answer questions only from the provided raw text. "
        "Do not use outside knowledge. "
        "Do NOT cite chapters or use inline citations. Just tell the answer directly looking at the raw text.\n\n"
        "IMPORTANT: You MUST write out your reasoning process first, explicitly enclosed within <think> and </think> tags. "
        "After the closing </think> tag, provide your final direct answer."
    )
    user_prompt = (
        f"""
        Answer the question based on the context:

        Question: {question}
        Context: {context_blocks}

        Provide a clear, concise answer based only on the context provided.
        """
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    if assistant_content is not None or reasoning_details is not None:
        assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
        if reasoning_details is not None:
            assistant_message["reasoning_details"] = reasoning_details
        messages.append(assistant_message)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Continue from the previous reasoning and answer only from the raw text below. "
                    "If the answer is not supported, say so clearly. "
                    "Do NOT cite chapters or use inline citations. Just tell the answer directly.\n\n"
                    + user_prompt
                ),
            }
        )
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
        "stream": True,
        "include_reasoning": True,
        "extra_body": {
            "reasoning": {
                "enabled": True
            }
        }
    }

    api_key = settings.OPENROUTER_API_KEY.strip()
    api_url = "https://api.featherless.ai/v1/chat/completions"
    timeout = settings.OPENROUTER_TIMEOUT_SECONDS

    if not api_key:
        yield _sse("error", {"message": "OPENROUTER_API_KEY is not configured."})
        return

    logger.info("[OPENROUTER] Starting streamed answer request with %s", settings.OPENROUTER_MODEL)
    yield _sse("log", {"stage": "answer", "message": "llm_stream_start"})

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    
                    reasoning = delta.get("reasoning")
                    if reasoning:
                        yield _sse("reasoning", {"text": reasoning, "model": payload["model"], "provider": "openrouter"})
                        
                    content = delta.get("content")
                    if content:
                        yield _sse("delta", {"text": content})
    except httpx.HTTPError as exc:
        logger.exception("[OPENROUTER] Streamed answer request failed")
        yield _sse("error", {"message": f"Answer stream failed: {exc}"})
        return

    yield _sse("done", {"message": "complete"})


def _post_llm_messages(payload: dict[str, Any], purpose: str) -> dict[str, Any]:
    api_key = settings.OPENROUTER_API_KEY.strip()
    api_url = "https://api.featherless.ai/v1/chat/completions"
    timeout = settings.OPENROUTER_TIMEOUT_SECONDS

    if not api_key:
        raise TreeRoutingError("OPENROUTER_API_KEY is not configured.")

    response = None
    last_error: Exception | None = None

    for attempt in range(MAX_LLM_RETRIES):
        try:
            logger.info("[OPENROUTER] %s attempt %s/%s", purpose, attempt + 1, MAX_LLM_RETRIES)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                break
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code == 429 and attempt < MAX_LLM_RETRIES - 1:
                wait_seconds = _extract_retry_delay_seconds(exc.response.text)
                logger.warning(
                    "[RETRY] API rate limited during %s. Waiting %ss",
                    purpose,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            raise TreeRoutingError(
                f"OpenRouter request failed during {purpose}: {exc}. Response: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < MAX_LLM_RETRIES - 1:
                logger.warning("[RETRY] OpenRouter HTTP error during %s. Waiting 2s", purpose)
                time.sleep(2)
                continue
            raise TreeRoutingError(f"OpenRouter request failed during {purpose}: {exc}") from exc

    if response is None:
        raise TreeRoutingError(
            f"OpenRouter request failed during {purpose} after retries: {last_error}"
        )

    try:
        return response.json()["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise TreeRoutingError(
            f"OpenRouter response for {purpose} did not contain message content."
        ) from exc


def _parse_model_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise TreeRoutingError("Model response was not valid JSON content.")
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise TreeRoutingError("Model response was not valid JSON content.") from exc


def _extract_retry_delay_seconds(response_text: str) -> int:
    match = re.search(r"try again in\s+([0-9.]+)s", response_text, re.IGNORECASE)
    if not match:
        return 6
    try:
        return max(1, int(float(match.group(1)) + 1))
    except ValueError:
        return 6
