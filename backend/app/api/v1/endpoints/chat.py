from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnswerGenerationRequest,
    AnswerGenerationResponse,
    NodeSelectionRequest,
    NodeSelectionResponse,
)
from app.services.rag_service import (
    TreeRoutingError,
    generate_answer_from_node_ids,
    generate_answer_stream_from_node_ids,
    load_tree_from_path,
    select_nodes_stream_from_tree,
    select_relevant_chapters,
)

router = APIRouter()


def _streaming_response(event_iter):
    return StreamingResponse(
        event_iter,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/select-nodes", response_model=NodeSelectionResponse)
def select_nodes(payload: NodeSelectionRequest) -> NodeSelectionResponse:
    try:
        tree = payload.tree or load_tree_from_path(payload.tree_path or "data/hp1_pageindex_tree.json")
        result = select_relevant_chapters(
            question=payload.question,
            tree=tree,
            max_chapters=payload.max_nodes,
        )
        return NodeSelectionResponse(
            question=result["question"],
            selected_node_ids=result["selected_node_ids"],
            model=result["model"],
            provider=result["provider"],
            reasoning=result.get("reasoning"),
            assistant_content=result.get("assistant_content"),
            reasoning_details=result.get("reasoning_details"),
        )
    except TreeRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/select-nodes/stream")
def select_nodes_stream(payload: NodeSelectionRequest):
    try:
        tree = payload.tree or load_tree_from_path(payload.tree_path or "data/hp1_pageindex_tree.json")
        event_iter = select_nodes_stream_from_tree(
            question=payload.question,
            tree=tree,
            max_nodes=payload.max_nodes,
        )
        return _streaming_response(event_iter)
    except TreeRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate-answer", response_model=AnswerGenerationResponse)
def generate_answer(payload: AnswerGenerationRequest) -> AnswerGenerationResponse:
    try:
        tree = payload.tree or load_tree_from_path(payload.tree_path or "data/hp1_pageindex_tree.json")
        result = generate_answer_from_node_ids(
            question=payload.question,
            tree=tree,
            node_ids=payload.node_ids,
            assistant_content=payload.assistant_content,
            reasoning_details=payload.reasoning_details,
        )
        return AnswerGenerationResponse(**result)
    except TreeRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate-answer/stream")
def generate_answer_stream(payload: AnswerGenerationRequest):
    try:
        tree = payload.tree or load_tree_from_path(payload.tree_path or "data/hp1_pageindex_tree.json")
        event_iter = generate_answer_stream_from_node_ids(
            question=payload.question,
            tree=tree,
            node_ids=payload.node_ids,
            assistant_content=payload.assistant_content,
            reasoning_details=payload.reasoning_details,
        )
        return _streaming_response(event_iter)
    except TreeRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
