from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnswerGenerationRequest,
    NodeSelectionRequest,
)
from app.services.rag_service import (
    TreeRoutingError,
    generate_answer_stream_from_node_ids,
    load_tree_from_path,
    select_nodes_stream_from_tree,
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
