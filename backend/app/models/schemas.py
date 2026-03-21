from typing import Any

from pydantic import BaseModel, Field


class NodeSelectionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    tree_path: str | None = Field(default="data/hp1_pageindex_tree.json")
    tree: dict[str, Any] | None = None
    max_nodes: int = Field(default=3, ge=1, le=3)


class AnswerGenerationRequest(BaseModel):
    question: str = Field(..., min_length=3)
    node_ids: list[str] = Field(..., min_length=1)
    tree_path: str | None = Field(default="data/hp1_pageindex_tree.json")
    tree: dict[str, Any] | None = None
    assistant_content: str | None = None
    reasoning_details: Any | None = None
