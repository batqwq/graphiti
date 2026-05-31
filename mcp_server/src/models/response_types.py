"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None
    summary: str | None
    group_id: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class QueueGroupStatus(TypedDict):
    pending: int
    processing: int
    completed: int
    failed: int
    worker_running: bool
    idle: bool
    last_started_at: str | None
    last_finished_at: str | None
    last_error: str | None


class QueueStatusResponse(TypedDict):
    status: str
    total_pending: int
    total_processing: int
    total_completed: int
    total_failed: int
    groups: dict[str, QueueGroupStatus]
