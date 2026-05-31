"""Episode query helpers for MCP output."""

from typing import Any, Literal

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_episodic_node_from_record,
)

EpisodeSortOrder = Literal['asc', 'desc']


def normalize_episode_sort_order(sort_order: str) -> EpisodeSortOrder:
    """Validate and normalize episode created_at sort order."""
    normalized = sort_order.strip().lower()
    if normalized == 'asc':
        return 'asc'
    if normalized == 'desc':
        return 'desc'
    raise ValueError("sort_order must be 'asc' or 'desc'")


def resolve_episode_limit(
    *,
    max_episodes: int | None,
    limit: int | None,
    last_n: int | None,
    default: int,
) -> int:
    """Resolve current and legacy episode limit argument names."""
    provided_limits = [
        value
        for value in (
            max_episodes,
            limit,
            last_n,
        )
        if value is not None
    ]

    if not provided_limits:
        return default

    first_limit = provided_limits[0]
    if any(value != first_limit for value in provided_limits[1:]):
        raise ValueError('max_episodes, limit, and last_n cannot disagree')

    return first_limit


async def get_episodes_by_created_at(
    driver: Any,
    group_ids: list[str],
    *,
    limit: int,
    offset: int,
    sort_order: EpisodeSortOrder,
) -> list[Any]:
    """Return episodes ordered by created_at with offset pagination."""
    if offset < 0:
        raise ValueError('offset must be zero or a positive integer')

    sort_direction = sort_order.upper()
    return_clause = (
        EPISODIC_NODE_RETURN_NEPTUNE
        if driver.provider == GraphProvider.NEPTUNE
        else EPISODIC_NODE_RETURN
    )

    records, _, _ = await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.group_id IN $group_ids
        RETURN DISTINCT
        """
        + return_clause
        + f"""
        ORDER BY created_at {sort_direction}, uuid {sort_direction}
        SKIP $offset
        LIMIT $limit
        """,
        group_ids=group_ids,
        offset=offset,
        limit=limit,
        routing_='r',
    )

    return [get_episodic_node_from_record(record) for record in records]
