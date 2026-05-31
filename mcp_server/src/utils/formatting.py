"""Formatting utilities for Graphiti MCP Server."""

import os
from collections.abc import Mapping, Sequence
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    return max(value, minimum)


DEFAULT_MAX_TEXT_CHARS = _env_int('MCP_OUTPUT_MAX_TEXT_CHARS', 1200)
DEFAULT_MAX_ATTRIBUTE_CHARS = _env_int('MCP_OUTPUT_MAX_ATTRIBUTE_CHARS', 800)
DEFAULT_MAX_COLLECTION_ITEMS = _env_int('MCP_OUTPUT_MAX_COLLECTION_ITEMS', 20, minimum=1)
DEFAULT_MAX_NESTING_DEPTH = _env_int('MCP_OUTPUT_MAX_NESTING_DEPTH', 3, minimum=1)


def truncate_text(text: str | None, max_chars: int | None = None) -> str | None:
    """Return a bounded string with a clear truncation marker."""
    if text is None:
        return None

    limit = DEFAULT_MAX_TEXT_CHARS if max_chars is None else max(0, max_chars)
    if limit == 0:
        return ''
    if len(text) <= limit:
        return text

    omitted = len(text)
    while True:
        suffix = f'... [truncated {omitted} chars]'
        prefix_limit = limit - len(suffix)
        if prefix_limit <= 0:
            return suffix[:limit]

        new_omitted = len(text) - prefix_limit
        if new_omitted == omitted:
            break
        omitted = new_omitted

    return f'{text[:prefix_limit].rstrip()}{suffix}'


def compact_value(
    value: Any,
    *,
    max_text_chars: int = DEFAULT_MAX_ATTRIBUTE_CHARS,
    max_collection_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
    max_depth: int = DEFAULT_MAX_NESTING_DEPTH,
    _depth: int = 0,
) -> Any:
    """Recursively bound large strings and collections for MCP tool output."""
    if isinstance(value, str):
        return truncate_text(value, max_text_chars)

    if value is None or isinstance(value, bool | int | float):
        return value

    if _depth >= max_depth:
        return truncate_text(str(value), max_text_chars)

    if isinstance(value, Mapping):
        compacted: dict[str, Any] = {}
        items = [
            (key, item_value)
            for key, item_value in value.items()
            if 'embedding' not in str(key).lower()
        ]

        for key, item_value in items[:max_collection_items]:
            key_string = str(key)
            compacted[key_string] = compact_value(
                item_value,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
                max_depth=max_depth,
                _depth=_depth + 1,
            )

        omitted = max(0, len(items) - max_collection_items)
        if omitted:
            compacted['_truncated_keys'] = omitted

        return compacted

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        compacted_items = [
            compact_value(
                item,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value[:max_collection_items]
        ]
        omitted = max(0, len(value) - max_collection_items)
        if omitted:
            compacted_items.append({'_truncated_items': omitted})
        return compacted_items

    return truncate_text(str(value), max_text_chars)


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """Format an entity node into a minimal result (uuid, name, summary only)."""
    return {
        'uuid': node.uuid,
        'name': node.name,
        'summary': node.summary,
    }


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a minimal result (uuid, name, fact only)."""
    return {
        'uuid': edge.uuid,
        'name': edge.name,
        'fact': edge.fact,
    }


def format_episode_result(
    episode: Any,
    *,
    content_max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> dict[str, Any]:
    """Format an episode without returning unbounded episode content by default."""
    content = episode.content or ''
    effective_content_limit = min(max(0, content_max_chars), DEFAULT_MAX_TEXT_CHARS)
    formatted_content = truncate_text(content, effective_content_limit)

    return {
        'uuid': episode.uuid,
        'name': episode.name,
        'content': formatted_content,
        'content_length': len(content),
        'content_truncated': formatted_content != content,
        'created_at': episode.created_at.isoformat() if episode.created_at else None,
        'source': episode.source.value if hasattr(episode.source, 'value') else str(episode.source),
        'source_description': truncate_text(
            episode.source_description, DEFAULT_MAX_ATTRIBUTE_CHARS
        ),
        'group_id': episode.group_id,
    }
