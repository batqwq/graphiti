#!/usr/bin/env python3
"""Re-index Neo4j vector properties to a new embedding dimension.

Safety:
- Requires a non-empty neo4j-admin dump backup file before execution.
- Preserves existing runtime embeddings in backup properties before replacing them.
- Validates every embedding dimension before writing.
- Tracks progress in migration_progress.json for crash-resume.
- Skips NULL/empty source text.
- Uses one committed UNWIND write per batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger('reindex')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

PROPERTY_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _ordered_response_embeddings(data: dict[str, Any], expected_count: int) -> list[list[float]]:
    response_items = data['data']
    if len(response_items) != expected_count:
        raise RuntimeError(
            f'OpenRouter returned {len(response_items)} embeddings for {expected_count} inputs'
        )
    if all('index' in item for item in response_items):
        response_items = sorted(response_items, key=lambda item: item['index'])
    return [item['embedding'] for item in response_items]


class OpenRouterBatchEmbedder:
    """Call OpenRouter embeddings endpoint with retry and backoff."""

    MAX_RETRIES = 3
    BASE_DELAY = 1.0
    RATE_LIMIT_DELAY = 30.0

    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.dim = dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import random

        import httpx

        url = f'{self.base_url}/embeddings'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        json={
                            'model': self.model,
                            'input': texts,
                            'dimensions': self.dim,
                        },
                    )
                    if resp.status_code == 429:
                        raise RuntimeError(f'429 rate limit: {resp.text[:300]}')
                    if resp.status_code >= 500:
                        raise RuntimeError(f'5xx server error: {resp.status_code}')
                    resp.raise_for_status()
                    return _ordered_response_embeddings(resp.json(), len(texts))
            except Exception as exc:
                last_exc = exc
                is_429 = '429' in str(exc)
                if attempt >= self.MAX_RETRIES:
                    raise
                delay = (
                    self.RATE_LIMIT_DELAY + random.uniform(0, 5)
                    if is_429
                    else self.BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                )
                logger.warning(
                    'Embed batch failed (attempt %d/%d, %.1fs backoff): %s',
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


@dataclass
class Progress:
    entity_offset: int = 0
    community_offset: int = 0
    edge_offset: int = 0

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    'entity_offset': self.entity_offset,
                    'community_offset': self.community_offset,
                    'edge_offset': self.edge_offset,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> Progress:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()


async def _migrate_batch(
    driver: Any,
    embedder: OpenRouterBatchEmbedder,
    label: str,
    read_query: str,
    write_query: str,
    batch_size: int,
    offset: int,
) -> tuple[int, bool]:
    """Read one batch, embed it, and write via UNWIND in one transaction."""
    records, _summary, _keys = await driver.execute_query(
        read_query,
        skip=offset,
        limit=batch_size,
    )
    if not records:
        return 0, False

    ids = [r['id'] for r in records]
    texts = [r['text'] for r in records]
    logger.info('%s batch: offset=%d count=%d', label, offset, len(texts))

    embeddings = await embedder.embed_batch(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError(f'{label}: got {len(embeddings)} embeddings for {len(texts)} texts')

    items: list[dict[str, Any]] = []
    for i, emb in enumerate(embeddings):
        if len(emb) != embedder.dim:
            raise RuntimeError(
                f'{label} elementId={ids[i]!r} text={texts[i][:80]!r}: '
                f'dim={len(emb)}, expected {embedder.dim}'
            )
        items.append({'id': ids[i], 'emb': emb})

    result, _summary, _keys = await driver.execute_query(
        write_query,
        items=items,
    )
    updated = result[0]['updated_count'] if result else 0
    if updated != len(items):
        raise RuntimeError(
            f'{label}: UNWIND count mismatch: updated {updated}, expected {len(items)}'
        )

    has_more = len(records) >= batch_size
    logger.info(
        '%s batch done: offset=%d wrote=%d updated=%d has_more=%s',
        label,
        offset,
        len(items),
        updated,
        has_more,
    )
    return updated, has_more


ENTITY_READ = """\
MATCH (n:Entity)
WHERE n.name IS NOT NULL AND n.name <> ""
RETURN elementId(n) AS id, n.name AS text
ORDER BY elementId(n)
SKIP $skip LIMIT $limit
"""

COMMUNITY_READ = """\
MATCH (c:Community)
WHERE c.name IS NOT NULL AND c.name <> ""
RETURN elementId(c) AS id, c.name AS text
ORDER BY elementId(c)
SKIP $skip LIMIT $limit
"""

EDGE_READ = """\
MATCH ()-[e:RELATES_TO]->()
WHERE e.fact IS NOT NULL AND e.fact <> ""
RETURN elementId(e) AS id, e.fact AS text
ORDER BY elementId(e)
SKIP $skip LIMIT $limit
"""


def _validate_property_name(name: str) -> str:
    if not PROPERTY_NAME_RE.fullmatch(name):
        raise ValueError(f'Unsafe Neo4j property name: {name!r}')
    return name


def _normalize_suffix(raw_suffix: str) -> str:
    if not raw_suffix:
        return ''
    return raw_suffix if raw_suffix.startswith('_') else f'_{raw_suffix}'


def _embedding_property(base_name: str, suffix: str) -> str:
    return _validate_property_name(f'{base_name}{suffix}')


def _backup_property(base_name: str, backup_suffix: str) -> str | None:
    if not backup_suffix:
        return None
    return _embedding_property(base_name, backup_suffix)


def _build_write_query(
    match_clause: str,
    variable: str,
    target_property: str,
    backup_property: str | None,
) -> str:
    target_property = _validate_property_name(target_property)
    set_clauses = []
    if backup_property is not None and backup_property != target_property:
        backup_property = _validate_property_name(backup_property)
        set_clauses.append(
            f'{variable}.{backup_property} = '
            f'coalesce({variable}.{backup_property}, {variable}.{target_property})'
        )
    set_clauses.append(f'{variable}.{target_property} = item.emb')

    return f"""\
UNWIND $items AS item
{match_clause}
WHERE elementId({variable}) = item.id
SET {', '.join(set_clauses)}
RETURN count({variable}) AS updated_count
"""


async def _migrate_loop(
    driver: Any,
    embedder: OpenRouterBatchEmbedder,
    label: str,
    read_query: str,
    write_query: str,
    batch_size: int,
    progress: Progress,
    progress_file: Path,
    offset_attr: str,
) -> int:
    """Run read, embed, and UNWIND-write batches with progress tracking."""
    total = 0
    while True:
        offset = getattr(progress, offset_attr)
        wrote, has_more = await _migrate_batch(
            driver,
            embedder,
            label,
            read_query,
            write_query,
            batch_size,
            offset,
        )
        if wrote == 0 and not has_more:
            break
        total += wrote
        setattr(progress, offset_attr, offset + wrote)
        progress.save(progress_file)
    logger.info('%s migration complete: %d total', label, total)
    return total


DEFAULT_PROGRESS_FILE = Path(__file__).parent / 'migration_progress.json'


async def main() -> None:
    parser = argparse.ArgumentParser(description='Re-index Neo4j embeddings')
    parser.add_argument('--neo4j-uri', required=True)
    parser.add_argument('--neo4j-user', required=True)
    parser.add_argument('--neo4j-password', required=True)
    parser.add_argument('--openrouter-key', required=True)
    parser.add_argument('--openrouter-model', default='qwen/qwen3-embedding-8b')
    parser.add_argument(
        '--openrouter-base-url',
        default='https://openrouter.ai/api/v1',
    )
    parser.add_argument('--embedding-dim', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--backup-file', required=True)
    parser.add_argument(
        '--target-suffix',
        default='',
        help=(
            'Suffix appended to runtime embedding properties. The default empty suffix '
            'updates name_embedding and fact_embedding so search uses the new vectors.'
        ),
    )
    parser.add_argument(
        '--preserve-original-suffix',
        default='_before_reindex',
        help=(
            'Suffix used to copy existing runtime embeddings before overwriting them. '
            'Use an empty value only when writing to sidecar fields or after an external backup.'
        ),
    )
    parser.add_argument(
        '--progress-file',
        type=Path,
        default=DEFAULT_PROGRESS_FILE,
        help='Crash-resume progress file path.',
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError('--batch-size must be a positive integer')
    if args.embedding_dim <= 0:
        raise ValueError('--embedding-dim must be a positive integer')

    target_suffix = _normalize_suffix(args.target_suffix)
    preserve_original_suffix = _normalize_suffix(args.preserve_original_suffix)
    name_property = _embedding_property('name_embedding', target_suffix)
    fact_property = _embedding_property('fact_embedding', target_suffix)
    backup_name_property = (
        _backup_property('name_embedding', preserve_original_suffix)
        if target_suffix == ''
        else None
    )
    backup_fact_property = (
        _backup_property('fact_embedding', preserve_original_suffix)
        if target_suffix == ''
        else None
    )
    entity_write = _build_write_query(
        'MATCH (n:Entity)', 'n', name_property, backup_name_property
    )
    community_write = _build_write_query(
        'MATCH (c:Community)', 'c', name_property, backup_name_property
    )
    edge_write = _build_write_query(
        'MATCH ()-[e:RELATES_TO]->()', 'e', fact_property, backup_fact_property
    )

    backup_path = Path(args.backup_file)
    if not backup_path.exists():
        logger.error('Backup file not found: %s', backup_path)
        sys.exit(1)
    if backup_path.stat().st_size == 0:
        logger.error('Backup file is empty: %s', backup_path)
        sys.exit(1)
    logger.info('Backup verified: %s (%d bytes)', backup_path, backup_path.stat().st_size)
    logger.info(
        'Target properties: entity/community=%s edge=%s backup_suffix=%s',
        name_property,
        fact_property,
        preserve_original_suffix or '<disabled>',
    )

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    await driver.verify_connectivity()
    logger.info('Neo4j connected: %s', args.neo4j_uri)

    embedder = OpenRouterBatchEmbedder(
        api_key=args.openrouter_key,
        base_url=args.openrouter_base_url,
        model=args.openrouter_model,
        dim=args.embedding_dim,
    )
    logger.info('Embedder: %s (dim=%d)', args.openrouter_model, args.embedding_dim)

    progress = Progress.load(args.progress_file)
    logger.info(
        'Resuming from: entities=%d, communities=%d, edges=%d',
        progress.entity_offset,
        progress.community_offset,
        progress.edge_offset,
    )

    try:
        entity_count = await _migrate_loop(
            driver,
            embedder,
            'Entity',
            ENTITY_READ,
            entity_write,
            args.batch_size,
            progress,
            args.progress_file,
            'entity_offset',
        )
        logger.info('Entities migrated: %d', entity_count)

        community_count = await _migrate_loop(
            driver,
            embedder,
            'Community',
            COMMUNITY_READ,
            community_write,
            args.batch_size,
            progress,
            args.progress_file,
            'community_offset',
        )
        logger.info('Communities migrated: %d', community_count)

        edge_count = await _migrate_loop(
            driver,
            embedder,
            'Edge',
            EDGE_READ,
            edge_write,
            args.batch_size,
            progress,
            args.progress_file,
            'edge_offset',
        )
        logger.info('Edges migrated: %d', edge_count)

        logger.info(
            '=== MIGRATION COMPLETE === entities=%d communities=%d edges=%d',
            entity_count,
            community_count,
            edge_count,
        )
    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
