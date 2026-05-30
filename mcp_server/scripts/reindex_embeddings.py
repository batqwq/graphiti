#!/usr/bin/env python3
"""Re-index Neo4j vector properties to a new embedding dimension.

Safety:
- Requires neo4j-admin dump backup to exist before execution.
- Writes to new fields (name_embedding_4096, fact_embedding_4096) — never overwrites originals.
- Validates every embedding dimension before writing.
- Tracks progress in migration_progress.json for crash-resume.
- Skips NULL/empty source text.
- Uses per-batch commits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("reindex")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# OpenRouter embedder (minimal, no graphiti-core dependency)
# ---------------------------------------------------------------------------

class OpenRouterBatchEmbedder:
    """Call OpenRouter embeddings endpoint with retry + backoff."""

    MAX_RETRIES = 3
    BASE_DELAY = 1.0
    RATE_LIMIT_DELAY = 30.0

    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import random

        import httpx

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": self.model,
                            "input": texts,
                        },
                    )
                    if resp.status_code == 429:
                        raise RuntimeError(f"429 rate limit: {resp.text[:300]}")
                    if resp.status_code >= 500:
                        raise RuntimeError(f"5xx server error: {resp.status_code}")
                    resp.raise_for_status()
                    data = resp.json()
                    return [
                        d["embedding"] for d in data["data"]
                    ]
            except Exception as exc:
                last_exc = exc
                is_429 = "429" in str(exc)
                if attempt >= self.MAX_RETRIES:
                    raise
                delay = (
                    self.RATE_LIMIT_DELAY + random.uniform(0, 5)
                    if is_429
                    else self.BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                )
                logger.warning(
                    "Embed batch failed (attempt %d/%d, %.1fs backoff): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class Progress:
    entity_offset: int = 0
    community_offset: int = 0
    edge_offset: int = 0

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "entity_offset": self.entity_offset,
                    "community_offset": self.community_offset,
                    "edge_offset": self.edge_offset,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Progress":
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------


async def _migrate_batch(
    driver: Any,
    embedder: OpenRouterBatchEmbedder,
    label: str,
    read_query: str,
    write_query: str,
    batch_size: int,
    offset: int,
) -> tuple[int, bool]:
    """Read one batch, embed, and write via UNWIND in a single transaction.

    Returns (rows_written, has_more).  Raises on dimension mismatch.
    """
    records, _summary, _keys = await driver.execute_query(
        read_query, skip=offset, limit=batch_size,
    )
    if not records:
        return 0, False

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    logger.info("%s batch: offset=%d count=%d", label, offset, len(texts))

    # Embed all texts in one batch call
    embeddings = await embedder.embed_batch(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"{label}: got {len(embeddings)} embeddings for {len(texts)} texts"
        )

    # Validate every embedding dimension before writing
    items: list[dict[str, Any]] = []
    for i, emb in enumerate(embeddings):
        if len(emb) != embedder.dim:
            raise RuntimeError(
                f"{label} elementId={ids[i]!r} text='{texts[i][:80]}': "
                f"dim={len(emb)}, expected {embedder.dim}"
            )
        items.append({"id": ids[i], "emb": emb})

    # UNWIND write in a single transaction, verify updated count
    result, _summary, _keys = await driver.execute_query(
        write_query, items=items,
    )
    updated = result[0]["updated_count"] if result else 0
    if updated != len(items):
        raise RuntimeError(
            f"{label}: UNWIND count mismatch: updated {updated}, expected {len(items)}"
        )

    has_more = len(records) >= batch_size
    logger.info(
        "%s batch done: offset=%d wrote=%d updated=%d has_more=%s",
        label, offset, len(items), updated, has_more,
    )
    return updated, has_more


# --- Cypher queries ---

ENTITY_READ = """\
MATCH (n:Entity)
WHERE n.name IS NOT NULL AND n.name <> ""
RETURN elementId(n) AS id, n.name AS text
ORDER BY elementId(n)
SKIP $skip LIMIT $limit
"""

ENTITY_WRITE = """\
UNWIND $items AS item
MATCH (n:Entity)
WHERE elementId(n) = item.id
SET n.name_embedding_4096 = item.emb
RETURN count(n) AS updated_count
"""

COMMUNITY_READ = """\
MATCH (c:Community)
WHERE c.name IS NOT NULL AND c.name <> ""
RETURN elementId(c) AS id, c.name AS text
ORDER BY elementId(c)
SKIP $skip LIMIT $limit
"""

COMMUNITY_WRITE = """\
UNWIND $items AS item
MATCH (c:Community)
WHERE elementId(c) = item.id
SET c.name_embedding_4096 = item.emb
RETURN count(c) AS updated_count
"""

EDGE_READ = """\
MATCH ()-[e:RELATES_TO]->()
WHERE e.fact IS NOT NULL AND e.fact <> ""
RETURN elementId(e) AS id, e.fact AS text
ORDER BY elementId(e)
SKIP $skip LIMIT $limit
"""

EDGE_WRITE = """\
UNWIND $items AS item
MATCH ()-[e:RELATES_TO]->()
WHERE elementId(e) = item.id
SET e.fact_embedding_4096 = item.emb
RETURN count(e) AS updated_count
"""


async def _migrate_loop(
    driver: Any,
    embedder: OpenRouterBatchEmbedder,
    label: str,
    read_query: str,
    write_query: str,
    batch_size: int,
    progress: Progress,
    offset_attr: str,
) -> int:
    """Run read→embed→UNWIND-write loop with progress tracking."""
    total = 0
    while True:
        offset = getattr(progress, offset_attr)
        wrote, has_more = await _migrate_batch(
            driver, embedder, label, read_query, write_query,
            batch_size, offset,
        )
        if wrote == 0 and not has_more:
            break
        total += wrote
        setattr(progress, offset_attr, offset + wrote)
        progress.save(PROGRESS_FILE)
    logger.info("%s migration complete: %d total", label, total)
    return total


PROGRESS_FILE = Path(__file__).parent / "migration_progress.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index Neo4j embeddings")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", required=True)
    parser.add_argument("--neo4j-password", required=True)
    parser.add_argument("--openrouter-key", required=True)
    parser.add_argument(
        "--openrouter-model", default="qwen/qwen3-embedding-8b"
    )
    parser.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--backup-file", required=True)
    args = parser.parse_args()

    # --- Safety check: backup file exists and is non-empty ---
    backup_path = Path(args.backup_file)
    if not backup_path.exists():
        logger.error("Backup file not found: %s", backup_path)
        sys.exit(1)
    if backup_path.stat().st_size == 0:
        logger.error("Backup file is empty: %s", backup_path)
        sys.exit(1)
    logger.info("Backup verified: %s (%d bytes)", backup_path, backup_path.stat().st_size)

    # --- Connect to Neo4j ---
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    await driver.verify_connectivity()
    logger.info("Neo4j connected: %s", args.neo4j_uri)

    # --- Create embedder ---
    embedder = OpenRouterBatchEmbedder(
        api_key=args.openrouter_key,
        base_url=args.openrouter_base_url,
        model=args.openrouter_model,
        dim=args.embedding_dim,
    )
    logger.info("Embedder: %s (dim=%d)", args.openrouter_model, args.embedding_dim)

    # --- Load progress (crash-resume) ---
    progress = Progress.load(PROGRESS_FILE)
    logger.info(
        "Resuming from: entities=%d, communities=%d, edges=%d",
        progress.entity_offset,
        progress.community_offset,
        progress.edge_offset,
    )

    try:
        # --- Entities ---
        entity_count = await _migrate_loop(
            driver, embedder, "Entity",
            ENTITY_READ, ENTITY_WRITE,
            args.batch_size, progress, "entity_offset",
        )
        logger.info("Entities migrated: %d", entity_count)

        # --- Communities ---
        community_count = await _migrate_loop(
            driver, embedder, "Community",
            COMMUNITY_READ, COMMUNITY_WRITE,
            args.batch_size, progress, "community_offset",
        )
        logger.info("Communities migrated: %d", community_count)

        # --- Edges ---
        edge_count = await _migrate_loop(
            driver, embedder, "Edge",
            EDGE_READ, EDGE_WRITE,
            args.batch_size, progress, "edge_offset",
        )
        logger.info("Edges migrated: %d", edge_count)

        logger.info(
            "=== MIGRATION COMPLETE === entities=%d communities=%d edges=%d",
            entity_count,
            community_count,
            edge_count,
        )
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
