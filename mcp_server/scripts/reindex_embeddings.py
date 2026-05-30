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


async def migrate_entities(
    driver: Any, embedder: OpenRouterBatchEmbedder, batch_size: int, progress: Progress
) -> int:
    """Embed Entity.name → n.name_embedding_4096. Returns total migrated."""
    total = 0
    while True:
        records, _summary, _keys = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.name IS NOT NULL AND n.name <> ""
              AND n.name_embedding_4096 IS NULL
            RETURN n.name AS name
            ORDER BY n.name
            SKIP $skip LIMIT $limit
            """,
            skip=progress.entity_offset,
            limit=batch_size,
        )
        if not records:
            break

        texts = [r["name"] for r in records]
        logger.info(
            "Entity batch: offset=%d count=%d", progress.entity_offset, len(texts)
        )

        embeddings = await embedder.embed_batch(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Mismatch: got {len(embeddings)} embeddings for {len(texts)} texts"
            )

        for i, emb in enumerate(embeddings):
            if len(emb) != embedder.dim:
                raise RuntimeError(
                    f"Entity '{texts[i][:80]}': embedding dim={len(emb)}, expected {embedder.dim}"
                )
            await driver.execute_query(
                """
                MATCH (n:Entity {name: $name})
                SET n.name_embedding_4096 = $emb
                """,
                name=texts[i],
                emb=emb,
            )

        total += len(texts)
        progress.entity_offset += len(texts)
        progress.save(PROGRESS_FILE)
        logger.info("Entity batch done: %d total migrated", total)

    logger.info("Entity migration complete: %d total", total)
    return total


async def migrate_communities(
    driver: Any, embedder: OpenRouterBatchEmbedder, batch_size: int, progress: Progress
) -> int:
    """Embed Community.name → c.name_embedding_4096."""
    total = 0
    while True:
        records, _summary, _keys = await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.name IS NOT NULL AND c.name <> ""
              AND c.name_embedding_4096 IS NULL
            RETURN c.name AS name
            ORDER BY c.name
            SKIP $skip LIMIT $limit
            """,
            skip=progress.community_offset,
            limit=batch_size,
        )
        if not records:
            break

        texts = [r["name"] for r in records]
        logger.info(
            "Community batch: offset=%d count=%d",
            progress.community_offset,
            len(texts),
        )

        embeddings = await embedder.embed_batch(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Community mismatch: {len(embeddings)} emb for {len(texts)} texts"
            )

        for i, emb in enumerate(embeddings):
            if len(emb) != embedder.dim:
                raise RuntimeError(
                    f"Community '{texts[i][:80]}': dim={len(emb)}, expected {embedder.dim}"
                )
            await driver.execute_query(
                """
                MATCH (c:Community {name: $name})
                SET c.name_embedding_4096 = $emb
                """,
                name=texts[i],
                emb=emb,
            )

        total += len(texts)
        progress.community_offset += len(texts)
        progress.save(PROGRESS_FILE)
        logger.info("Community batch done: %d total migrated", total)

    logger.info("Community migration complete: %d total", total)
    return total


async def migrate_edges(
    driver: Any, embedder: OpenRouterBatchEmbedder, batch_size: int, progress: Progress
) -> int:
    """Embed RELATES_TO.fact → e.fact_embedding_4096 (directed match)."""
    total = 0
    while True:
        records, _summary, _keys = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.fact IS NOT NULL AND e.fact <> ""
              AND e.fact_embedding_4096 IS NULL
            RETURN e.fact AS fact
            ORDER BY e.fact
            SKIP $skip LIMIT $limit
            """,
            skip=progress.edge_offset,
            limit=batch_size,
        )
        if not records:
            break

        texts = [r["fact"] for r in records]
        logger.info(
            "Edge batch: offset=%d count=%d", progress.edge_offset, len(texts)
        )

        embeddings = await embedder.embed_batch(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Edge mismatch: {len(embeddings)} emb for {len(texts)} texts"
            )

        for i, emb in enumerate(embeddings):
            if len(emb) != embedder.dim:
                raise RuntimeError(
                    f"Edge fact '{texts[i][:80]}': dim={len(emb)}, expected {embedder.dim}"
                )
            await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->()
                WHERE e.fact = $fact
                SET e.fact_embedding_4096 = $emb
                """,
                fact=texts[i],
                emb=emb,
            )

        total += len(texts)
        progress.edge_offset += len(texts)
        progress.save(PROGRESS_FILE)
        logger.info("Edge batch done: %d total migrated", total)

    logger.info("Edge migration complete: %d total", total)
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
        entity_count = await migrate_entities(
            driver, embedder, args.batch_size, progress
        )
        logger.info("Entities migrated: %d", entity_count)

        # --- Communities ---
        community_count = await migrate_communities(
            driver, embedder, args.batch_size, progress
        )
        logger.info("Communities migrated: %d", community_count)

        # --- Edges ---
        edge_count = await migrate_edges(
            driver, embedder, args.batch_size, progress
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
