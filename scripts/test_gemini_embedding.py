"""Smoke test for Google Gemini Embedding 2."""

import asyncio
import os
from pathlib import Path

from dotenv import dotenv_values

from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

DEFAULT_MODEL = 'gemini-embedding-2'
DEFAULT_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 1

TEST_TEXTS = [
    '锅盖正在把 AI 记忆迁移到远程 MCP。',
    'Graphiti 用于构建带时间感的知识图谱记忆。',
    'Gemini Embedding 2 通过 Google Gemini API 提供向量。',
]


def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for env_file in (repo_root / '.env', repo_root / 'mcp_server' / '.env'):
        if not env_file.exists():
            continue

        for key, value in dotenv_values(env_file).items():
            if value:
                os.environ[key] = value


async def main() -> None:
    load_env()

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY is not set in environment or .env')

    model = os.getenv('GRAPHITI_EMBEDDING_MODEL', DEFAULT_MODEL)
    dimensions = int(os.getenv('GRAPHITI_EMBEDDING_DIM', DEFAULT_DIMENSIONS))
    batch_size = max(1, int(os.getenv('GRAPHITI_EMBEDDING_BATCH_SIZE', DEFAULT_BATCH_SIZE)))

    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model=model,
            embedding_dim=dimensions,
        ),
        batch_size=batch_size,
    )
    embeddings = await embedder.create_batch(TEST_TEXTS)

    assert len(embeddings) == len(TEST_TEXTS), (
        f'Expected {len(TEST_TEXTS)} embeddings, got {len(embeddings)}'
    )
    for index, embedding in enumerate(embeddings, start=1):
        assert len(embedding) == dimensions, (
            f'Embedding {index} has {len(embedding)} dimensions, expected {dimensions}'
        )

    print(
        f'Gemini embedding test passed: {len(embeddings)} embeddings, {dimensions} dimensions each.'
    )


if __name__ == '__main__':
    asyncio.run(main())
