"""Smoke test for OpenRouter Gemini Embedding 2 Preview."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import APIStatusError, AsyncOpenAI

DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'google/gemini-embedding-2-preview'
DEFAULT_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 1

TEST_TEXTS = [
    '锅盖正在把 AI 记忆迁移到远程 MCP。',
    'Graphiti 用于构建带时间感的知识图谱记忆。',
    'Gemini Embedding 2 Preview 通过 OpenRouter 提供向量。',
]


def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    file_values: dict[str, str] = {}

    for env_file in (repo_root / '.env', repo_root / 'mcp_server' / '.env'):
        if not env_file.exists():
            continue

        for key, value in dotenv_values(env_file).items():
            if value:
                file_values[key] = value

    for key, value in file_values.items():
        os.environ.setdefault(key, value)


async def create_embeddings(
    client: AsyncOpenAI,
    texts: list[str],
    model: str,
    dimensions: int,
    batch_size: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        try:
            response = await client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimensions,
            )
            batch_embeddings = [
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            ]

            if len(batch_embeddings) != len(batch):
                raise ValueError(f'Expected {len(batch)} embeddings, got {len(batch_embeddings)}')

            embeddings.extend(batch_embeddings)
        except Exception as batch_error:
            if len(batch) == 1:
                raise

            for text in batch:
                response = await client.embeddings.create(
                    model=model,
                    input=text,
                    dimensions=dimensions,
                )
                if not response.data:
                    raise ValueError('OpenRouter returned no embedding data') from batch_error
                embeddings.append(response.data[0].embedding)

    return embeddings


async def main() -> None:
    load_env()

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY is not set in environment or .env')

    base_url = os.getenv('OPENROUTER_BASE_URL', DEFAULT_BASE_URL)
    if os.getenv('GRAPHITI_EMBEDDING_PROVIDER') == 'openrouter':
        model = os.getenv('GRAPHITI_EMBEDDING_MODEL', DEFAULT_MODEL)
    else:
        model = os.getenv('OPENROUTER_EMBEDDING_MODEL', DEFAULT_MODEL)
    dimensions = int(os.getenv('GRAPHITI_EMBEDDING_DIM', DEFAULT_DIMENSIONS))
    batch_size = max(1, int(os.getenv('GRAPHITI_EMBEDDING_BATCH_SIZE', DEFAULT_BATCH_SIZE)))

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        embeddings = await create_embeddings(client, TEST_TEXTS, model, dimensions, batch_size)
    except APIStatusError as exc:
        raise RuntimeError(
            f'OpenRouter embedding request failed with HTTP {exc.status_code}. '
            'Check model access, provider routing, and provider Terms of Service.'
        ) from None

    assert len(embeddings) == len(TEST_TEXTS), (
        f'Expected {len(TEST_TEXTS)} embeddings, got {len(embeddings)}'
    )
    for index, embedding in enumerate(embeddings, start=1):
        assert len(embedding) == dimensions, (
            f'Embedding {index} has {len(embedding)} dimensions, expected {dimensions}'
        )

    print(
        f'OpenRouter embedding test passed: {len(embeddings)} embeddings, '
        f'{dimensions} dimensions each.'
    )


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError as error:
        print(f'OpenRouter embedding test failed: {error}', file=sys.stderr)
        raise SystemExit(1) from None
