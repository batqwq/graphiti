"""OpenRouter embedder using the OpenAI-compatible embeddings endpoint."""

import logging
from collections.abc import Iterable

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from openai import AsyncOpenAI
from pydantic import Field

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_OPENROUTER_EMBEDDING_MODEL = 'google/gemini-embedding-2-preview'
DEFAULT_OPENROUTER_EMBEDDING_DIM = 1536
DEFAULT_OPENROUTER_BATCH_SIZE = 1


class OpenRouterEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_OPENROUTER_EMBEDDING_MODEL
    api_key: str
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    batch_size: int = Field(default=DEFAULT_OPENROUTER_BATCH_SIZE, ge=1)


class OpenRouterEmbedder(EmbedderClient):
    """Embedding client for OpenRouter's OpenAI-compatible embeddings API."""

    def __init__(
        self,
        config: OpenRouterEmbedderConfig,
        client: AsyncOpenAI | None = None,
    ):
        self.config = config
        self.batch_size = max(1, config.batch_size)
        self.client = client or AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def _create_embeddings(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ):
        return await self.client.embeddings.create(
            input=input_data,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dim,
        )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self._create_embeddings(input_data)

        if not result.data:
            raise ValueError('No embeddings returned from OpenRouter API')

        return result.data[0].embedding

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []

        embeddings: list[list[float]] = []

        for i in range(0, len(input_data_list), self.batch_size):
            batch = input_data_list[i : i + self.batch_size]

            try:
                result = await self._create_embeddings(batch)
                ordered_embeddings = [
                    item.embedding for item in sorted(result.data, key=lambda item: item.index)
                ]

                if len(ordered_embeddings) != len(batch):
                    raise ValueError(
                        f'OpenRouter returned {len(ordered_embeddings)} embeddings for '
                        f'{len(batch)} inputs'
                    )

                embeddings.extend(ordered_embeddings)
            except Exception as batch_error:
                if len(batch) == 1:
                    raise

                logger.warning(
                    'OpenRouter batch embedding failed, retrying inputs individually: %s',
                    batch_error,
                )
                for item in batch:
                    single_result = await self._create_embeddings(item)
                    if not single_result.data:
                        raise ValueError(
                            'No embedding returned from OpenRouter API'
                        ) from batch_error
                    embeddings.append(single_result.data[0].embedding)

        return embeddings
