"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
else:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            'google-genai is required for GeminiEmbedder. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'text-embedding-001'  # gemini-embedding-001 or text-embedding-005

DEFAULT_BATCH_SIZE = 100

# Retry configuration for embedding API calls
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds

T = TypeVar('T')


async def _retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> T:
    """Execute an async function with exponential backoff retry on transient errors.

    Retries on rate-limit (429), server errors (500/502/503), timeouts, and
    quota-exceeded responses. Non-retriable errors are raised immediately.

    Args:
        func: An async callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).

    Returns:
        The result of the async callable.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            is_retriable = any(
                keyword in error_str
                for keyword in (
                    '429', 'rate', 'quota', 'resource_exhausted',
                    '500', '502', '503',
                    'timeout', 'unavailable', 'deadline',
                    'connection', 'reset',
                )
            )
            if not is_retriable or attempt == max_retries:
                raise
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                f'Embedding API error (attempt {attempt + 1}/{max_retries + 1}), '
                f'retrying in {delay:.1f}s: {e}'
            )
            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    assert last_exception is not None
    raise last_exception


class GeminiEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class GeminiEmbedder(EmbedderClient):
    """
    Google Gemini Embedder Client
    """

    def __init__(
        self,
        config: GeminiEmbedderConfig | None = None,
        client: 'genai.Client | None' = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the GeminiEmbedder with the provided configuration and client.

        Args:
            config (GeminiEmbedderConfig | None): The configuration for the GeminiEmbedder, including API key, model, base URL, temperature, and max tokens.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
            batch_size (int | None): An optional batch size to use. If not provided, the default batch size will be used.
        """
        if config is None:
            config = GeminiEmbedderConfig()

        self.config = config

        if client is None:
            self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client

        if batch_size is None and self.config.embedding_model == 'gemini-embedding-001':
            # Gemini API has a limit on the number of instances per request
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
            self.batch_size = 1
        elif batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data using Google's Gemini embedding model.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Capture variables for the closure
        model = self.config.embedding_model or DEFAULT_EMBEDDING_MODEL
        dim = self.config.embedding_dim

        async def _call():
            return await self.client.aio.models.embed_content(
                model=model,
                contents=[input_data],  # type: ignore[arg-type]  # mypy fails on broad union type
                config=types.EmbedContentConfig(output_dimensionality=dim),
            )

        # Generate embeddings with retry
        result = await _retry_with_backoff(_call)

        if not result.embeddings or len(result.embeddings) == 0 or not result.embeddings[0].values:
            raise ValueError('No embeddings returned from Gemini API in create()')

        return result.embeddings[0].values

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of input data using Google's Gemini embedding model.

        This method handles batching to respect the Gemini API's limits on the number
        of instances that can be processed in a single request.

        Args:
            input_data_list: A list of strings to create embeddings for.

        Returns:
            A list of embedding vectors (each vector is a list of floats).
        """
        if not input_data_list:
            return []

        batch_size = self.batch_size
        all_embeddings = []
        model = self.config.embedding_model or DEFAULT_EMBEDDING_MODEL
        dim = self.config.embedding_dim

        # Process inputs in batches
        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i : i + batch_size]

            try:
                # Capture batch in closure for retry
                current_batch = batch

                async def _batch_call(current_batch=current_batch):
                    return await self.client.aio.models.embed_content(
                        model=model,
                        contents=current_batch,  # type: ignore[arg-type]  # mypy fails on broad union type
                        config=types.EmbedContentConfig(
                            output_dimensionality=dim
                        ),
                    )

                # Generate embeddings for this batch with retry
                result = await _retry_with_backoff(_batch_call)

                if not result.embeddings or len(result.embeddings) == 0:
                    raise Exception('No embeddings returned')

                # Process embeddings from this batch
                for embedding in result.embeddings:
                    if not embedding.values:
                        raise ValueError('Empty embedding values returned')
                    all_embeddings.append(embedding.values)

            except Exception as e:
                # If batch processing fails, fall back to individual processing
                logger.warning(
                    f'Batch embedding failed for batch {i // batch_size + 1}, falling back to individual processing: {e}'
                )

                for item in batch:
                    try:
                        # Capture item in closure for retry
                        current_item = item

                        async def _single_call(current_item=current_item):
                            return await self.client.aio.models.embed_content(
                                model=model,
                                contents=[current_item],  # type: ignore[arg-type]  # mypy fails on broad union type
                                config=types.EmbedContentConfig(
                                    output_dimensionality=dim
                                ),
                            )

                        # Process each item individually with retry
                        result = await _retry_with_backoff(_single_call)

                        if not result.embeddings or len(result.embeddings) == 0:
                            raise ValueError('No embeddings returned from Gemini API')
                        if not result.embeddings[0].values:
                            raise ValueError('Empty embedding values returned')

                        all_embeddings.append(result.embeddings[0].values)

                    except Exception as individual_error:
                        logger.error(f'Failed to embed individual item: {individual_error}')
                        raise individual_error

        return all_embeddings
