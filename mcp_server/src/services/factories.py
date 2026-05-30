"""Factory classes for creating LLM, Embedder, and Database clients."""

import asyncio
import logging
import random

from config.schema import (
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

logger = logging.getLogger(__name__)

# ---- Embedding retry utilities ----

_RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503, 504}
_RETRYABLE_KEYWORDS: tuple[str, ...] = (
    'rate', 'quota', 'resource_exhausted',
    'timeout', 'unavailable', 'deadline',
    'connection', 'reset',
)
_MAX_RETRIES: int = 3
_BASE_DELAY: float = 1.0
_RATE_LIMIT_DELAY: float = 30.0


def _is_retryable_error(exception: Exception) -> bool:
    """Check whether an embedding error is retryable (429, 5xx, timeout, connection)."""
    error_str = str(exception).lower()
    # Check HTTP status codes
    for status in _RETRYABLE_STATUSES:
        if str(status) in error_str:
            return True
    # Check known error keywords
    for keyword in _RETRYABLE_KEYWORDS:
        if keyword in error_str:
            return True
    # Check for common exception types
    if isinstance(exception, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return True
    # Check for httpx / aiohttp timeouts
    type_name = type(exception).__name__.lower()
    if 'timeout' in type_name or 'connection' in type_name:
        return True
    return False


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check whether an error is specifically a 429 rate limit."""
    error_str = str(exception).lower()
    return '429' in error_str or 'rate' in error_str or 'quota' in error_str


def _backoff_delay(attempt: int, is_rate_limit: bool) -> float:
    """Compute backoff delay: 30s + jitter for 429, exponential for others."""
    if is_rate_limit:
        return _RATE_LIMIT_DELAY + random.uniform(0, 5)
    return _BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)


class RetryableEmbedder:
    """Wraps an EmbedderClient with exponential-backoff retry for transient errors.

    Gemini embedder already has its own retry, so it should NOT be wrapped.
    """

    def __init__(self, inner: 'EmbedderClient') -> None:  # noqa: F821
        self._inner = inner

    async def create(
        self, input_data: str | list[str] | list[int] | list[list[int]]
    ) -> list[float]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._inner.create(input_data)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt >= _MAX_RETRIES:
                    raise
                is_rl = _is_rate_limit_error(exc)
                delay = _backoff_delay(attempt, is_rl)
                logger.warning(
                    'Embedding create failed (attempt %d/%d, %.1fs backoff): %s',
                    attempt + 1, _MAX_RETRIES, delay, exc
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._inner.create_batch(input_data_list)
            except NotImplementedError:
                raise
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt >= _MAX_RETRIES:
                    raise
                is_rl = _is_rate_limit_error(exc)
                delay = _backoff_delay(attempt, is_rl)
                logger.warning(
                    'Embedding create_batch failed (attempt %d/%d, %.1fs backoff): %s',
                    attempt + 1, _MAX_RETRIES, delay, exc
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

try:
    from graphiti_core.driver.kuzu_driver import KuzuDriver  # noqa: F401

    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from services.openrouter_embedder import OpenRouterEmbedder, OpenRouterEmbedderConfig

# Try to import additional providers if available
try:
    from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

    HAS_AZURE_EMBEDDER = True
except ImportError:
    HAS_AZURE_EMBEDDER = False

try:
    from graphiti_core.embedder.gemini import GeminiEmbedder

    HAS_GEMINI_EMBEDDER = True
except ImportError:
    HAS_GEMINI_EMBEDDER = False

try:
    from graphiti_core.embedder.voyage import VoyageAIEmbedder

    HAS_VOYAGE_EMBEDDER = True
except ImportError:
    HAS_VOYAGE_EMBEDDER = False

try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

    HAS_AZURE_LLM = True
except ImportError:
    HAS_AZURE_LLM = False

try:
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from graphiti_core.llm_client.groq_client import GroqClient

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key

    Raises:
        ValueError: If API key is None or empty
    """
    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                # Use the same model for both main and small model slots
                small_model = config.model

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    small_model=small_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Check if this is a reasoning model (o1, o3, gpt-5 family)
                reasoning_prefixes = ('o1', 'o3', 'gpt-5')
                is_reasoning_model = config.model.startswith(reasoning_prefixes)

                # Only pass reasoning/verbosity parameters for reasoning models (gpt-5 family)
                if is_reasoning_model:
                    return OpenAIClient(config=llm_config, reasoning='minimal', verbosity='low')
                else:
                    # For non-reasoning models, explicitly pass None to disable these parameters
                    return OpenAIClient(config=llm_config, reasoning=None, verbosity=None)

            case 'azure_openai':
                if not HAS_AZURE_LLM:
                    raise ValueError(
                        'Azure OpenAI LLM client not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )

                # Then create the LLMConfig
                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                return AzureOpenAILLMClient(
                    azure_client=azure_client,
                    config=llm_config,
                    max_tokens=config.max_tokens,
                )

            case 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ValueError(
                        'Anthropic client not available in current graphiti-core version'
                    )
                if not config.providers.anthropic:
                    raise ValueError('Anthropic provider configuration not found')

                api_key = config.providers.anthropic.api_key
                _validate_api_key('Anthropic', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return AnthropicClient(config=llm_config)

            case 'gemini':
                if not HAS_GEMINI:
                    raise ValueError('Gemini client not available in current graphiti-core version')
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GeminiClient(config=llm_config)

            case 'groq':
                if not HAS_GROQ:
                    raise ValueError('Groq client not available in current graphiti-core version')
                if not config.providers.groq:
                    raise ValueError('Groq provider configuration not found')

                api_key = config.providers.groq.api_key
                _validate_api_key('Groq', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=config.providers.groq.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GroqClient(config=llm_config)

            case 'fireworks':
                if not config.providers.fireworks:
                    raise ValueError('Fireworks provider configuration not found')

                fireworks_config = config.providers.fireworks
                api_key = fireworks_config.api_key
                _validate_api_key('Fireworks', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=fireworks_config.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return OpenAIGenericClient(config=llm_config, max_tokens=config.max_tokens)

            case _:
                raise ValueError(f'Unsupported LLM provider: {provider}')


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create an Embedder client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                    base_url=config.providers.openai.api_url,  # Support custom endpoints like Ollama
                    embedding_dim=config.dimensions,  # Support custom embedding dimensions
                )
                return RetryableEmbedder(OpenAIEmbedder(config=embedder_config))

            case 'openrouter':
                if not config.providers.openrouter:
                    raise ValueError('OpenRouter provider configuration not found')

                openrouter_config = config.providers.openrouter
                api_key = openrouter_config.api_key
                _validate_api_key('OpenRouter Embedder', api_key, logger)

                embedder_config = OpenRouterEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                    base_url=openrouter_config.api_url,
                    embedding_dim=config.dimensions,
                    batch_size=config.batch_size,
                )
                return RetryableEmbedder(OpenRouterEmbedder(config=embedder_config))

            case 'azure_openai':
                if not HAS_AZURE_EMBEDDER:
                    raise ValueError(
                        'Azure OpenAI embedder not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI Embedder', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )

                return RetryableEmbedder(AzureOpenAIEmbedderClient(
                    azure_client=azure_client,
                    model=config.model or 'text-embedding-3-small',
                ))

            case 'gemini':
                if not HAS_GEMINI_EMBEDDER:
                    raise ValueError(
                        'Gemini embedder not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Embedder', api_key, logger)

                from graphiti_core.embedder.gemini import GeminiEmbedderConfig

                gemini_config = GeminiEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'gemini-embedding-2',
                    embedding_dim=config.dimensions or 768,
                )
                return RetryableEmbedder(GeminiEmbedder(config=gemini_config, batch_size=config.batch_size))

            case 'voyage':
                if not HAS_VOYAGE_EMBEDDER:
                    raise ValueError(
                        'Voyage embedder not available in current graphiti-core version'
                    )
                if not config.providers.voyage:
                    raise ValueError('Voyage provider configuration not found')

                api_key = config.providers.voyage.api_key
                _validate_api_key('Voyage Embedder', api_key, logger)

                from graphiti_core.embedder.voyage import VoyageAIEmbedderConfig

                voyage_config = VoyageAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'voyage-3',
                    embedding_dim=config.dimensions or 1024,
                )
                return RetryableEmbedder(VoyageAIEmbedder(config=voyage_config))

            case _:
                raise ValueError(f'Unsupported Embedder provider: {provider}')


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary based on the configured provider."""
        provider = config.provider.lower()

        match provider:
            case 'neo4j':
                # Use Neo4j config if provided, otherwise use defaults
                if config.providers.neo4j:
                    neo4j_config = config.providers.neo4j
                else:
                    # Create default Neo4j configuration
                    from config.schema import Neo4jProviderConfig

                    neo4j_config = Neo4jProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                uri = os.environ.get('NEO4J_URI', neo4j_config.uri)
                username = os.environ.get('NEO4J_USER', neo4j_config.username)
                password = os.environ.get('NEO4J_PASSWORD', neo4j_config.password)

                return {
                    'uri': uri,
                    'user': username,
                    'password': password,
                    # Note: database and use_parallel_runtime would need to be passed
                    # to the driver after initialization if supported
                }

            case 'falkordb':
                if not HAS_FALKOR:
                    raise ValueError(
                        'FalkorDB driver not available in current graphiti-core version'
                    )

                # Use FalkorDB config if provided, otherwise use defaults
                if config.providers.falkordb:
                    falkor_config = config.providers.falkordb
                else:
                    # Create default FalkorDB configuration
                    from config.schema import FalkorDBProviderConfig

                    falkor_config = FalkorDBProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os
                from urllib.parse import urlparse

                uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
                password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)

                # Parse the URI to extract host and port
                parsed = urlparse(uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6379

                return {
                    'driver': 'falkordb',
                    'host': host,
                    'port': port,
                    'password': password,
                    'database': falkor_config.database,
                }

            case 'kuzu':
                if not HAS_KUZU:
                    raise ValueError(
                        'Kuzu driver not available in current graphiti-core version. '
                        'Install it with: pip install graphiti-core[kuzu] or pip install kuzu'
                    )

                if config.providers.kuzu:
                    kuzu_config = config.providers.kuzu
                else:
                    from config.schema import KuzuProviderConfig

                    kuzu_config = KuzuProviderConfig()

                import os
                from pathlib import Path

                db = os.environ.get('KUZU_DB', kuzu_config.db)
                db_path = Path(db)
                if not db_path.is_absolute():
                    db_path = Path.cwd() / db_path
                db_path.parent.mkdir(parents=True, exist_ok=True)

                return {
                    'driver': 'kuzu',
                    'db': str(db_path),
                    'max_concurrent_queries': kuzu_config.max_concurrent_queries,
                }

            case _:
                raise ValueError(f'Unsupported Database provider: {provider}')
