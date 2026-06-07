#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from html import escape
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import parse_qs, urlparse

import graphiti_core.graphiti as _graphiti_core
import graphiti_core.search.search_utils as _search_utils
import graphiti_core.utils.maintenance.node_operations as _node_ops
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from config.schema import GraphitiConfig, ServerConfig
from models.response_types import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeSearchResponse,
    QueueStatusResponse,
    StatusResponse,
    SuccessResponse,
)
from services.cross_encoder import NoOpCrossEncoder
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from services.kuzu_compat import ensure_kuzu_database_attribute
from services.oauth_provider import PasswordOAuthProvider
from services.queue_service import QueueService
from utils.episodes import (
    get_episodes_by_created_at,
    normalize_episode_sort_order,
    resolve_episode_limit,
)
from utils.formatting import (
    DEFAULT_MAX_TEXT_CHARS,
    format_episode_result,
    format_fact_result,
    format_node_result,
)

# Load .env file from mcp_server directory
mcp_server_dir = Path(__file__).parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    # Try current working directory as fallback
    load_dotenv()


# Semaphore limit for concurrent Graphiti operations.
#
# This controls how many episodes can be processed simultaneously. Each episode
# processing involves multiple LLM calls (entity extraction, deduplication, etc.),
# so the actual number of concurrent LLM requests will be higher.
#
# TUNING GUIDELINES:
#
# LLM Provider Rate Limits (requests per minute):
# - OpenAI Tier 1 (free):     3 RPM   -> SEMAPHORE_LIMIT=1-2
# - OpenAI Tier 2:            60 RPM   -> SEMAPHORE_LIMIT=5-8
# - OpenAI Tier 3:           500 RPM   -> SEMAPHORE_LIMIT=10-15
# - OpenAI Tier 4:         5,000 RPM   -> SEMAPHORE_LIMIT=20-50
# - Anthropic (default):     50 RPM   -> SEMAPHORE_LIMIT=5-8
# - Anthropic (high tier): 1,000 RPM   -> SEMAPHORE_LIMIT=15-30
# - Azure OpenAI (varies):  Consult your quota -> adjust accordingly
#
# SYMPTOMS:
# - Too high: 429 rate limit errors, increased costs from parallel processing
# - Too low: Slow throughput, underutilized API quota
#
# MONITORING:
# - Watch logs for rate limit errors (429)
# - Monitor episode processing times
# - Check LLM provider dashboard for actual request rates
#
# DEFAULT: 10 (suitable for OpenAI Tier 3, mid-tier Anthropic)
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    return max(value, minimum)


MAX_MCP_RESULTS = _env_int('MCP_OUTPUT_MAX_RESULTS', 20)


def _bounded_result_limit(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return min(value, MAX_MCP_RESULTS)


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


# Configure structured logging with timestamps
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce access log noise
logging.getLogger('mcp.server.streamable_http_manager').setLevel(
    logging.WARNING
)  # Reduce MCP noise

# Patch graphiti-core constants to limit prompt context size.
# Without this, extraction prompts grow past model context windows (~200K).
# GRAPHITI_RELEVANT_SCHEMA_LIMIT env var (default: 1) controls previous episodes.
# GRAPHITI_NODE_DEDUP_LIMIT    env var (default: 3) controls dedup candidates.
_schema_limit = _env_int('GRAPHITI_RELEVANT_SCHEMA_LIMIT', 1)
_dedup_limit = _env_int('GRAPHITI_NODE_DEDUP_LIMIT', 3)
_search_utils.RELEVANT_SCHEMA_LIMIT = _schema_limit
_graphiti_core.RELEVANT_SCHEMA_LIMIT = _schema_limit
_node_ops.NODE_DEDUP_CANDIDATE_LIMIT = _dedup_limit

logger = logging.getLogger(__name__)
logger.info(
    'RELEVANT_SCHEMA_LIMIT=%d NODE_DEDUP_CANDIDATE_LIMIT=%d',
    _schema_limit,
    _dedup_limit,
)


# Patch uvicorn's logging config to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uvicorn_logger = logging.getLogger(logger_name)
        # Remove existing handlers and add our own with proper formatting
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


# Create global config instance - will be properly initialized later
config: GraphitiConfig

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a persistent knowledge-graph memory service. It stores source episodes, extracts entity nodes,
and derives relationship facts. Every item belongs to a group_id partition. Keep one stable group_id for one
knowledge domain and always pass it when known; unrelated groups must not be mixed.

Reading workflow:
1. Use search_memory_facts first for factual, relational, or temporal questions.
2. Use search_nodes to discover entities, inspect summaries, or obtain a node UUID for a centered fact search.
3. Use get_entity_edge to inspect one returned fact in detail.
4. Use get_episodes to inspect source episodes, provenance, or locate an episode UUID.
5. An empty search result is not proof that no memory exists. Try a more specific query, verify group_ids, and
   check get_memory_queue_status before concluding that information is absent.

Writing workflow:
1. Use add_memory for new source material. Write one coherent episode with explicit entity names, relationships,
   dates, and uncertainty. Avoid ambiguous pronouns and keyword-only fragments.
2. add_memory only queues work. Poll get_memory_queue_status until pending=0 and processing=0, then search to
   verify extraction. Report failed jobs instead of claiming completion.
3. Graphiti has no in-place update tool. To correct data, locate the exact episode or fact UUID, delete only the
   incorrect item when appropriate, add a corrected episode, wait for processing, and verify again.

Deletion rules:
- Never guess UUIDs. Obtain them from search_memory_facts, get_entity_edge, or get_episodes.
- delete_episode removes the source episode but may not remove every previously derived node, fact, or summary.
- delete_entity_edge removes one derived relationship fact but not its source episode.
- clear_graph is an irreversible group-level reset. Use it only after explicit user authorization.
"""


ADD_MEMORY_DESCRIPTION = """
Queue one source episode for asynchronous extraction into the knowledge graph.

Use this tool to add new evidence or a corrected source statement. Provide one coherent episode with explicit
entity names, relationships, dates, and uncertainty; do not submit keyword fragments. Reuse the same group_id
for the same knowledge domain. The response only confirms that the episode was queued, not that nodes and facts
were created. After calling this tool, poll get_memory_queue_status until the group has no pending or processing
jobs, then verify with search_memory_facts and search_nodes.
""".strip()

QUEUE_STATUS_DESCRIPTION = """
Inspect asynchronous memory-ingestion progress without modifying the graph.

Call this after add_memory and before validating newly written memories. A group is caught up only when pending
and processing are both zero. Any failed count or last_error must be reported and investigated; do not claim
that an import completed successfully while failures remain.
""".strip()

SEARCH_NODES_DESCRIPTION = """
Search entity nodes by natural-language meaning, name, aliases, summary, and optional entity type.

Use this for entity discovery, identity lookup, summary inspection, or obtaining a node UUID to pass as
center_node_uuid to search_memory_facts. For answering relationship or factual questions, prefer
search_memory_facts. Node summaries are generated context and may be incomplete or stale, so verify critical
claims against facts or source episodes. Empty results are not proof of absence.
""".strip()

SEARCH_FACTS_DESCRIPTION = """
Search derived relationship facts using a specific natural-language question.

This is the primary retrieval tool for factual, relational, and temporal questions. Include the relevant entity
names, relationship, and time constraint in the query, and pass group_ids whenever known. Use center_node_uuid
only after obtaining an exact node UUID from search_nodes. Results are relevance-ranked rather than guaranteed
complete; inspect an important fact with get_entity_edge and consult source episodes when provenance matters.
""".strip()

DELETE_EDGE_DESCRIPTION = """
Permanently delete exactly one derived relationship fact identified by its UUID.

Use only after locating the exact fact with search_memory_facts and, for important deletions, verifying it with
get_entity_edge. Never guess a UUID. This does not delete the source episode or connected entity nodes, and a
source episode may cause similar facts to exist or be derived again.
""".strip()

DELETE_EPISODE_DESCRIPTION = """
Permanently delete exactly one source episode identified by its UUID.

Use get_episodes to locate and verify the episode first; never guess a UUID. Deleting an episode removes that
source record and its direct links, but it does not guarantee removal of every node, derived fact, or generated
summary previously influenced by the episode. Delete incorrect derived facts separately when necessary.
""".strip()

GET_EDGE_DESCRIPTION = """
Retrieve the full stored representation of one relationship fact by UUID without modifying memory.

Use a UUID returned by search_memory_facts. This tool is appropriate for inspecting provenance, temporal fields,
attributes, and the exact fact before deletion or before relying on it in a high-confidence answer.
""".strip()

GET_EPISODES_DESCRIPTION = """
List raw source episodes from one or more groups in deterministic created_at order.

Use this for provenance review, recent-history inspection, pagination, or locating an episode UUID before
delete_episode. This is not semantic search; use search_memory_facts or search_nodes to find information by
meaning. group_ids is preferred; group_id, limit, and last_n exist only as compatibility aliases.
""".strip()

CLEAR_GRAPH_DESCRIPTION = """
Irreversibly delete all graph data in the specified group partitions.

This is a destructive reset, not a cleanup or correction tool. Use only when the user explicitly requests a
group reset and the intended group_ids have been verified. If group_ids is omitted, the server's default group
is cleared. This tool does not create a backup and cannot be undone.
""".strip()

GET_STATUS_DESCRIPTION = """
Check whether the MCP service is initialized and can query its graph database.

Use this for connectivity diagnostics only. A healthy result does not mean ingestion queues are empty, memories
exist, searches are accurate, or external LLM and embedding providers are functioning for new writes.
""".strip()

oauth_provider: PasswordOAuthProvider | None = None


def create_auth_components() -> tuple[AuthSettings | None, PasswordOAuthProvider | None]:
    if not _env_bool('MCP_AUTH_ENABLED'):
        return None, None

    public_url = os.getenv('MCP_PUBLIC_URL', '').rstrip('/')
    if not public_url:
        raise RuntimeError(
            'MCP_AUTH_ENABLED=true requires MCP_PUBLIC_URL, e.g. https://raz.942778.online'
        )

    approval_password = os.getenv('MCP_AUTH_APPROVAL_PASSWORD', '')
    if not approval_password:
        raise RuntimeError('MCP_AUTH_ENABLED=true requires MCP_AUTH_APPROVAL_PASSWORD')

    scopes = os.getenv('MCP_AUTH_SCOPES', 'graphiti:read graphiti:write').split()
    resource_path = os.getenv('MCP_AUTH_RESOURCE_PATH', '/mcp/').strip() or '/mcp/'
    if not resource_path.startswith('/'):
        resource_path = f'/{resource_path}'
    client_store_path = os.getenv('MCP_AUTH_CLIENT_STORE_PATH')
    if not client_store_path:
        client_store_path = str(mcp_server_dir / 'data' / 'oauth_clients.json')
    token_store_path = os.getenv('MCP_AUTH_TOKEN_STORE_PATH')
    if not token_store_path:
        token_store_path = str(mcp_server_dir / 'data' / 'oauth_tokens.json')

    auth_provider = PasswordOAuthProvider(
        public_url=public_url,
        approval_password=approval_password,
        scopes=scopes,
        token_ttl_seconds=_env_int('MCP_AUTH_TOKEN_TTL_SECONDS', 60 * 60 * 24 * 30),
        client_store_path=client_store_path,
        token_store_path=token_store_path,
    )
    auth_settings = AuthSettings(
        issuer_url=public_url,
        resource_server_url=f'{public_url}{resource_path}',
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=scopes,
            default_scopes=scopes,
        ),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=scopes,
    )
    return auth_settings, auth_provider


auth_settings, oauth_provider = create_auth_components()


def create_transport_security_settings() -> TransportSecuritySettings:
    allowed_hosts = ['127.0.0.1:*', 'localhost:*', '[::1]:*']
    allowed_origins = ['http://127.0.0.1:*', 'http://localhost:*', 'http://[::1]:*']

    public_url = os.getenv('MCP_PUBLIC_URL', '').rstrip('/')
    if public_url:
        parsed_url = urlparse(public_url)
        if parsed_url.netloc:
            allowed_hosts.extend([parsed_url.netloc, f'{parsed_url.netloc}:*'])
            allowed_origins.append(public_url)

    extra_hosts = os.getenv('MCP_ALLOWED_HOSTS', '')
    allowed_hosts.extend(host for host in extra_hosts.split() if host)

    extra_origins = os.getenv('MCP_ALLOWED_ORIGINS', '')
    allowed_origins.extend(origin for origin in extra_origins.split() if origin)

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
    auth=auth_settings,
    auth_server_provider=oauth_provider,
    transport_security=create_transport_security_settings(),
)

# Global services
graphiti_service: Optional['GraphitiService'] = None
queue_service: QueueService | None = None

# Global client for backward compatibility
graphiti_client: Graphiti | None = None
semaphore: asyncio.Semaphore


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        self.config = config
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.entity_types = None

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            # Create clients using factories
            llm_client = None
            embedder_client = None
            cross_encoder_client = NoOpCrossEncoder()

            # Create LLM client based on configured provider
            try:
                llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            # Create embedder client based on configured provider
            try:
                embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                gemini_reranker_model = os.getenv('GEMINI_RERANKER_MODEL')
                if google_api_key and gemini_reranker_model:
                    from graphiti_core.cross_encoder.gemini_reranker_client import (
                        GeminiRerankerClient,
                    )
                    from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                    cross_encoder_client = GeminiRerankerClient(
                        config=CoreLLMConfig(
                            api_key=google_api_key,
                            model=gemini_reranker_model,
                            temperature=0,
                            max_tokens=16,
                        )
                    )
            except Exception as e:
                logger.warning(f'Failed to create Gemini reranker, using no-op reranker: {e}')

            # Get database configuration
            db_config = DatabaseDriverFactory.create_config(self.config.database)

            # Build entity types from configuration
            custom_types = None
            if self.config.graphiti.entity_types:
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    # Create a dynamic Pydantic model for each entity type
                    # Note: Don't use 'name' as it's a protected Pydantic attribute
                    entity_model = type(
                        entity_type.name,
                        (BaseModel,),
                        {
                            '__doc__': entity_type.description,
                        },
                    )
                    custom_types[entity_type.name] = entity_model

            # Store entity types for later use
            self.entity_types = custom_types

            # Initialize Graphiti client with appropriate driver
            try:
                db_provider = self.config.database.provider.lower()
                if db_provider == 'falkordb':
                    # For FalkorDB, create a FalkorDriver instance directly
                    from graphiti_core.driver.falkordb_driver import FalkorDriver

                    falkor_driver = FalkorDriver(
                        host=db_config['host'],
                        port=db_config['port'],
                        password=db_config['password'],
                        database=db_config['database'],
                    )

                    self.client = Graphiti(
                        graph_driver=falkor_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        cross_encoder=cross_encoder_client,
                        max_coroutines=self.semaphore_limit,
                    )
                elif db_provider == 'kuzu':
                    from graphiti_core.driver.kuzu_driver import KuzuDriver

                    kuzu_driver = KuzuDriver(
                        db=db_config['db'],
                        max_concurrent_queries=db_config['max_concurrent_queries'],
                    )
                    ensure_kuzu_database_attribute(kuzu_driver, db_config['db'])

                    self.client = Graphiti(
                        graph_driver=kuzu_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        cross_encoder=cross_encoder_client,
                        max_coroutines=self.semaphore_limit,
                    )
                else:
                    # For Neo4j (default), use the original approach
                    self.client = Graphiti(
                        uri=db_config['uri'],
                        user=db_config['user'],
                        password=db_config['password'],
                        llm_client=llm_client,
                        embedder=embedder_client,
                        cross_encoder=cross_encoder_client,
                        max_coroutines=self.semaphore_limit,
                    )
            except Exception as db_error:
                # Check for connection errors
                error_msg = str(db_error).lower()
                if 'connection refused' in error_msg or 'could not connect' in error_msg:
                    db_provider = self.config.database.provider
                    if db_provider.lower() == 'falkordb':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: FalkorDB is not running\n'
                            f'{"=" * 70}\n\n'
                            f'FalkorDB at {db_config["host"]}:{db_config["port"]} is not accessible.\n\n'
                            f'To start FalkorDB:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose up\n'
                            f'  - Or run FalkorDB manually: docker run -p 6379:6379 falkordb/falkordb\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    elif db_provider.lower() == 'neo4j':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: Neo4j is not running\n'
                            f'{"=" * 70}\n\n'
                            f'Neo4j at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'To start Neo4j:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose -f docker/docker-compose-neo4j.yml up\n'
                            f'  - Or install Neo4j Desktop from: https://neo4j.com/download/\n'
                            f'  - Or run Neo4j manually: docker run -p 7474:7474 -p 7687:7687 neo4j:latest\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    else:
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: {db_provider} is not running\n'
                            f'{"=" * 70}\n\n'
                            f'{db_provider} at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'Please ensure {db_provider} is running and accessible.\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                # Re-raise other errors
                raise

            # Build indices
            await self.client.build_indices_and_constraints()

            logger.info('Successfully initialized Graphiti client')

            # Log configuration details
            if llm_client:
                logger.info(
                    f'Using LLM provider: {self.config.llm.provider} / {self.config.llm.model}'
                )
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            if embedder_client:
                logger.info(f'Using Embedder provider: {self.config.embedder.provider}')
            else:
                logger.info('No Embedder client configured - search will be limited')

            if self.entity_types:
                entity_type_names = list(self.entity_types.keys())
                logger.info(f'Using custom entity types: {", ".join(entity_type_names)}')
            else:
                logger.info('Using default entity types')

            logger.info(f'Using database: {self.config.database.provider}')
            logger.info(f'Using group_id: {self.config.graphiti.group_id}')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti client: {e}')
            raise

    async def get_client(self) -> Graphiti:
        """Get the Graphiti client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


@mcp.tool(
    title='Add memory episode',
    description=ADD_MEMORY_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def add_memory(
    name: Annotated[
        str,
        Field(
            description='Short, descriptive title for this source episode. Use a stable topic or event name.'
        ),
    ],
    episode_body: Annotated[
        str,
        Field(
            description=(
                'Complete source content to ingest. Use explicit names, relationships, dates, and uncertainty. '
                "When source='json', provide a valid JSON-encoded string rather than an object."
            )
        ),
    ],
    group_id: Annotated[
        str | None,
        Field(
            description=(
                'Exact knowledge-domain partition for this episode. Reuse one stable group_id for related '
                'memories. Omit only when the server default group is intended.'
            )
        ),
    ] = None,
    source: Annotated[
        str,
        Field(
            description=(
                "Episode source format: 'text' for prose, 'json' for a JSON-encoded string, or 'message' for "
                'conversation-style content.'
            )
        ),
    ] = 'text',
    source_description: Annotated[
        str,
        Field(
            description=(
                'Brief provenance description, such as the document type or conversation context. This helps '
                'future source review.'
            )
        ),
    ] = '',
    uuid: Annotated[
        str | None,
        Field(
            description=(
                'Optional caller-supplied episode UUID. Normally omit it. If supplied, it must uniquely and '
                'stably identify this source episode.'
            )
        ),
    ] = None,
) -> SuccessResponse | ErrorResponse:
    """Queue a source episode for asynchronous graph extraction."""
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        # Try to parse the source as an EpisodeType enum, with fallback to text
        episode_type = EpisodeType.text  # Default
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                # If the source doesn't match any enum value, use text as default
                logger.warning(f"Unknown source type '{source}', using 'text' as default")
                episode_type = EpisodeType.text

        # Submit to queue service for async processing
        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            entity_types=graphiti_service.entity_types,
            uuid=uuid or None,  # Ensure None is passed if uuid is None
        )

        return SuccessResponse(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode: {error_msg}')


@mcp.tool(
    title='Get memory queue status',
    description=QUEUE_STATUS_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_memory_queue_status(
    group_id: Annotated[
        str | None,
        Field(
            description=(
                'Exact group to inspect. Omit to return status for every queue group known to this server.'
            )
        ),
    ] = None,
) -> QueueStatusResponse | ErrorResponse:
    """Return background graph-building progress."""
    global queue_service

    if queue_service is None:
        return ErrorResponse(error='Queue service not initialized')

    return queue_service.get_queue_status(group_id)


@mcp.tool(
    title='Search entity nodes',
    description=SEARCH_NODES_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def search_nodes(
    query: Annotated[
        str,
        Field(
            description=(
                'Specific natural-language entity lookup. Include a canonical name, alias, role, or identifying '
                'context instead of isolated generic keywords.'
            )
        ),
    ],
    group_ids: Annotated[
        list[str] | None,
        Field(
            description=(
                'Exact knowledge-domain partitions to search. Omit only to search the server default group.'
            )
        ),
    ] = None,
    max_nodes: Annotated[
        int,
        Field(
            description=(
                'Maximum number of ranked nodes to return. Must be positive and is capped by the server.'
            )
        ),
    ] = 10,
    entity_types: Annotated[
        list[str] | None,
        Field(
            description=(
                'Optional exact configured entity labels to include. Omit when the labels are unknown.'
            )
        ),
    ] = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search entity nodes using hybrid retrieval."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        max_nodes = _bounded_result_limit(max_nodes, 'max_nodes')
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Create search filters
        search_filters = SearchFilters(
            node_labels=entity_types,
        )

        # Use the search_ method with node search config
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        results = await client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=search_filters,
        )

        # Extract nodes from results
        nodes = results.nodes[:max_nodes] if results.nodes else []

        if not nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        node_results = [format_node_result(node, minimal=True) for node in nodes]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=node_results)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool(
    title='Search memory facts',
    description=SEARCH_FACTS_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def search_memory_facts(
    query: Annotated[
        str,
        Field(
            description=(
                'Specific natural-language question describing the entities, relationship, and relevant time '
                'constraint. Prefer a complete question over a bag of keywords.'
            )
        ),
    ],
    group_ids: Annotated[
        list[str] | None,
        Field(
            description=(
                'Exact knowledge-domain partitions to search. Omit only to search the server default group.'
            )
        ),
    ] = None,
    max_facts: Annotated[
        int,
        Field(
            description=(
                'Maximum number of ranked facts to return. Must be positive and is capped by the server.'
            )
        ),
    ] = 10,
    center_node_uuid: Annotated[
        str | None,
        Field(
            description=(
                'Optional exact entity-node UUID from search_nodes used to constrain retrieval around that '
                'entity. Do not pass an entity name or guessed UUID.'
            )
        ),
    ] = None,
) -> FactSearchResponse | ErrorResponse:
    """Search derived relationship facts."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        max_facts = _bounded_result_limit(max_facts, 'max_facts')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge, minimal=True) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool(
    title='Delete relationship fact',
    description=DELETE_EDGE_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def delete_entity_edge(
    uuid: Annotated[
        str,
        Field(
            description=(
                'Exact relationship-fact UUID returned by search_memory_facts and preferably verified with '
                'get_entity_edge.'
            )
        ),
    ],
) -> SuccessResponse | ErrorResponse:
    """Delete one relationship fact by UUID."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool(
    title='Delete source episode',
    description=DELETE_EPISODE_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def delete_episode(
    uuid: Annotated[
        str,
        Field(description='Exact source-episode UUID returned and verified by get_episodes.'),
    ],
) -> SuccessResponse | ErrorResponse:
    """Delete one source episode by UUID."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool(
    title='Get relationship fact',
    description=GET_EDGE_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_entity_edge(
    uuid: Annotated[
        str,
        Field(description='Exact relationship-fact UUID returned by search_memory_facts.'),
    ],
) -> dict[str, Any] | ErrorResponse:
    """Retrieve one full relationship fact by UUID."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool(
    title='List source episodes',
    description=GET_EPISODES_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_episodes(
    group_ids: Annotated[
        list[str] | None,
        Field(
            description=(
                'Preferred group filter. Exact knowledge-domain partitions to list. Takes precedence over the '
                'legacy group_id argument.'
            )
        ),
    ] = None,
    max_episodes: Annotated[
        int | None,
        Field(
            description=(
                'Preferred page-size argument. Must be positive and is capped by the server. Do not combine '
                'with disagreeing limit or last_n values.'
            )
        ),
    ] = None,
    content_max_chars: Annotated[
        int,
        Field(
            description=(
                'Maximum preview characters per episode. Use zero to omit content previews; the server caps '
                'large values.'
            )
        ),
    ] = DEFAULT_MAX_TEXT_CHARS,
    sort_order: Annotated[
        str,
        Field(description="Created-at order: 'desc' for newest first or 'asc' for oldest first."),
    ] = 'desc',
    offset: Annotated[
        int,
        Field(description='Number of sorted episodes to skip for pagination. Must be zero or positive.'),
    ] = 0,
    group_id: Annotated[
        str | None,
        Field(
            description=(
                'Legacy single-group alias. Prefer group_ids. Used only when group_ids is omitted.'
            )
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Field(
            description=(
                'Legacy page-size alias. Prefer max_episodes. Must not disagree with max_episodes or last_n.'
            )
        ),
    ] = None,
    last_n: Annotated[
        int | None,
        Field(
            description=(
                'Legacy page-size alias. Prefer max_episodes. Must not disagree with max_episodes or limit.'
            )
        ),
    ] = None,
) -> EpisodeSearchResponse | ErrorResponse:
    """List source episodes with deterministic pagination."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        requested_max_episodes = resolve_episode_limit(
            max_episodes=max_episodes,
            limit=limit,
            last_n=last_n,
            default=10,
        )
        max_episodes = _bounded_result_limit(requested_max_episodes, 'max_episodes')
        normalized_sort_order = normalize_episode_sort_order(sort_order)
        if offset < 0:
            return ErrorResponse(error='offset must be zero or a positive integer')
        if content_max_chars < 0:
            return ErrorResponse(error='content_max_chars must be zero or a positive integer')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [group_id]
            if group_id
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        if effective_group_ids:
            episodes = await get_episodes_by_created_at(
                client.driver,
                effective_group_ids,
                limit=max_episodes,
                offset=offset,
                sort_order=normalized_sort_order,
            )
        else:
            # If no group IDs, we need to use a different approach
            # For now, return empty list when no group IDs specified
            episodes = []

        if not episodes:
            return EpisodeSearchResponse(message='No episodes found', episodes=[])

        episode_results = [
            format_episode_result(
                episode,
                content_max_chars=content_max_chars,
            )
            for episode in episodes
        ]

        return EpisodeSearchResponse(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool(
    title='Clear graph groups',
    description=CLEAR_GRAPH_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def clear_graph(
    group_ids: Annotated[
        list[str] | None,
        Field(
            description=(
                'Exact group partitions to erase permanently. Omit only when the explicitly authorized target '
                "is the server's default group."
            )
        ),
    ] = None,
) -> SuccessResponse | ErrorResponse:
    """Permanently clear graph data for selected groups."""
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids or [config.graphiti.group_id] if config.graphiti.group_id else []
        )

        if not effective_group_ids:
            return ErrorResponse(error='No group IDs specified for clearing')

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        return SuccessResponse(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool(
    title='Get server status',
    description=GET_STATUS_DESCRIPTION,
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_status() -> StatusResponse:
    """Check MCP service and graph database connectivity."""
    global graphiti_service

    if graphiti_service is None:
        return StatusResponse(status='error', message='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Test database connection with a simple query
        async with client.driver.session() as session:
            result = await session.run('MATCH (n) RETURN count(n) as count')
            # Consume the result to verify query execution
            if result:
                _ = [record async for record in result]

        # Use the provider from the service's config, not the global
        provider_name = graphiti_service.config.database.provider
        return StatusResponse(
            status='ok',
            message=f'Graphiti MCP server is running and connected to {provider_name} database',
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking database connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but database connection failed: {error_msg}',
        )


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


@mcp.custom_route('/oauth/confirm', methods=['GET', 'POST'])
async def oauth_confirm(request: Request) -> HTMLResponse | RedirectResponse:
    """Password-gated OAuth approval page for Claude remote MCP connectors."""
    if oauth_provider is None:
        return HTMLResponse('OAuth is not enabled', status_code=404)

    if request.method == 'GET':
        request_id = request.query_params.get('request_id', '')
        escaped_request_id = escape(request_id, quote=True)
        return HTMLResponse(
            f"""
<!doctype html>
<html>
  <head>
    <title>Authorize Graphiti MCP</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 520px; margin: 48px auto; padding: 0 20px; }}
      input, button {{ font: inherit; box-sizing: border-box; width: 100%; padding: 10px; margin-top: 10px; }}
      button {{ cursor: pointer; }}
      .note {{ color: #555; line-height: 1.4; }}
    </style>
  </head>
  <body>
    <h1>Authorize Graphiti MCP</h1>
    <p class="note">Enter your local MCP approval password to let this Claude connector access Graphiti memory.</p>
    <form method="post">
      <input type="hidden" name="request_id" value="{escaped_request_id}" />
      <label>
        Approval password
        <input name="approval_password" type="password" autocomplete="current-password" autofocus />
      </label>
      <button type="submit">Authorize</button>
    </form>
  </body>
</html>
""",
            status_code=200,
        )

    body = (await request.body()).decode('utf-8')
    form = parse_qs(body, keep_blank_values=True)
    request_id = form.get('request_id', [''])[0]
    approval_password = form.get('approval_password', [''])[0]
    redirect_url = oauth_provider.complete_authorization(request_id, approval_password)
    if redirect_url is None:
        return HTMLResponse('Authorization failed or expired', status_code=401)

    return RedirectResponse(redirect_url, status_code=302)


async def initialize_server() -> ServerConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, graphiti_service, queue_service, graphiti_client, semaphore

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with YAML configuration support'
    )

    # Configuration file argument
    # Default to config/config.yaml relative to the mcp_server directory
    default_config = Path(__file__).parent.parent / 'config' / 'config.yaml'
    parser.add_argument(
        '--config',
        type=Path,
        default=default_config,
        help='Path to YAML configuration file (default: config/config.yaml)',
    )

    # Transport arguments
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'http'],
        help='Transport to use: http (recommended, default), stdio (standard I/O), or sse (deprecated)',
    )
    parser.add_argument(
        '--host',
        help='Host to bind the MCP server to',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind the MCP server to',
    )

    # Provider selection arguments
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'azure_openai', 'anthropic', 'gemini', 'groq', 'fireworks'],
        help='LLM provider to use',
    )
    parser.add_argument(
        '--embedder-provider',
        choices=['openai', 'openrouter', 'azure_openai', 'gemini', 'voyage'],
        help='Embedder provider to use',
    )
    parser.add_argument(
        '--database-provider',
        choices=['neo4j', 'falkordb', 'kuzu'],
        help='Database provider to use',
    )

    # LLM configuration arguments
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--small-model', help='Small model name to use with the LLM client')
    parser.add_argument(
        '--temperature', type=float, help='Temperature setting for the LLM (0.0-2.0)'
    )

    # Embedder configuration arguments
    parser.add_argument('--embedder-model', help='Model name to use with the embedder')

    # Graphiti-specific arguments
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, uses config file or generates random UUID.',
    )
    parser.add_argument(
        '--user-id',
        help='User ID for tracking operations',
    )
    parser.add_argument(
        '--destroy-graph',
        action='store_true',
        help='Destroy all Graphiti graphs on startup',
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    # Also apply legacy CLI args for backward compatibility
    if hasattr(args, 'destroy_graph'):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info('Using configuration:')
    logger.info(f'  - LLM: {config.llm.provider} / {config.llm.model}')
    logger.info(f'  - Embedder: {config.embedder.provider} / {config.embedder.model}')
    logger.info(f'  - Database: {config.database.provider}')
    logger.info(f'  - Group ID: {config.graphiti.group_id}')
    logger.info(f'  - Transport: {config.server.transport}')

    # Log graphiti-core version
    try:
        import graphiti_core

        graphiti_version = getattr(graphiti_core, '__version__', 'unknown')
        logger.info(f'  - Graphiti Core: {graphiti_version}')
    except Exception:
        # Check for Docker-stored version file
        version_file = Path('/app/.graphiti-core-version')
        if version_file.exists():
            graphiti_version = version_file.read_text().strip()
            logger.info(f'  - Graphiti Core: {graphiti_version}')
        else:
            logger.info('  - Graphiti Core: version unavailable')

    # Handle graph destruction if requested
    if hasattr(config, 'destroy_graph') and config.destroy_graph:
        logger.warning('Destroying all Graphiti graphs as requested...')
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info('All graphs destroyed')

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT)
    queue_service = QueueService()
    await graphiti_service.initialize()

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with the client
    await queue_service.initialize(graphiti_client)

    # Set MCP server settings
    if config.server.host:
        mcp.settings.host = config.server.host
    if config.server.port:
        mcp.settings.port = config.server.port

    # Return MCP configuration for transport
    return config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info(f'Access the server at: http://{mcp.settings.host}:{mcp.settings.port}/sse')
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        # Use localhost for display if binding to 0.0.0.0
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.info(
            f'Running MCP server with streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  MCP Endpoint: http://{display_host}:{mcp.settings.port}/mcp/')
        logger.info('  Transport: HTTP (streamable)')

        # Show FalkorDB Browser UI access if enabled and relevant
        if config.database.provider.lower() == 'falkordb' and os.environ.get('BROWSER', '1') == '1':
            logger.info(f'  FalkorDB Browser UI: http://{display_host}:3000/')

        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /mcp/ endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f'Unsupported transport: {mcp_config.transport}. Use "sse", "stdio", or "http"'
        )


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
