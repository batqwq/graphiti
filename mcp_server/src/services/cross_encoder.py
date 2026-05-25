"""Cross-encoder helpers for MCP server startup."""

from graphiti_core.cross_encoder.client import CrossEncoderClient


class NoOpCrossEncoder(CrossEncoderClient):
    """Fallback reranker that preserves input order without external API calls."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0) for passage in passages]
