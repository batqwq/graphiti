"""Compatibility helpers for Graphiti Kuzu driver versions."""

from typing import Any


def ensure_kuzu_database_attribute(driver: Any, database: str) -> None:
    """Ensure KuzuDriver exposes the private database name expected by Graphiti.

    graphiti-core's Graphiti.add_episode code reads driver._database, while some
    released KuzuDriver wheels do not initialize it. Setting it here keeps the MCP
    server compatible without patching installed site-packages.
    """
    if not hasattr(driver, '_database'):
        driver._database = database
