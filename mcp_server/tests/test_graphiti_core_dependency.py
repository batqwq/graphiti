from pathlib import Path

MCP_SERVER_DIR = Path(__file__).resolve().parents[1]


def test_graphiti_core_dependency_requires_kuzu_database_fix():
    pyproject = (MCP_SERVER_DIR / 'pyproject.toml').read_text(encoding='utf-8')
    lockfile = (MCP_SERVER_DIR / 'uv.lock').read_text(encoding='utf-8')

    assert 'graphiti-core[falkordb,kuzu,google-genai]>=0.29.1' in pyproject
    assert 'name = "graphiti-core"\nversion = "0.29.1"' in lockfile
    assert (
        '{ name = "graphiti-core", extras = ["falkordb", "kuzu", "google-genai"], specifier = ">=0.29.1" }'
        in lockfile
    )
