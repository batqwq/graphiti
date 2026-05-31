import sys
from pathlib import Path

import pytest

scripts_path = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_path))

from reindex_embeddings import (  # noqa: E402
    _build_write_query,
    _embedding_property,
    _normalize_suffix,
    _ordered_response_embeddings,
)


def test_ordered_response_embeddings_uses_api_indexes():
    result = _ordered_response_embeddings(
        {
            'data': [
                {'index': 1, 'embedding': [0.2]},
                {'index': 0, 'embedding': [0.1]},
            ]
        },
        expected_count=2,
    )

    assert result == [[0.1], [0.2]]


def test_embedding_property_normalizes_and_rejects_unsafe_suffixes():
    assert _embedding_property('name_embedding', _normalize_suffix('4096')) == (
        'name_embedding_4096'
    )

    with pytest.raises(ValueError, match='Unsafe Neo4j property name'):
        _embedding_property('name_embedding', '_bad suffix')


def test_write_query_preserves_original_before_runtime_update():
    query = _build_write_query(
        'MATCH (n:Entity)',
        'n',
        'name_embedding',
        'name_embedding_before_reindex',
    )

    assert 'n.name_embedding_before_reindex = coalesce(' in query
    assert 'n.name_embedding = item.emb' in query
    assert 'RETURN count(n) AS updated_count' in query
