import sys
from pathlib import Path

import pytest

scripts_path = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_path))

from reindex_embeddings import (  # noqa: E402
    _build_write_query,
    _embedding_property,
    _migrate_batch,
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


def test_ordered_response_embeddings_reports_invalid_provider_shape():
    with pytest.raises(RuntimeError, match='missing the required data array'):
        _ordered_response_embeddings({'error': 'unsupported'}, expected_count=1)

    with pytest.raises(RuntimeError, match='missing a valid embedding array'):
        _ordered_response_embeddings({'data': [{'index': 0}]}, expected_count=1)


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
    assert 'RETURN collect(elementId(n)) AS updated_ids' in query


@pytest.mark.asyncio
async def test_migrate_batch_reports_missing_element_ids():
    class FakeDriver:
        def __init__(self):
            self.calls = 0

        async def execute_query(self, query, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return (
                    [{'id': 'node-1', 'text': 'one'}, {'id': 'node-2', 'text': 'two'}],
                    None,
                    None,
                )
            return ([{'updated_ids': ['node-1']}], None, None)

    class FakeEmbedder:
        dim = 1

        async def embed_batch(self, texts):
            return [[0.1] for _ in texts]

    with pytest.raises(RuntimeError, match="missing elementIds=\\['node-2'\\]"):
        await _migrate_batch(
            FakeDriver(),
            FakeEmbedder(),
            'Entity',
            'READ',
            'WRITE',
            batch_size=2,
            offset=0,
        )
