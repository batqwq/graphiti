from types import SimpleNamespace

from utils.formatting import (
    DEFAULT_MAX_TEXT_CHARS,
    compact_value,
    format_episode_result,
    format_fact_result,
    format_node_result,
    truncate_text,
)


class FakeEdge:
    def model_dump(self, mode: str, exclude: set[str]):
        assert mode == 'json'
        assert 'fact_embedding' in exclude
        return {
            'uuid': 'edge-1',
            'fact': 'x' * 2000,
            'fact_embedding': [0.1, 0.2],
            'attributes': {
                'notes': 'y' * 2000,
                'fact_embedding': [0.3, 0.4],
            },
        }


def test_truncate_text_marks_omitted_content():
    result = truncate_text('abcdef' * 20, 40)

    assert result is not None
    assert len(result) <= 40
    assert result.endswith(']')


def test_truncate_text_respects_tiny_limits():
    result = truncate_text('abcdef', 4)

    assert result is not None
    assert len(result) <= 4


def test_compact_value_removes_embeddings_and_limits_collections():
    result = compact_value(
        {
            'name': 'node',
            'name_embedding': [1, 2, 3],
            'items': [1, 2, 3, 4],
        },
        max_text_chars=20,
        max_collection_items=2,
    )

    assert result['name'] == 'node'
    assert 'name_embedding' not in result
    assert result['items'] == [1, 2, {'_truncated_items': 2}]


def test_format_fact_result_drops_embeddings_and_truncates_text():
    result = format_fact_result(FakeEdge())

    assert 'fact_embedding' not in result
    assert 'fact_embedding' not in result['attributes']
    assert result['fact'].endswith(']')
    assert result['attributes']['notes'].endswith(']')


def test_format_fact_result_can_return_minimal_search_shape():
    edge = SimpleNamespace(uuid='edge-1', name='RELATES_TO', fact='Alice knows Bob')

    result = format_fact_result(edge, minimal=True)

    assert result == {
        'uuid': 'edge-1',
        'name': 'RELATES_TO',
        'fact': 'Alice knows Bob',
    }


def test_format_node_result_can_return_minimal_search_shape():
    node = SimpleNamespace(uuid='node-1', name='Alice', summary='Engineer')

    result = format_node_result(node, minimal=True)

    assert result == {
        'uuid': 'node-1',
        'name': 'Alice',
        'summary': 'Engineer',
    }


def test_format_episode_result_truncates_content_by_default():
    episode = SimpleNamespace(
        uuid='episode-1',
        name='Long Episode',
        content='z' * 80,
        created_at=None,
        source='text',
        source_description='source',
        group_id='main',
    )

    result = format_episode_result(episode, content_max_chars=40)

    assert result['content_length'] == 80
    assert result['content_truncated'] is True
    assert len(result['content']) <= 40
    assert result['content'].endswith(']')


def test_format_episode_result_keeps_short_content():
    episode = SimpleNamespace(
        uuid='episode-1',
        name='Long Episode',
        content='z' * 80,
        created_at=None,
        source='text',
        source_description='source',
        group_id='main',
    )

    result = format_episode_result(episode, content_max_chars=200)

    assert result['content_length'] == 80
    assert result['content_truncated'] is False
    assert result['content'] == 'z' * 80


def test_format_episode_result_caps_requested_preview_size():
    episode = SimpleNamespace(
        uuid='episode-1',
        name='Long Episode',
        content='z' * 2000,
        created_at=None,
        source='text',
        source_description='source',
        group_id='main',
    )

    result = format_episode_result(episode, content_max_chars=5000)

    assert result['content_length'] == 2000
    assert result['content_truncated'] is True
    assert len(result['content']) <= DEFAULT_MAX_TEXT_CHARS
