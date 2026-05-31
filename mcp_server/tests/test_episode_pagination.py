from datetime import datetime, timezone

import pytest
from graphiti_core.driver.driver import GraphProvider

from utils.episodes import (
    get_episodes_by_created_at,
    normalize_episode_sort_order,
    resolve_episode_limit,
)


class FakeDriver:
    provider = GraphProvider.NEO4J

    def __init__(self):
        self.query = ''
        self.params = {}

    async def execute_query(self, query: str, **params):
        self.query = query
        self.params = params
        return (
            [
                {
                    'uuid': 'episode-new',
                    'name': 'Latest',
                    'group_id': 'raz',
                    'created_at': datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
                    'source': 'text',
                    'source_description': 'test',
                    'content': 'new content',
                    'valid_at': datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
                    'entity_edges': [],
                }
            ],
            None,
            None,
        )


def test_normalize_episode_sort_order_accepts_expected_values():
    assert normalize_episode_sort_order(' DESC ') == 'desc'
    assert normalize_episode_sort_order('asc') == 'asc'


def test_normalize_episode_sort_order_rejects_unknown_values():
    with pytest.raises(ValueError, match='sort_order'):
        normalize_episode_sort_order('latest')


def test_resolve_episode_limit_accepts_current_and_legacy_aliases():
    assert resolve_episode_limit(max_episodes=7, limit=None, last_n=None, default=10) == 7
    assert resolve_episode_limit(max_episodes=None, limit=8, last_n=None, default=10) == 8
    assert resolve_episode_limit(max_episodes=None, limit=None, last_n=9, default=10) == 9
    assert resolve_episode_limit(max_episodes=None, limit=None, last_n=None, default=10) == 10
    assert resolve_episode_limit(max_episodes=5, limit=5, last_n=5, default=10) == 5


def test_resolve_episode_limit_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match='cannot disagree'):
        resolve_episode_limit(max_episodes=5, limit=6, last_n=None, default=10)


@pytest.mark.asyncio
async def test_get_episodes_by_created_at_uses_sort_order_and_offset():
    driver = FakeDriver()

    episodes = await get_episodes_by_created_at(
        driver,
        ['raz'],
        limit=5,
        offset=20,
        sort_order='desc',
    )

    assert episodes[0].uuid == 'episode-new'
    assert 'ORDER BY created_at DESC, uuid DESC' in driver.query
    assert 'SKIP $offset' in driver.query
    assert 'LIMIT $limit' in driver.query
    assert driver.params['group_ids'] == ['raz']
    assert driver.params['offset'] == 20
    assert driver.params['limit'] == 5


@pytest.mark.asyncio
async def test_get_episodes_by_created_at_rejects_negative_offset():
    with pytest.raises(ValueError, match='offset'):
        await get_episodes_by_created_at(
            FakeDriver(),
            ['raz'],
            limit=5,
            offset=-1,
            sort_order='desc',
        )
