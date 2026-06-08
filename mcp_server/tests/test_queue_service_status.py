import asyncio
import json

import pytest

import services.queue_service as queue_module
from services.queue_service import QueueService


@pytest.mark.asyncio
async def test_queue_status_tracks_processing_completion_and_failures():
    queue_service = QueueService()
    group_id = 'status-test'
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_task():
        started.set()
        await release.wait()

    async def failing_task():
        raise RuntimeError('expected failure')

    await queue_service.add_episode_task(group_id, slow_task)
    await started.wait()

    status = queue_service.get_queue_status(group_id)
    assert status['status'] == 'processing'
    assert status['total_processing'] == 1
    assert status['total_retried'] == 0
    assert status['groups'][group_id]['processing'] == 1
    assert status['groups'][group_id]['idle'] is False

    release.set()
    await queue_service._episode_queues[group_id].join()

    status = queue_service.get_queue_status(group_id)
    assert status['status'] == 'idle'
    assert status['total_completed'] == 1
    assert status['groups'][group_id]['pending'] == 0
    assert status['groups'][group_id]['processing'] == 0

    await queue_service.add_episode_task(group_id, failing_task)
    await queue_service._episode_queues[group_id].join()

    status = queue_service.get_queue_status(group_id)
    assert status['total_failed'] == 1
    assert status['groups'][group_id]['last_error'] == 'expected failure'


@pytest.mark.asyncio
async def test_queue_starts_one_worker_per_group_for_rapid_submissions(monkeypatch):
    queue_service = QueueService()
    group_id = 'rapid-submit'
    worker_starts = 0
    release = asyncio.Event()

    async def fake_worker(group_id: str):
        nonlocal worker_starts
        worker_starts += 1
        await release.wait()

    monkeypatch.setattr(queue_service, '_process_episode_queue', fake_worker)

    async def task():
        pass

    await asyncio.gather(
        queue_service.add_episode_task(group_id, task),
        queue_service.add_episode_task(group_id, task),
        queue_service.add_episode_task(group_id, task),
    )
    await asyncio.sleep(0)

    release.set()
    assert worker_starts == 1


@pytest.mark.asyncio
async def test_episode_processing_retries_transient_json_failures(monkeypatch):
    queue_service = QueueService()
    group_id = 'retry-json'
    calls = 0

    class FakeGraphiti:
        async def add_episode(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise json.JSONDecodeError('invalid JSON', 'x', 0)

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(queue_module.asyncio, 'sleep', no_delay)
    await queue_service.initialize(FakeGraphiti())

    await queue_service.add_episode(
        group_id=group_id,
        name='Retry test',
        content='Test content',
        source_description='test',
        episode_type='text',
        entity_types={},
        uuid=None,
    )
    await queue_service._episode_queues[group_id].join()

    status = queue_service.get_queue_status(group_id)
    assert calls == 3
    assert status['total_completed'] == 1
    assert status['total_failed'] == 0
    assert status['total_retried'] == 2
    assert status['groups'][group_id]['retried'] == 2
