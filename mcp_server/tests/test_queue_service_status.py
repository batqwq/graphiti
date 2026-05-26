import asyncio

import pytest

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
