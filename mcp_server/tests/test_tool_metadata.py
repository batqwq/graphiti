"""Regression tests for MCP tool discovery metadata."""

import re

import pytest

import graphiti_mcp_server as server

EXPECTED_TOOLS = {
    'add_memory',
    'get_memory_queue_status',
    'search_nodes',
    'search_memory_facts',
    'delete_entity_edge',
    'delete_episode',
    'get_entity_edge',
    'get_episodes',
    'clear_graph',
    'get_status',
}

READ_ONLY_TOOLS = {
    'get_memory_queue_status',
    'search_nodes',
    'search_memory_facts',
    'get_entity_edge',
    'get_episodes',
    'get_status',
}

DESTRUCTIVE_TOOLS = {
    'delete_entity_edge',
    'delete_episode',
    'clear_graph',
}


@pytest.fixture
async def tools_by_name():
    tools = await server.mcp.list_tools()
    return {tool.name: tool for tool in tools}


@pytest.mark.asyncio
async def test_all_tools_have_complete_discovery_metadata(tools_by_name):
    assert set(tools_by_name) == EXPECTED_TOOLS

    for tool in tools_by_name.values():
        assert tool.title
        assert tool.description
        assert len(tool.description) >= 100
        assert tool.annotations is not None
        assert tool.annotations.openWorldHint is False

        for parameter_name, parameter_schema in tool.inputSchema.get('properties', {}).items():
            assert parameter_schema.get('description'), (
                f'{tool.name}.{parameter_name} is missing a parameter description'
            )


@pytest.mark.asyncio
async def test_read_only_and_mutating_annotations_are_accurate(tools_by_name):
    for name in READ_ONLY_TOOLS:
        annotations = tools_by_name[name].annotations
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is True

    add_annotations = tools_by_name['add_memory'].annotations
    assert add_annotations.readOnlyHint is False
    assert add_annotations.destructiveHint is False
    assert add_annotations.idempotentHint is False

    for name in DESTRUCTIVE_TOOLS:
        annotations = tools_by_name[name].annotations
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is True
        assert annotations.idempotentHint is True


@pytest.mark.asyncio
async def test_descriptions_explain_critical_workflows_and_risks(tools_by_name):
    assert 'get_memory_queue_status' in tools_by_name['add_memory'].description
    assert 'primary retrieval tool' in tools_by_name['search_memory_facts'].description
    assert 'prefer' in tools_by_name['search_nodes'].description
    assert 'search_memory_facts' in tools_by_name['search_nodes'].description
    assert 'does not guarantee removal' in tools_by_name['delete_episode'].description
    assert 'Never guess a UUID' in tools_by_name['delete_entity_edge'].description
    assert 'explicitly requests' in tools_by_name['clear_graph'].description
    assert 'cannot be undone' in tools_by_name['clear_graph'].description


def test_server_instructions_reference_real_tools_and_safe_workflows():
    instructions = server.GRAPHITI_MCP_INSTRUCTIONS

    assert re.search(r'\bsearch_facts\b', instructions) is None
    assert 'search_memory_facts' in instructions
    assert 'pending=0 and processing=0' in instructions
    assert 'Never guess UUIDs' in instructions
    assert 'explicit user authorization' in instructions
