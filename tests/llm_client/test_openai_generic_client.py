from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    value: str


class DummyCompletions:
    def __init__(self, *, reject_json_schema: bool = False):
        self.reject_json_schema = reject_json_schema
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_json_schema and kwargs['response_format']['type'] == 'json_schema':
            request = httpx.Request('POST', 'https://compatible.example/v1/chat/completions')
            response = httpx.Response(400, request=request)
            raise openai.BadRequestError(
                'response_format json_schema is unsupported',
                response=response,
                body=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'))]
        )


class DummyClient:
    def __init__(self, *, reject_json_schema: bool = False):
        self.chat = SimpleNamespace(
            completions=DummyCompletions(reject_json_schema=reject_json_schema)
        )


@pytest.mark.asyncio
async def test_auto_response_format_falls_back_and_remembers_json_object():
    dummy = DummyClient(reject_json_schema=True)
    client = OpenAIGenericClient(
        config=LLMConfig(response_format='auto'),
        client=dummy,
    )
    messages = [
        Message(role='system', content='Original system instruction'),
        Message(role='user', content='Return a value'),
    ]

    result = await client._generate_response(messages, ResponseModel)

    assert result == {'value': 'ok'}
    assert [call['response_format']['type'] for call in dummy.chat.completions.calls] == [
        'json_schema',
        'json_object',
    ]
    fallback_messages = dummy.chat.completions.calls[1]['messages']
    assert fallback_messages[0]['content'].startswith('Original system instruction')
    assert 'Return a valid JSON object only' in fallback_messages[0]['content']
    assert messages[0].content == 'Original system instruction'
    assert client._response_format_mode == 'json_object'

    await client._generate_response(messages, ResponseModel)
    assert dummy.chat.completions.calls[-1]['response_format']['type'] == 'json_object'


@pytest.mark.asyncio
async def test_explicit_json_object_preserves_existing_system_message_position():
    dummy = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(response_format='json_object'),
        client=dummy,
    )
    messages = [
        Message(role='system', content='Original system instruction'),
        Message(role='user', content='Return a value'),
    ]

    await client._generate_response(messages, ResponseModel)

    sent_messages = dummy.chat.completions.calls[0]['messages']
    assert len(sent_messages) == 2
    assert sent_messages[0]['role'] == 'system'
    assert sent_messages[0]['content'].startswith('Original system instruction')
    assert 'Return a valid JSON object only' in sent_messages[0]['content']


def test_invalid_response_format_is_rejected():
    with pytest.raises(ValueError, match='response_format must be'):
        OpenAIGenericClient(
            config=LLMConfig(response_format='invalid'),
            client=DummyClient(),
        )
