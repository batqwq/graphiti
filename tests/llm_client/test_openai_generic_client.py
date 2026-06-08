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


class RecursiveResponseModel(BaseModel):
    value: str
    child: 'RecursiveResponseModel | None' = None


class DummyCompletions:
    def __init__(
        self,
        *,
        reject_json_schema: bool = False,
        responses: list[tuple[str | None, str]] | None = None,
    ):
        self.reject_json_schema = reject_json_schema
        self.responses = responses or [('{"value":"ok"}', 'stop')]
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
        content, finish_reason = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


class DummyClient:
    def __init__(
        self,
        *,
        reject_json_schema: bool = False,
        responses: list[tuple[str | None, str]] | None = None,
    ):
        self.chat = SimpleNamespace(
            completions=DummyCompletions(
                reject_json_schema=reject_json_schema,
                responses=responses,
            )
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
    assert 'EXAMPLE JSON OUTPUT SHAPE' in sent_messages[0]['content']
    assert '{"value": "string"}' in sent_messages[0]['content']


@pytest.mark.asyncio
async def test_generate_response_uses_per_call_max_tokens():
    dummy = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(response_format='json_object'),
        client=dummy,
        max_tokens=4096,
    )

    await client._generate_response(
        [Message(role='user', content='Return a value')],
        ResponseModel,
        max_tokens=16384,
    )

    assert dummy.chat.completions.calls[0]['max_tokens'] == 16384


@pytest.mark.asyncio
async def test_generate_response_retries_empty_json_content():
    dummy = DummyClient(
        responses=[
            (None, 'stop'),
            ('{"value":"ok"}', 'stop'),
        ]
    )
    client = OpenAIGenericClient(
        config=LLMConfig(response_format='json_object'),
        client=dummy,
    )

    result = await client.generate_response(
        [
            Message(role='system', content='Original system instruction'),
            Message(role='user', content='Return a value'),
        ],
        ResponseModel,
    )

    assert result == {'value': 'ok'}
    assert len(dummy.chat.completions.calls) == 2
    retry_messages = dummy.chat.completions.calls[1]['messages']
    assert 'empty content' in retry_messages[-1]['content']


@pytest.mark.asyncio
async def test_generate_response_retries_truncated_json_content():
    dummy = DummyClient(
        responses=[
            ('{"value":"unfinished', 'length'),
            ('{"value":"ok"}', 'stop'),
        ]
    )
    client = OpenAIGenericClient(
        config=LLMConfig(response_format='json_object'),
        client=dummy,
    )

    result = await client.generate_response(
        [
            Message(role='system', content='Original system instruction'),
            Message(role='user', content='Return a value'),
        ],
        ResponseModel,
        max_tokens=8192,
    )

    assert result == {'value': 'ok'}
    assert len(dummy.chat.completions.calls) == 2
    assert all(call['max_tokens'] == 8192 for call in dummy.chat.completions.calls)
    retry_messages = dummy.chat.completions.calls[1]['messages']
    assert 'truncated at max_tokens=8192' in retry_messages[-1]['content']


def test_json_output_example_handles_recursive_schemas():
    example = OpenAIGenericClient._json_example_from_schema(
        RecursiveResponseModel.model_json_schema()
    )

    assert example == {'value': 'string', 'child': None}


def test_invalid_response_format_is_rejected():
    with pytest.raises(ValueError, match='response_format must be'):
        OpenAIGenericClient(
            config=LLMConfig(response_format='invalid'),
            client=DummyClient(),
        )
