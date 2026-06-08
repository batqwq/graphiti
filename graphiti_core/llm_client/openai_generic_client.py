"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens
        response_format = getattr(config, 'response_format', 'auto').strip().lower()
        if response_format not in {'auto', 'json_schema', 'json_object'}:
            raise ValueError(
                'OpenAI-compatible response_format must be auto, json_schema, or json_object'
            )
        self._response_format_auto = response_format == 'auto'
        self._response_format_mode = (
            'json_schema' if self._response_format_auto else response_format
        )

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    @staticmethod
    def _loads_json_result(result: str) -> dict[str, Any]:
        stripped = result.strip()
        if stripped.startswith('```'):
            lines = stripped.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            stripped = '\n'.join(lines).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            starts = [idx for idx in (stripped.find('{'), stripped.find('[')) if idx >= 0]
            start = min(starts) if starts else -1
            end = max(stripped.rfind('}'), stripped.rfind(']'))
            if start >= 0 and end > start:
                return json.loads(stripped[start : end + 1])
            raise

    @staticmethod
    def _json_object_messages(
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModel] | None,
    ) -> list[ChatCompletionMessageParam]:
        json_instruction = 'Return a valid JSON object only. Do not return markdown, prose, or code fences.'
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            json_instruction += f' The JSON object must conform to this schema: {schema}'

        result = [dict(message) for message in messages]
        for message in result:
            if message.get('role') == 'system':
                existing_content = message.get('content') or ''
                message['content'] = f'{existing_content}\n\n{json_instruction}'.strip()
                break
        else:
            result.insert(0, {'role': 'system', 'content': json_instruction})
        return typing.cast(list[ChatCompletionMessageParam], result)

    @staticmethod
    def _supports_json_object_fallback(error: openai.BadRequestError) -> bool:
        error_text = str(error).lower()
        return 'json_schema' in error_text or 'response_format' in error_text

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            cleaned_content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': cleaned_content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': cleaned_content})
        try:
            async def create_completion(response_format_mode: str):
                request_messages = openai_messages
                response_format: dict[str, Any] = {'type': 'json_object'}
                if response_model is not None and response_format_mode == 'json_schema':
                    schema_name = getattr(response_model, '__name__', 'structured_response')
                    response_format = {
                        'type': 'json_schema',
                        'json_schema': {
                            'name': schema_name,
                            'schema': response_model.model_json_schema(),
                        },
                    }
                elif response_format_mode == 'json_object':
                    request_messages = self._json_object_messages(openai_messages, response_model)

                completion_kwargs: dict[str, Any] = {
                    'model': self.model or DEFAULT_MODEL,
                    'messages': request_messages,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'response_format': response_format,
                }
                return await self.client.chat.completions.create(**completion_kwargs)

            try:
                response = await create_completion(self._response_format_mode)
            except openai.BadRequestError as error:
                if (
                    not self._response_format_auto
                    or self._response_format_mode != 'json_schema'
                    or not self._supports_json_object_fallback(error)
                ):
                    raise
                logger.warning(
                    'OpenAI-compatible endpoint rejected json_schema; falling back to json_object'
                )
                self._response_format_mode = 'json_object'
                response = await create_completion('json_object')

            result = response.choices[0].message.content or ''
            return self._loads_json_result(result)
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None

            while retry_count <= self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                    return response
                except (RateLimitError, RefusalError):
                    # These errors should not trigger retries
                    span.set_status('error', str(last_error))
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ):
                    # Let OpenAI's client handle these retries
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e

                    # Don't retry if we've hit the max retries
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1

                    # Construct a detailed error message for the LLM
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response, ensuring the output matches '
                        f'the expected format and constraints.'
                    )

                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

            # If we somehow get here, raise the last error
            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
