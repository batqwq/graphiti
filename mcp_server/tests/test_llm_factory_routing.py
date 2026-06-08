from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig
from services.factories import LLMClientFactory, _is_official_openai_api_url


def _config(api_url: str, response_format: str = 'auto') -> LLMConfig:
    return LLMConfig(
        provider='openai',
        model='test-model',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(
                api_key='test-key',
                api_url=api_url,
                response_format=response_format,
            )
        ),
    )


def test_official_openai_url_detection_uses_parsed_positive_match():
    assert _is_official_openai_api_url('https://api.openai.com/v1')
    assert _is_official_openai_api_url('https://api.openai.com/v1/')
    assert not _is_official_openai_api_url('http://api.openai.com/v1')
    assert not _is_official_openai_api_url('https://api.openai.com.evil.example/v1')
    assert not _is_official_openai_api_url('https://compatible.example/v1')


def test_factory_routes_official_and_compatible_endpoints_explicitly():
    assert isinstance(LLMClientFactory.create(_config('https://api.openai.com/v1')), OpenAIClient)

    generic = LLMClientFactory.create(
        _config('https://compatible.example/v1', response_format='json_object')
    )
    assert isinstance(generic, OpenAIGenericClient)
    assert generic._response_format_mode == 'json_object'
