import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull

from services.oauth_provider import PasswordOAuthProvider


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


@pytest.mark.asyncio
async def test_password_oauth_provider_issues_and_verifies_tokens():
    provider = PasswordOAuthProvider(
        public_url='https://raz.942778.online',
        approval_password='secret',
        scopes=['graphiti:read', 'graphiti:write'],
    )
    client = OAuthClientInformationFull(
        client_id='client-1',
        redirect_uris=['https://claude.ai/callback'],
        token_endpoint_auth_method='none',
        scope='graphiti:read graphiti:write',
    )
    await provider.register_client(client)

    authorization_url = await provider.authorize(
        client,
        AuthorizationParams(
            state='state-1',
            scopes=['graphiti:read'],
            code_challenge=pkce_challenge('verifier-1'),
            redirect_uri='https://claude.ai/callback',
            redirect_uri_provided_explicitly=True,
            resource='https://raz.942778.online/mcp/',
        ),
    )
    request_id = parse_qs(urlparse(authorization_url).query)['request_id'][0]

    assert provider.complete_authorization(request_id, 'wrong') is None

    redirect_url = provider.complete_authorization(request_id, 'secret')

    assert redirect_url is not None
    redirect_params = parse_qs(urlparse(redirect_url).query)
    code = redirect_params['code'][0]
    assert redirect_params['state'] == ['state-1']

    authorization_code = await provider.load_authorization_code(client, code)
    assert authorization_code is not None

    token_response = await provider.exchange_authorization_code(client, authorization_code)
    access_token = await provider.load_access_token(token_response.access_token)

    assert access_token is not None
    assert access_token.client_id == 'client-1'
    assert access_token.scopes == ['graphiti:read']


@pytest.mark.asyncio
async def test_password_oauth_provider_rejects_expired_pending_authorization():
    provider = PasswordOAuthProvider(
        public_url='https://raz.942778.online',
        approval_password='secret',
        scopes=['graphiti:read'],
        auth_code_ttl_seconds=-1,
    )
    client = OAuthClientInformationFull(
        client_id='client-1',
        redirect_uris=['https://claude.ai/callback'],
        token_endpoint_auth_method='none',
        scope='graphiti:read',
    )
    await provider.register_client(client)

    authorization_url = await provider.authorize(
        client,
        AuthorizationParams(
            state=None,
            scopes=['graphiti:read'],
            code_challenge=pkce_challenge('verifier-1'),
            redirect_uri='https://claude.ai/callback',
            redirect_uri_provided_explicitly=True,
            resource=None,
        ),
    )
    request_id = parse_qs(urlparse(authorization_url).query)['request_id'][0]

    assert provider.complete_authorization(request_id, 'secret') is None
