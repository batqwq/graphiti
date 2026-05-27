"""Small OAuth provider for protecting a self-hosted Graphiti MCP server."""

import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)


@dataclass
class PendingAuthorization:
    client_id: str
    params: AuthorizationParams
    expires_at: float


class PasswordOAuthProvider:
    """Small OAuth provider with a password-gated authorization page.

    Registered clients and issued tokens can be persisted locally so trusted MCP clients do not
    need to re-authorize after a server restart.
    """

    def __init__(
        self,
        *,
        public_url: str,
        approval_password: str,
        scopes: list[str],
        token_ttl_seconds: int = 60 * 60 * 24 * 30,
        auth_code_ttl_seconds: int = 300,
        client_store_path: str | Path | None = None,
        token_store_path: str | Path | None = None,
    ):
        self.public_url = public_url.rstrip('/')
        self.approval_password = approval_password
        self.scopes = scopes
        self.token_ttl_seconds = token_ttl_seconds
        self.auth_code_ttl_seconds = auth_code_ttl_seconds
        self.client_store_path = Path(client_store_path) if client_store_path else None
        self.token_store_path = Path(token_store_path) if token_store_path else None

        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.pending_authorizations: dict[str, PendingAuthorization] = {}
        self.authorization_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}
        self._load_clients()
        self._load_tokens()

    def _load_clients(self) -> None:
        if self.client_store_path is None or not self.client_store_path.exists():
            return

        try:
            raw_clients = json.loads(self.client_store_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning('Failed to load OAuth client store %s: %s', self.client_store_path, error)
            return

        if not isinstance(raw_clients, dict):
            logger.warning('Ignoring OAuth client store %s because it is not an object', self.client_store_path)
            return

        for client_id, client_data in raw_clients.items():
            try:
                client = OAuthClientInformationFull.model_validate(client_data)
            except Exception as error:
                logger.warning('Ignoring invalid OAuth client %s: %s', client_id, error)
                continue
            if client.client_id is not None:
                self.clients[client.client_id] = client

    def _save_clients(self) -> None:
        if self.client_store_path is None:
            return

        self.client_store_path.parent.mkdir(parents=True, exist_ok=True)
        clients = {client_id: client.model_dump(mode='json') for client_id, client in self.clients.items()}
        self.client_store_path.write_text(json.dumps(clients, indent=2, sort_keys=True), encoding='utf-8')

    def _load_tokens(self) -> None:
        if self.token_store_path is None or not self.token_store_path.exists():
            return

        try:
            raw_tokens = json.loads(self.token_store_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning('Failed to load OAuth token store %s: %s', self.token_store_path, error)
            return

        if not isinstance(raw_tokens, dict):
            logger.warning('Ignoring OAuth token store %s because it is not an object', self.token_store_path)
            return

        now = int(time.time())
        for token_value, token_data in raw_tokens.get('access_tokens', {}).items():
            try:
                token = AccessToken.model_validate(token_data)
            except Exception as error:
                logger.warning('Ignoring invalid OAuth access token %s: %s', token_value, error)
                continue
            if token.expires_at is None or token.expires_at >= now:
                self.access_tokens[token.token] = token

        for token_value, token_data in raw_tokens.get('refresh_tokens', {}).items():
            try:
                token = RefreshToken.model_validate(token_data)
            except Exception as error:
                logger.warning('Ignoring invalid OAuth refresh token %s: %s', token_value, error)
                continue
            if token.expires_at is None or token.expires_at >= now:
                self.refresh_tokens[token.token] = token

    def _save_tokens(self) -> None:
        if self.token_store_path is None:
            return

        self.token_store_path.parent.mkdir(parents=True, exist_ok=True)
        tokens = {
            'access_tokens': {
                token: access_token.model_dump(mode='json')
                for token, access_token in self.access_tokens.items()
            },
            'refresh_tokens': {
                token: refresh_token.model_dump(mode='json')
                for token, refresh_token in self.refresh_tokens.items()
            },
        }
        temp_path = self.token_store_path.with_suffix(f'{self.token_store_path.suffix}.tmp')
        temp_path.write_text(json.dumps(tokens, indent=2, sort_keys=True), encoding='utf-8')
        temp_path.replace(self.token_store_path)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id is None:
            raise ValueError('client_id is required')
        self.clients[client_info.client_id] = client_info
        self._save_clients()

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if client.client_id is None:
            raise AuthorizeError('invalid_request', 'client_id is required')

        request_id = secrets.token_urlsafe(32)
        self.pending_authorizations[request_id] = PendingAuthorization(
            client_id=client.client_id,
            params=params,
            expires_at=time.time() + self.auth_code_ttl_seconds,
        )
        return f'{self.public_url}/oauth/confirm?{urlencode({"request_id": request_id})}'

    def complete_authorization(self, request_id: str, approval_password: str) -> str | None:
        if not hmac.compare_digest(approval_password, self.approval_password):
            return None

        pending = self.pending_authorizations.pop(request_id, None)
        if pending is None or pending.expires_at < time.time():
            return None

        code = secrets.token_urlsafe(32)
        scopes = pending.params.scopes or self.scopes
        authorization_code = AuthorizationCode(
            code=code,
            scopes=scopes,
            expires_at=time.time() + self.auth_code_ttl_seconds,
            client_id=pending.client_id,
            code_challenge=pending.params.code_challenge,
            redirect_uri=pending.params.redirect_uri,
            redirect_uri_provided_explicitly=pending.params.redirect_uri_provided_explicitly,
            resource=pending.params.resource,
        )
        self.authorization_codes[code] = authorization_code

        return construct_redirect_uri(
            str(pending.params.redirect_uri),
            code=code,
            state=pending.params.state,
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.authorization_codes.get(authorization_code)

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        self.authorization_codes.pop(authorization_code.code, None)
        return self._issue_tokens(
            client_id=authorization_code.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self.refresh_tokens.get(refresh_token)
        if token and token.expires_at and token.expires_at < int(time.time()):
            self.refresh_tokens.pop(refresh_token, None)
            self._save_tokens()
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self.refresh_tokens.pop(refresh_token.token, None)
        self._save_tokens()
        return self._issue_tokens(
            client_id=refresh_token.client_id,
            scopes=scopes,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        access_token = self.access_tokens.get(token)
        if access_token and access_token.expires_at and access_token.expires_at < int(time.time()):
            self.access_tokens.pop(token, None)
            self._save_tokens()
            return None
        return access_token

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.access_tokens.pop(token.token, None)
        self.refresh_tokens.pop(token.token, None)
        self._save_tokens()

    def _issue_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None = None,
    ) -> OAuthToken:
        access_token_value = secrets.token_urlsafe(48)
        refresh_token_value = secrets.token_urlsafe(48)
        expires_at = int(time.time()) + self.token_ttl_seconds

        access_token = AccessToken(
            token=access_token_value,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=resource,
        )
        refresh_token = RefreshToken(
            token=refresh_token_value,
            client_id=client_id,
            scopes=scopes,
        )

        self.access_tokens[access_token_value] = access_token
        self.refresh_tokens[refresh_token_value] = refresh_token
        self._save_tokens()

        return OAuthToken(
            access_token=access_token_value,
            token_type='Bearer',
            expires_in=self.token_ttl_seconds,
            refresh_token=refresh_token_value,
            scope=' '.join(scopes),
        )
