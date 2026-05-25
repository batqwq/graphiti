# Embedding Setup for Graphiti MCP on Windows

This document describes the current self-hosted Graphiti MCP setup at:

```text
H:\AI-Memory\graphiti
```

The active embedding path is Google AI Studio direct API-key auth through Graphiti's `GeminiEmbedder`. OpenRouter is kept only as a commented fallback because the current account receives HTTP `403` from the OpenRouter Gemini embedding provider route.

## Current Choice

Plan A is active:

```env
GOOGLE_API_KEY=
GRAPHITI_EMBEDDING_PROVIDER=gemini
GRAPHITI_EMBEDDING_MODEL=gemini-embedding-2
GRAPHITI_EMBEDDING_DIM=1536
GRAPHITI_EMBEDDING_BATCH_SIZE=1
```

The requested candidates were tested first:

- `gemini-embedding-exp-03-07`: unavailable for this API key on Gemini API `v1beta`.
- `text-embedding-004`: unavailable for this API key on Gemini API `v1beta`.
- `gemini-embedding-2`: available and verified with 1536-dimensional output.

Graphiti's current `GeminiEmbedder` uses:

```python
genai.Client(api_key=config.api_key)
```

That is Google AI Studio API-key authentication, not Vertex AI or ADC. The embedder calls Gemini API `embed_content` with `output_dimensionality=1536`.

## Project Structure

- MCP startup entry: `mcp_server\main.py`
- MCP server implementation: `mcp_server\src\graphiti_mcp_server.py`
- Embedding config: `mcp_server\config\config.yaml`
- Embedder factory: `mcp_server\src\services\factories.py`
- Gemini embedder: `graphiti_core\embedder\gemini.py`
- OpenAI embedder: `graphiti_core\embedder\openai.py`
- OpenRouter fallback adapter: `mcp_server\src\services\openrouter_embedder.py`
- Local Kuzu driver fix: `graphiti_core\driver\kuzu_driver.py`

`OpenAIEmbedder` supports `api_key`, `base_url`, and `embedding_model`, and Graphiti config maps `api_url` into `base_url`. It also has `embedding_dim`, but the stock OpenAI embedder slices vectors locally instead of sending an upstream `dimensions` parameter. The OpenRouter fallback adapter sends `dimensions` to OpenRouter explicitly.

## Clone And Install

```powershell
New-Item -ItemType Directory -Force H:\AI-Memory | Out-Null
git clone https://github.com/getzep/graphiti.git H:\AI-Memory\graphiti
Set-Location H:\AI-Memory\graphiti

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -e ".[falkordb,kuzu,google-genai]" -e ".\mcp_server"
```

## Create `.env`

Create `H:\AI-Memory\graphiti\mcp_server\.env` without a UTF-8 BOM so the first key is parsed correctly:

```powershell
$envText = @'
GOOGLE_API_KEY=
GRAPHITI_EMBEDDING_PROVIDER=gemini
GRAPHITI_EMBEDDING_MODEL=gemini-embedding-2
GRAPHITI_EMBEDDING_DIM=1536
GRAPHITI_EMBEDDING_BATCH_SIZE=1
GEMINI_RERANKER_MODEL=

LLM_PROVIDER=fireworks
FIREWORKS_API_KEY=
FIREWORKS_API_URL=https://api.fireworks.ai/inference/v1
MODEL_NAME=accounts/fireworks/models/minimax-m2p7
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=4096

DATABASE_PROVIDER=kuzu
KUZU_DB=H:\AI-Memory\graphiti\data\kuzu
KUZU_MAX_CONCURRENT_QUERIES=1
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10

# Plan B only. Leave commented/empty while OpenRouter returns 403.
# OPENROUTER_API_KEY=
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# GRAPHITI_EMBEDDING_PROVIDER=openrouter
# GRAPHITI_EMBEDDING_MODEL=google/gemini-embedding-2-preview
'@
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText("H:\AI-Memory\graphiti\mcp_server\.env", $envText, $utf8NoBom)
```

Then paste real keys into `mcp_server\.env`:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
FIREWORKS_API_KEY=your_fireworks_key
```

Do not commit `.env`.

## Run The Embedding Test

```powershell
Set-Location H:\AI-Memory\graphiti
.\.venv\Scripts\python.exe scripts\test_embedding.py
```

Expected success shape:

```text
Embedding test passed: provider=gemini, model=gemini-embedding-2, 3 embeddings, 1536 dimensions each.
```

The test reads the active provider from `.env`, sends three text inputs, and asserts:

- exactly 3 embeddings are returned
- every vector has 1536 dimensions

## Start Graphiti MCP Server

```powershell
Set-Location H:\AI-Memory\graphiti\mcp_server
& H:\AI-Memory\graphiti\.venv\Scripts\python.exe main.py --transport http --host 127.0.0.1 --port 8000
```

Check the health endpoint from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

MCP clients should connect to:

```text
http://127.0.0.1:8000/mcp/
```

## Switching Plan A And Plan B

Plan A, Google AI Studio direct:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
GRAPHITI_EMBEDDING_PROVIDER=gemini
GRAPHITI_EMBEDDING_MODEL=gemini-embedding-2
GRAPHITI_EMBEDDING_DIM=1536
GRAPHITI_EMBEDDING_BATCH_SIZE=1
```

Plan B, OpenRouter after 403 is fixed:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GRAPHITI_EMBEDDING_PROVIDER=openrouter
GRAPHITI_EMBEDDING_MODEL=google/gemini-embedding-2-preview
GRAPHITI_EMBEDDING_DIM=1536
GRAPHITI_EMBEDDING_BATCH_SIZE=1
```

No code change is required for the switch.

## Why Batch Size Is 1

Gemini embedding provider behavior can vary by model and endpoint. `GRAPHITI_EMBEDDING_BATCH_SIZE=1` is the conservative setting for a memory server: slower, but predictable. The Gemini embedder also falls back to individual embedding calls if a batch request fails.

## Why 1536 Dimensions

Google's current Gemini embedding documentation lists flexible output dimensions from 128 to 3072 and recommends 768, 1536, or 3072. `1536` is the middle option: materially richer than 768, but cheaper to store and search than 3072.

## Cost And Rate Limits

Google AI Studio free-tier limits are account/model dependent and can change; check the live Gemini API rate-limit page before relying on a fixed number. For OpenRouter fallback, `google/gemini-embedding-2-preview` is listed at about `$0.20 / 1M input tokens` with no output-token cost for embeddings, but the current local OpenRouter route returns HTTP `403`.

## Operation Log

Date: 2026-05-25

Operator: Codex

Current runtime observed in logs:

```text
LLM: fireworks / accounts/fireworks/models/minimax-m2p7
Embedder: gemini / gemini-embedding-2
Database: kuzu
Transport: http
```

Validation completed:

```text
scripts\test_embedding.py -> passed
/health                   -> healthy
get_status                -> OK
search_memory_facts       -> returned facts from Kuzu
```

Security notes:

- Do not commit `mcp_server\.env`.
- API keys are read from environment variables or `.env`.
- The Google API key was pasted during setup; rotate it after deployment is stable.

## Files Added Or Modified

Added:

```text
docs\embedding-setup.md
scripts\test_embedding.py
scripts\test_gemini_embedding.py
scripts\test_openrouter_embedding.py
mcp_server\src\services\cross_encoder.py
mcp_server\src\services\openrouter_embedder.py
```

Modified:

```text
.env.example
.gitignore
graphiti_core\driver\kuzu_driver.py
mcp_server\.env.example
mcp_server\README.md
mcp_server\config\config.yaml
mcp_server\pyproject.toml
mcp_server\src\config\schema.py
mcp_server\src\graphiti_mcp_server.py
mcp_server\src\services\factories.py
```

## Rollback

Restore modified tracked files:

```powershell
Set-Location H:\AI-Memory\graphiti
git restore .env.example .gitignore graphiti_core\driver\kuzu_driver.py mcp_server\.env.example mcp_server\README.md mcp_server\config\config.yaml mcp_server\pyproject.toml mcp_server\src\config\schema.py mcp_server\src\graphiti_mcp_server.py mcp_server\src\services\factories.py
```

Remove added files:

```powershell
Remove-Item docs\embedding-setup.md, scripts\test_embedding.py, scripts\test_gemini_embedding.py, scripts\test_openrouter_embedding.py, mcp_server\src\services\cross_encoder.py, mcp_server\src\services\openrouter_embedder.py -Force
```
