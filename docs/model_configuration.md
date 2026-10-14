# Effective research model configuration (S93)

Compiler entity extraction/disambiguation, CodeAgent synthesis, critique, and
same-memory finalization share one provider/model configuration. There are no
active stage model overrides. Select one of these configurations:

| Setting | Flash baseline | Luna challenger |
| --- | --- | --- |
| `LLM_BACKEND` | `deepseek` (default) | `openai` |
| Model variable | `DEEPSEEK_MODEL=deepseek-flash` (default) | `OPENAI_MODEL=gpt-5.6-luna` (default) |
| Credential | `DEEPSEEK_API_KEY` | `OPENAI_API_KEY` |
| Endpoint | `https://api.deepseek.com` (also accepts `/v1`) | `https://api.openai.com/v1` |
| Output budget variable | `DEEPSEEK_MAX_TOKENS=32768` | `OPENAI_MAX_TOKENS=32768` |
| `LLM_REASONING_EFFORT` default | `high` | `medium` |
| Supported effort values | `none`, `low`, `high`, `max` | `none`, `low`, `medium`, `high`, `xhigh`, `max` |
| Context denominator | 1,000,000 (conservative decimal 1M) | 1,050,000 |
| Maximum output budget | 393,216 | 128,000 |
| Wire token argument | `max_tokens` | `max_completion_tokens` |

Current limits and exact names were checked September 19, 2026 against
[DeepSeek's request reference](https://api-docs.deepseek.com/api/create-chat-completion/),
[DeepSeek's model details](https://api-docs.deepseek.com/quick_start/pricing/), and
[the Luna model reference](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
OpenAI's output budget includes reasoning tokens; see the
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Precedence and migration

Explicit CLI/worker backend and model arguments take precedence over environment
values. They are applied to startup settings and passed to the spawned mission
child. Exported environment values take precedence over `.env`; missing values
use the defaults above. Only the selected provider's model, key, endpoint, and
budget are read. Both credentials may be provisioned without ambiguity. Changing
`LLM_BACKEND` changes all active stages. `LLM_REQUEST_TIMEOUT_SECONDS` controls
all requests (default 300 seconds).

| Previous setting/behavior | S93 behavior |
| --- | --- |
| Compiler-only `LLM_BACKEND` | Selects every active stage |
| Hard-coded synthesis/critique `deepseek-v4-pro` | Resolved shared setting; explicit `deepseek-v4-pro` remains available for rollback |
| `deepseek-reasoner`, `deepseek-chat`, unknown aliases | Rejected for production; choose an exact supported ID |
| `REFLECTION_*_MODEL` | Retired development helper only; does not select the production critic |
| `DEEPSEEK_TEMPERATURE` | Retired development setting; active requests omit temperature |
| Gemini, MiniMax, Ollama adapters | Retained for legacy development callers; worker preflight rejects them |
| Custom provider base URL | Rejected before dispatch; keys cannot be sent to a different origin |
| Unknown backend or malformed token/timeout value | Preserved as invalid and diagnosed by preflight; no silent rerouting |

Preflight v5 publishes an `effective_model` check with redacted configuration,
including credential **variable name**, endpoint, requested model, reasoning and
limits. It makes no inference call and does not use `/models` as a compatibility
probe. A configured key does not establish account access. Runtime configuration
fingerprints and synthesis telemetry carry the same effective metadata. Context
pressure is unassessed when an injected/old result has no measured model config.

## Transport and failure behavior

The installed stack is smolagents 1.24.0, openai 1.109.1 and langchain-openai
0.3.35. Chat Completions remains the transport. smolagents omits the stop argument
for GPT-5 models and trims stop sequences locally. It preserves the raw response
for finish-reason checks. Its model defaults take precedence over call kwargs.
The compiler uses JSON-in-prompt plus Pydantic validation, not a promise of strict
schema generation. LangChain rewrites token arguments unconditionally; DeepSeek's
budget is therefore passed through SDK `extra_body`, verified on serialized HTTP
requests. Both targets omit temperature. DeepSeek explicitly enables/disables
thinking with the selected effort.

Compiler permanent request/authentication/configuration errors stop after one
request. Transient transport, rate-limit and server failures remain bounded by
three wrapper attempts, with SDK retries disabled. Safe category, allowlisted
provider code, HTTP status, attempt, stage and requested model persist as
`provider_failure` rows in the existing contract `entity_extraction_decisions`
surface when deterministic fallback is used. No exception message or response
body enters these rows or logs. Truncated, filtered or tool-call responses are
not accepted as structured extraction. Cancellation propagates through compiler
bridges; the existing spawned-process fence remains the hard cancellation bound.

The local construction fallback is capped at the lesser of the selected output
budget and 16,384 tokens. It only handles local `ValueError`; it cannot retry an
API request. Every same-memory finalization call retains the selected model.
S92's once-only recovery acceptance still requires provider `stop` plus nonempty
cited output. No full-run restart follows paid tokens or retained action memory.

Offline wire tests establish request construction and orchestration. They do not
prove account access, provider inference compatibility or research quality; those
remain for the bounded S93 evaluation after offline acceptance.
