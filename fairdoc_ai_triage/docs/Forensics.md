<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Updated Stabilization \& Extension Plan for Fairdoc AI

Before adding new power features, we must **restore the rock-solid v1 chat and then layer v2 on as a plug-in**.
All work below respects your 200-300 line/file cap and leaves every v1 endpoint contract untouched.

## 1 — Root-Cause of the Current “No-Chat” Failure

| Layer | Mandatory Fix | Why it Breaks Chat | Evidence |
| :-- | :-- | :-- | :-- |
| `core/dependencies.py` | YES | New functions (`get_safety_validator`, `get_keyword_service`, etc.) are imported even when a v1 endpoint calls `Depends()`. They instantiate heavy DSPy modules in the event loop and blow up if DSPy is still booting. | Trace shows DSPy initialising before `lifespan.startup` completes, then FastAPI never returns a response. |
| `src/app/main.py` | YES | Both v1 \& v2 routers are included unconditionally; any request to `/api/v1/chat` still triggers v2-only startup code (keyword tables, DSPy). | Uvicorn logs display safety modules initialising even on v1-only requests. |
| `services/ai/ollama_service.py` (next-gen) | YES | It now expects every `ConversationContext.messages` element to be a dict. The stable v1 stored raw strings → `KeyError`. | Regression from stable branch diff. |
| `src/app/core/context/manager.py` | YES | Stable branch returned list of `(role,text)` tuples; next-gen switched to full dict objects. v1 chat still passes tuples, breaking the new `OllamaService._build_system_prompt()`. |  |

## 2 — High-Level Solution Architecture

1. **Hard isolation flag**

```python
# src/app/main.py
USE_NEXT_GEN: bool = settings.NEXT_GEN  # already in .env
```

*Routers \& heavy services initialise only when `USE_NEXT_GEN` is True.*
2. **Dual dependency trees**
    * Create **`dependencies_v1.py`** (≤120 lines) that reproduces the light-weight objects from 0.2.5.
    * Keep current `dependencies.py` as v2 provider.
    * `src/app/api/v1/router.py` switches its import to the new file.
3. **Stable data-model shim**
Add a 30-line converter in `context/manager.py`:

```python
def legacy_message(record: Union[Tuple[str,str], Dict]) -> Dict:
    if isinstance(record, dict):
        return record
    role, text = record
    return {"user_message" if role=="user" else "ai_response": {"text": text}}
```

4. **Service plug-in pattern for v2**
v2 endpoints wrap v1 logic instead of forking it:

```python
async def chat_v2(req: ChatV2Request, ...):
    v1_resp = await chat_core(req.message, req.session_id)
    extra = await safety_validator.enhance(v1_resp, req.context)
    return {**v1_resp, "safety": extra}
```

5. **Immutable v1 API**
No signature or schema changes; only internal fixes hidden behind shims.

## 3 — File-by-File Action List

| File (≤200-300 lines after edit) | Mandatory Change | Summary of Edits |
| :-- | :-- | :-- |
| **src/app/main.py** | YES | a) gate router inclusion by `USE_NEXT_GEN`; b) start only required services for chosen version. |
| **src/app/core/dependencies.py** | YES | rename heavy providers with `_v2_…`; export nothing to v1. |
| **NEW ⇒ src/app/core/dependencies_v1.py** | YES | copy light objects from 0.2.5 (`get_db`, `get_context_manager`, `get_ollama_service`). No DSPy, no keyword service. |
| **src/app/api/v1/router.py** | YES | import `dependencies_v1`; ensure only `/chat`, `/health`, `/thinking` are registered. |
| **src/app/core/context/manager.py** | YES | add `legacy_message()` shim and call it inside `add_message()` \& `get_context()`. |
| **src/app/services/ai/ollama_service.py** | YES | Guard dict key access with `dict.get()` so tuple fallback works (4-line patch). |
| **src/app/core/logging.py** | OPTIONAL | Lower default log level for v1 to suppress DSPy noise when unused. |
| **src/app/api/v2/router.py** | OPTIONAL | Leave as is, but mark each include with `tags=["chat-v2"]`, etc. to avoid Swagger collisions. |

No other files touched; v1 code size remains <300 lines each.

## 4 — Implementation Steps (Solo-Dev Friendly)

1. **Morning (2 hrs)**
    - Copy `dependencies_v1.py` from 0.2.5 branch.
    - Patch `main.py` flag logic \& router selection.
2. **Mid-day (1 hr)**
    - Apply 4-line safe-get fix in `ollama_service.py`.
    - Add `legacy_message()` shim to context manager.
3. **Afternoon (1 hr)**
    - Update v1 router import \& tidy Swagger tags.
4. **Smoke tests (30 min)**
`uvicorn src.app.main:app --reload` → call
`curl -X POST localhost:8000/api/v1/chat -d '{"message":"hi"}'`.
5. **Commit \& Tag**
`git commit -m "fix: isolate v1 and restore chat"; git tag v0.2.6-hotfix`.

## 5 — Change Classification

| Change | Category |
| :-- | :-- |
| Router gating, light deps, context shim | **Mandatory** (chat currently broken) |
| Swagger tag cleanup, log-level tweak | **Strongly recommended** (dev-UX) |
| Future vector/search endpoints in v2 | **Idea** (next phase, not now) |

## 6 — Guardrails to Keep You Within Line Limits

* Write all new helpers in their owning file (`dependencies_v1.py`) — 1 purpose, 1 file.
* Keep every patch surgical; avoid cross-concerns.
* Add `# <300 lines>` comment header; your CI can grep for it later.


## 7 — After v1 Is Green

Only then proceed to:

1. Re-enable `NEXT_GEN=true` locally to confirm v2 still boots.
2. Finish TODO-driven work (pgvector proof routes, analytics, etc.).
3. Incrementally migrate selected clients to v2.

### Next Command to Run

```bash
# restore v1 chat in dev
export NEXT_GEN=false
uvicorn src.app.main:app --reload
```

Once you see **✅ “Fairdoc AI Triage System started – v1-only mode”** in the logs, the hotfix is successful.

<div style="text-align: center">⁂</div>

[^1]: Forensics.md

[^2]: dspy_config.py

[^3]: medical_constants.py

[^4]: ollama_service.py

[^5]: thinking_processor.py

[^6]: emergency_handler.py

[^7]: evaluation_metrics.py

[^8]: keyword_service.py

[^9]: prediction_processor.py

[^10]: safety_signatures.py

[^11]: safety_validator.py

[^12]: medical_keyword_db.py

[^13]: Fairdoc-AI-Triage-System_-Detailed-47-Day_roadmap.md

[^14]: TODO.md

[^15]: https://github.com/hr1juldey/FairWorks_FairDOC/tree/38bacf434ef18618ab7b7a26e603dec84887ec30/fairdoc_ai_triage/src/app/api/v1/endpoints

[^16]: https://github.com/hr

