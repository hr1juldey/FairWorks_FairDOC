# Fairdoc AI Triage System – Incremental “v2-on-Plug-in” Execution Plan  

*Goal: restore and freeze v1 exactly as it runs today, grow v2 side-by-side behind the `NEXT_GEN` flag without ever breaking the legacy API or exceeding 200-line file caps.*

## 1  |  Guiding Principles  

1. **Frozen Baseline (`v0.2.5-stable`)**  
   -  Tag and protect the branch; never edit v1 code in-place.  
2. **Feature-Flag Isolation**  
   -  One flag only: `NEXT_GEN=true` → mount v2 sub-app and load heavy deps.  
3. **One-File-One-Responsibility**  
   -  ≤ 200 LOC target (300 LOC absolute cap) checked by CI.  
4. **Strangler-Fig Migration**  
   -  Extract one bounded context at a time into v2; route traffic gradually.  
5. **Contract-First**  
   -  v1 schemas stay immutable; v2 introduces new Pydantic models under `src/app_v2/...`.  
6. **100% Test Coverage Delta**  
   -  Every new file ships with unit + integration tests; E2E updated weekly.  

## 2  |  Repository Layout After Day 3  

```
fairdoc_ai_tri```/
│
├── src```   ├── app```             ```↔ v1 code (LOCKED)
│  ```─ app_v2/            ```↔ all next```n code
│      ```─ api/
│      ```  ├── v2/
│      ```  │   ├── endpoints``` # ≤200 LOC each```      │   │```└── __init__.py
│      ```  └── __init__.py
│      ```─ core/
│      ```  ├── config```        # loads```nv, handles```XT_GEN
│      ```  ├── dependencies```  # only v```eavies
│      ```  ├── dependencies```.py  # copied```ght deps (≤120 ```)
│       │```└── context```       │      ```─ versioning.py  ```Redis Git-like layer```      │      ```─ __init__.py
│      ```─ services```       │  ```─ safety_gateway/
│      ```  ├── awe_router/
│      ```  ├── rag_service/
│      ```  └── __init__.py
│      ```─ main.py          ```orchestrates mount```├── tests/
│  ```─ unit/
│  ```─ integration```   └── e2e/
└── docker```mpose.dev.yml
```

## 3  |  Phase-0 Hot-Fix (Day 0-1) – Unbreak v1  

| File (≤ 200 LOC) | Action | Rationale |
|------------------|--------|-----------|
| `src/app/core/dependencies_v1.py` | **NEW** – paste pre-DSPy providers only. | Keeps v1 cold-start time tiny. |
| `src/app/main.py` | Wrap router inclusion: `if settings.NEXT_GEN: app.mount("/api/v2", v2_subapp)` | Hard isolation. |
| `src/app/core/context/manager.py` | Add `legacy_message()` shim (≤30 LOC). | Accept both tuple & dict history. |
| `src/app/services/ai/ollama_service.py` | Safe-get patch (4 LOC). | Works with tuples. |
| `tests/e2e/test_chat_v1.py` | Assert chat works with `NEXT_GEN=false`. | Regression guard. |

Tag `v0.2.6-hotfix`.

## 4  |  Phase-1 Foundation (Week 1, Roadmap D-1 → D-7)  

### 4.1 New Skeleton Micro-Services  

| Service | FastAPI file (≤ 150 LOC) | Docker exposure | Route prefix |
|---------|--------------------------|-----------------|--------------|
| Safety Gateway | `safety_gateway/main.py` | `8002` | `/validate` |
| AWE Router     | `awe_router/main.py`     | `8003` | `/route` |
| RAG Service    | `rag_service/main.py`    | `8004` | `/search` |

Each returns `{"status": "not_implemented"}` plus health-check.

### 4.2 Sub-App Mount Logic  

```python
# src```p_v2/main.py  ```120 LOC)
from fast``` import Fast```
from .core.config import```ttings
from .```.v2 import```uter as v2_router

v2_app```FastAPI(title="Fairdoc``` v2")
v2_app.include_router```_router)

def get```bapp():
   ```turn v2_app
```

In v1 `main.py`:

```python
if```ttings.NEXT_GEN:
   ```om src.app_v2.main import```t_subapp
   ```p.mount("/api/v2", get```bapp(), name```2")
```

### 4.3 Separate Swagger Docs  

* v1 docs at `/docs` (default).  
* v2 docs auto-namespaced at `/api/v2/docs`.

No tag collisions because v2 router uses its own FastAPI instance.

## 5  |  Phase-2 Context Engineering (Week 2)  

1. **`context/versioning.py` (≤ 180 LOC)**  
   * Redis commit, fork, merge, lineage traversal.  
   * Unit tests mock Redis with `fakeredis`.  
2. **Patch v1 context manager** to call versioning only when `NEXT_GEN`.  
3. **Prometheus Exporter** in `services/metrics.py` (≤ 100 LOC).  

## 6  |  Phase-3 Safety Gateway & DSPy Guardrails (Week 3)  

1. **Signature File** `medical_guardrails_signature.py` (≤ 80 LOC).  
2. **Gateway Endpoint** parses request, runs DSPy module, returns JSON flags.  
3. **Integration**: In v2 chat endpoint, call gateway pre- and post-generation.  
4. **Traffic Shadowing**: When `NEXT_GEN=true && SAFETY_SHADOW=true`, v1 responses are copied to gateway for dry-run scoring (no user impact).  

## 7  |  Phase-4 Progressive Feature Migration  

| Sprint | v1 change | v2 addition | Switch Mechanism |
|--------|-----------|-------------|------------------|
| S-4    | none      | **RAG search** endpoint → AWE | Shadow queries only |
| S-5    | none      | Multi-LLM router | 5% user bucket via header flag |
| S-6    | none      | Image/ECG analysis | Available only on `/api/v2/chat` |
| Go-Live | Deprecate `/api/v1/chat` *after* 60 days stable | Move all traffic | 308 redirect |

## 8  |  Testing & CI  

* **`pytest --cov` mandatory pass** on every PR.  
* **`tests/unit`** have per-file coverage.  
* **`tests/integration`** spin up docker-compose with stub services.  
* **`tests/e2e`** run real Redis/Postgres with `NEXT_GEN={false,true}` matrix.  
* GitHub Actions checks file line count with:

```bash
if [[ $(wc -l "$file" | cut -d' ' -f1) -gt 300 ]]; then```it 1; fi
```

## 9  |  Library Governance  

1. Before adding a **new pip package**:  
   -  create `ADR-###-add-.md` ➜ pros/cons, LOC impact, test changes.  
   -  require approval from **QA + Compliance**.  
2. Vend minimal wrappers (e.g., `fastapi-featureflags`) in `src/ext/` if tiny.  

## 10  |  Daily Developer Workflow  

```bash
# Day```-day
git switch``` feat/
poetry install``` or pip-tools sync```test -q
uvicorn src```p/main:app --reload ```v1
NEXT_GEN=true uv```rn src/app/main:app --reload``` v1+v2
```

## 11  |  First Three Tickets to Open Monday  

| Ticket | Description | Est. | Owner |
|--------|-------------|------|-------|
| **01 | Create `dependencies_v1.py`, patch main isolation, restore chat (Phase-0). | 0.5 d | BE-1 |
| **02 | Add v2 sub-app skeleton & mount, docs separation. | 0.5 d | BE-2 |
| **03 | Compose file for safety-gateway stub + health route + CI test. | 1 d | DevOps |

Once #101-#103 pass CI & E2E, tag `v0.2.7-pre-nextgen` and begin Week 1 roadmap.

### ✅ Outcome  

-  **v1** remains untouched, lightweight, and green.  
-  **v2** grows under strict LOC boundaries, behind flags, with full test coverage and no Swagger duplication.  
-  Developers follow a clear, granular roadmap that aligns with 30-/47-day plans while respecting single-file responsibilities and incremental micro-service rollout.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/9c354a49-9967-4746-afaf-83afb2348ffd/a-practical-guide-to-building-agents.pdf

[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/16cfb508-7846-4b44-8db2-1fd60c2566e1/Fairdoc-AI-Triage-System_-Detailed-47-Day_roadmap.md

[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/918ed1fa-fcd3-4fd4-b413-ceb927198340/TODO.md

[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/1ac6f7b8-adae-4c5c-bc7a-7909465fdb04/engineering_v2.md

[5] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/05014cfc-c793-43b9-953f-68c02670353d/brief_v2.md

[6] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/ff2154d5-d374-45b9-b20e-526b985ce08e/solution.md

[7] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/9dbcd5f8-5ae3-4c27-b7ff-34f92c0659ed/Forensics.md

[8] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/21593bfb-f630-4e0c-8ebc-fa24f7908e28/brief.md

[9] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/5f3865b2-335d-4fa4-8032-4a627a5ced35/30-Day-Implementation-Plan.md

[10] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/cfdfa5fb-65dc-41b4-8210-319e85ea405b/docker-compose.yml

[11] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/f376b0b9-dd94-4673-927a-b31ffbbb2bea/pyproject.toml

[12] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56357229/4158039e-cee3-426b-9d79-a10f475d9854/init.sql

[13] https://ieeexplore.ieee.org/document/9927331/

[14] https://ieeexplore.ieee.org/document/10175521/

[15] https://www.mdpi.com/2072-666X/13/8/1355

[16] https://ieeexplore.ieee.org/document/9745756/

[17] https://ieeexplore.ieee.org/document/9426840/

[18] https://ieeexplore.ieee.org/document/9599705/

[19] https://pubs.acs.org/doi/10.1021/acsanm.1c00582

[20] https://ieeexplore.ieee.org/document/9331660/

[21] https://ieeexplore.ieee.org/document/8972794/

[22] https://ieeexplore.ieee.org/document/10657315/

[23] https://dl.acm.org/doi/pdf/10.1145/3694715.3695976

[24] https://arxiv.org/pdf/2211.01473.pdf

[25] https://www.mdpi.com/2078-2489/11/2/108/pdf

[26] http://arxiv.org/pdf/2205.10133.pdf

[27] https://www.geeksforgeeks.org/python/fastapi-mounting-a-sub-app/

[28] https://github.com/tiangolo/fastapi/issues/641

[29] https://github.com/Pytlicek/fastapi-featureflags

[30] https://macsphere.mcmaster.ca/bitstream/11375/30255/2/Hassan_zaker_thesis%201.pdf

[31] https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf

[32] https://fastapi.tiangolo.com/advanced/sub-applications/

[33] https://stackoverflow.com/questions/77796418/overriding-a-route-dependency-in-fastapi

[34] https://www.vintasoftware.com/blog/taming-irreversibility-feature-flags-python

[35] https://www.index.dev/blog/monolithic-to-microservices-migration

[36] https://www.youtube.com/watch?v=tVw3CwrN5-8

[37] https://www.getorchestra.io/guides/fastapi-sub-applications-a-detailed-tutorial-with-examples

[38] https://github.com/tiangolo/fastapi/issues/2481

[39] https://dev.to/yanagisawahidetoshi/efficiently-using-environment-variables-in-fastapi-4lal

[40] https://brainhub.eu/library/monolith-to-microservices-using-strangler-pattern

[41] http://thesai.org/Downloads/Volume14No2/Paper_33-Optimized_Strategy_for_Inter_Service_Communication_in_Microservices.pdf

[42] https://zenodo.org/record/4550449/files/MAP-EuroPlop2020bPaper.pdf

[43] https://www.mdpi.com/1424-8220/20/18/5103/pdf

[44] http://arxiv.org/pdf/2305.16329.pdf

[45] https://arxiv.org/ftp/arxiv/papers/2309/2309.02058.pdf

[46] https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/cpe.4175

[47] https://github.com/BoxiYu/DSPy-Guardrails

[48] https://github.com/rifatrakib/fast-subs

[49] https://dev.to/uponthesky/python-a-simple-guide-how-to-mock-dependencies-for-unit-testing-in-fastapi-2g97

[50] https://launchdarkly.com/blog/python-flask-google-maps-api-feature-flags/

[51] https://www.xcubelabs.com/blog/microservices-architecture-the-ultimate-migration-guide/

[52] https://dspy.ai/production/