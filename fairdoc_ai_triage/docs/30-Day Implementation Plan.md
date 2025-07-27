# 30-Day Implementation Plan

Each day below is capped at ~ 6 h of focused engineering time per person. Sprint reviews every Friday afternoon. The running system (Phase 2.5) remains online; all new work is delivered behind feature flags or on side-car micro-services.

| Day | Goal & Deliverables | Rationale / Standards | Owner(s) | Success Check |
| :-- | :-- | :-- | :-- | :-- |
| **D-0 (Wed 23 Jul 25 – tonight)** | - Tag `v0.2.5` of current repo<br>- Enable dark-launch feature flag `NEXT_GEN=true` | Freeze working baseline | Dev Lead | Tag appears in Git & CI passes |
| **D-1** | **Architecture Kick-off**<br>- Approve Draft-2 PRD<br>- Finalise micro-service boundaries (Context, Safety, AWE)<br>- Create ADR-001 (Why FHIR R4B) & ADR-002 (DSPy Guardrails) | Align all leads | CTO + Arch Team | ADRs merged |
| **D-2** | **Service Skeletons**<br>- Generate FastAPI stubs for: `safety-gateway`, `awe-router`, `rag-service`<br>- gRPC IDL drafted | Keeps files <300 LOC | Backend | Stubs compile; contracts in repo |
| **D-3** | **FHIR Compliance Layer**<br>- Add `fhir.resources` Pydantic models<br>- “EvidenceBundle” schema + unit tests | Removes hard-coding | Compliance | `pytest` green |
| **D-4** | **Dynamic Metrics Registry**<br>- Replace hard-coded scores with `metrics.yml` + Prometheus exporter | Observability | DevOps | Grafana shows new panel |
| **D-5 (Fri review)** | Sprint-1 demo: Skeleton services running in docker-compose-dev; gateway returns 501 | — | All | Review sign-off |

## Week-2 (S-2) — DSPy Guardrails \& Safety Gateway

| Day | Tasks |
| :-- | :-- |
| **D-8** | Integrate DSPy `BootstrapFewShot` compiler in `safety-gateway`; import NICE CG95 examples as gold set (≈ 15) |
| **D-9** | Implement reward function: pass if answer contains FHIR `CarePlan` reference \& zero hallucinations (per WangLab method) |
| **D-10** | Auto-compile guardrail prompt; expose `/validate` endpoint returning flag list |
| **D-11** | Connect chat endpoint → safety-gateway (pre- \& post-filter) via async HTTP; feature-flagged |
| **D-12 (Fri)** | Load-test 1 k rps through gateway; ASR ≤ 10% on test set |

## Week-3 (S-3) — Knowledge Fabric (GraphRAG)

| **Day**      | **Task** |
|--------------|----------|
| **D-15**     | Load NICE guidelines JSON → pgvector & Chroma; build `MedicalRAGService.search()` |
| **D-16**     | Build retrieval pipeline returning 2+ unique `DocumentReference` resources |
| **D-17**     | Add provenance builder: SHA-256 + `FHIR.Provenance` |
| **D-18**     | Extend AWE router: if query class ∈ {symptom, drug}, call RAG first |
| **D-19 (Fri)** | End-to-end demo: query → evidence bundle → safety → response with citations |

## Week-4 (S-4) — Multi-LLM Pool \& Scorecard

| **Day**       | **Task** |
|---------------|----------|
| **D-22**      | Stand-up vLLM server with Mixtral-8x7B; register in `LLMRegistry.yml` |
| **D-23**      | Implement scorecard collector (latency, cost, safety flags) → Postgres |
| **D-24**      | AWE routing policy YAML → selects best model under SLA (≤10 s, safety ≤5 flags) |
| **D-25**      | Fallback logic: DeepSeek → Mixtral → GPT-4o |
| **D-26 (Fri)**| Chaos drill: kill DeepSeek container, observe automatic model switch |

## Week-5 (S-5) — Multimodal Perception (ECG-Chat MVP)

| **Day**        | **Task** |
|----------------|----------|
| **D-29**       | Containerise Bio-ViT image model; expose `/analyze/xray` |
| **D-30**       | Speech pipeline: Whisper-Medical STT → text → chat flow |
| **D-31**       | DICOM-SR → FHIR.ImagingStudy transformer |
| **D-32**       | Update safety-gateway to include image modality flags |
| **D-33 (Fri)** | Demo: upload chest-X-ray → AI triage with citations |

## Week-6 (S-6) — Compliance \& Audit

| **Day**        | **Task** |
|----------------|----------|
| **D-36**       | Implement `FHIR.Provenance` logger middleware across services |
| **D-37**       | Encrypt MinIO buckets; rotate KMS keys |
| **D-38**       | Kibana dashboard for provenance + safety metrics |
| **D-39**       | HIPAA gap analysis report |
| **D-40 (Fri)** | Pen-test on staging; zero critical vulns |

## Week-7 (S-7) — Production Readiness

| **Day**        | **Task** |
|----------------|----------|
| **D-43**       | Canary deploy to staging cluster; monitor metrics |
| **D-44**       | Chaos monkey tests: Redis outage, LLM latency spike |
| **D-45**       | Auto-rollback scripts verified |
| **D-46**       | Legal review: NICE / FHIR conformity evidence |
| **D-47 (Fri)** | Go-live readiness check; tag `v1.0-rc1` |

### Ongoing (Daily)

- **Morning 15 min** stand-up
- **Evening 10 min** safety-flag triage
- **Weekly** sprint retro \& roadmap adjust

**You now have a day-by-day, non-disruptive execution plan that layers DSPy guardrails, standards compliance, multi-LLM routing, GraphRAG knowledge, and multimodal perception—without breaking the current service.**
Let me know which sprint you’d like to refine further.

<div style="text-align: center">⁂</div>

[^1]: Fairdoc-AI-Triage-System.md

[^2]: Medical-AI-Triage-System_-Progressive-Architecture.md
