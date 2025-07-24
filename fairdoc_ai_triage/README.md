# Fairdoc AI Triage System

A standards-aligned, safety-first medical-AI platform that evolves from a single-LLM chat prototype to a multi-modal, multi-LLM, FHIR-compliant triage assistant.

---

## 🌟 Key Features

| Capability | Status | Standards |
|------------|--------|-----------|
| Context-aware chat | ✅ | — |
| DSPy safety gateway | ✅ | NICE CG95 |
| FHIR R4B evidence bundle | ✅ | HL7 FHIR |
| Multi-LLM pool (DeepSeek, Mixtral, GPT-4o) | 🔄 | — |
| Knowledge GraphRAG | 🔄 | PubMed, NICE |
| Multimodal (image, voice, ECG) | 🔄 | DICOM-SR |
| HIPAA / GDPR audit | 🔄 | FHIR Provenance |

---

## 🛠️ Quick-Start (5 min)

```bash
# 1. clone & switch to stable tag
$ git clone https://github.com/hr1juldey/FairWorks_FairDOC.git
$ cd FairWorks_FairDOC && git checkout v0.2.5-stable

# 2. copy & edit .env
$ cp .env.example .env

# 3. start core stack (DeepSeek only)
$ docker compose up -d

# 4. open docs
→ http://localhost:8000/docs
```

### Enable **next-gen** micro-services

```bash
$ export NEXT_GEN=true        # feature flag
$ docker compose -f docker-compose.dev.yml up -d  # starts safety-gateway, rag-service, awe
```

---

## 📂 Repository Map

```text
├─ docker/              # container configs
├─ src/app/             # Phase-2.5 monolith (still live)
├─ src/services/        # NEW micro-services (NEXT_GEN)
│   ├─ safety_gateway/
│   ├─ knowledge_fabric/
│   └─ awe_orchestrator/
├─ docs/                # markdown docs rendered by GitHub Pages
└─ README.md            # you are here!
```

---

## 🚀 Development Workflow

1. **Fork & branch**: `feat/<ticket>`  
2. **Rule**: every Python/MD file ≤ 300 LOC  
3. `make lint && make test` must pass  
4. PR → CI → staging (feature flag OFF)

---

## 📖 Documentation Index

| Doc | Purpose |
|-----|---------|
| [`docs/overview.md`](docs/overview.md) | Big-picture goals & roadmap |
| [`docs/setup.md`](docs/setup.md) | Idiot-proof install & bootstrap |
| [`docs/architecture.md`](docs/architecture.md) | Component-level detail & Mermaid graphs |
| [`docs/development.md`](docs/development.md) | Coding standards & contribution guide |
| [`docs/sprints.md`](docs/sprints.md) | 47-day granular plan |

---

## 🤝 Contributing

We welcome PRs from clinicians, ML engineers, and open-source enthusiasts. See [`docs/development.md`](docs/development.md).

---

© 2025 Fairdoc AI Foundation — Apache 2.0