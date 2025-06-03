# Technical Architecture Overview
# 🔧 Technical Architecture Overview

This section details the technical architecture of Fairdoc AI, broken down by layers and components.

## Overall System Architecture (Conceptual v1.0)

```mermaid
graph TD
    subgraph "User Interaction Layer"
        WA[📱 WhatsApp] --> GW
        TG[🤖 Telegram] --> GW
        SL[💬 Signal] --> GW
        WebUI[🌐 Web UI (Mesop for v0)] --> GW
    end

    subgraph "Gateway & Orchestration"
        GW[ FastAPI API Gateway \n (Auth, Routing, WebSocket)] --> AO[🤖 AI Orchestration Engine]
        AO --> MS[🧠 Model Selector (Ollama)]
        AO --> BM[⚖️ Bias Monitor]
        AO --> CM[📦 Context Manager (RAG)]
    end

    subgraph "Core Services Layer"
        CM --> VS[📚 ChromaDB Vector Store \n (Clinical Guidelines, Case Embeds)]
        AO --> MQ[🐇 Celery Task Queue]
        MQ --> Redis[🗃️ Redis Cache \n (Session, Temp Data)]
    end

    subgraph "Specialized AI/ML Workers (via Celery)"
        W_NLP[📜 Text NLP Service \n (Gemma 4B, mxbai-embed)] --> MQ
        W_IMG[🖼️ Image Analysis Service \n (OpenCV, SAM2.1-mocked, Gemma 4B-VLM)] --> MQ
        W_AUD[🔊 Audio Processing (Future)] --> MQ
        W_RSK[⚕️ Risk Classifier \n (Reasoning LLM - DeepSeek/Mistral-like)] --> MQ
        W_PDF[📄 PDF Report Generator \n (ReportLab)] --> MQ
    end

    subgraph "Data & Integration Layer"
        W_NLP --> DB[(🗄️ PostgreSQL DB \n Users, Cases, Protocols)]
        W_IMG --> DB
        W_RSK --> DB
        W_PDF --> DB
        W_PDF --> FS[📦 MinIO/S3 Object Storage \n (Files, PDF Reports)]
        W_IMG ----> FS
        GW ----> FS 
        DB <--> VS 
        EHR[🏥 NHS/ABDM EHR (Future)] <--> AO
    end

    subgraph "Monitoring & Logging"
        GW --> LOG
        AO --> LOG
        MQ --> LOG
        DB --> LOG
        LOG[🕵️ Logging (ELK/Prometheus) & Sentry]
    end

    classDef user fill:#E0F7FA,stroke:#00796B;
    classDef gateway fill:#FFF3E0,stroke:#E65100;
    classDef core fill:#EDE7F6,stroke:#512DA8;
    classDef aiml fill:#F3E5F5,stroke:#8E24AA;
    classDef data fill:#E8F5E9,stroke:#2E7D32;
    classDef monitor fill:#ECEFF1,stroke:#37474F;

    class WA,TG,SL,WebUI user;
    class GW,AO,MS,BM,CM gateway;
    class VS,MQ,Redis core;
    class W_NLP,W_IMG,W_AUD,W_RSK,W_PDF aiml;
    class DB,FS,EHR,VS data;
    class LOG monitor;

```
