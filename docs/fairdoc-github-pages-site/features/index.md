# 🧩 Core Features & Modules (v1.0 & Beyond)

This section outlines the comprehensive features of Fairdoc AI. The [v0 Prototype](./v0-prototype-user-story.md) implements a subset of these.

## 📱 Chat Frontend & Patient Interaction
* **Multi-platform Connectors:** Seamless integration with WhatsApp, Telegram, Signal, and a dedicated Web UI (Mesop for v0).
* **Bi-directional Streaming:** Real-time, responsive chat interaction between user and AI.
* **File Upload & Handling:** Secure upload for images (X-rays, skin conditions), PDF lab reports, and potentially audio snippets (e.g., cough sounds).
* **Multilingual Support:** English, Hindi, and major regional Indian languages. UK focus primarily on English, Welsh.
* **Accessibility Features:** Adherence to WCAG guidelines for screen reader compatibility, adjustable text sizes, and clear navigation.
* **Structured Questioning:** Guided symptom collection based on clinical protocols (e.g., NICE guidelines for specific conditions).

## 🔄 API Gateway
* **Unified Entrypoint:** FastAPI-based REST/gRPC endpoint.
* **Authentication & Authorization:** Secure access control, JWT-based for users and API keys for services. Rate limiting to prevent abuse.
* **Version Routing:** Support for API versioning (e.g., `/api/v1/`, `/api/v2/`) for backward compatibility and phased rollouts.
* **WebSocket Proxy:** For real-time chat communication.

## 🤖 AI Orchestration Engine
* **Router Engine:** Intelligently determines which downstream AI/ML services or predefined clinical pathways to invoke based on initial user input and conversation context.
* **Model Selector:** Dynamically chooses the optimal AI model based on task requirements, cost, VRAM availability, and desired accuracy/speed trade-off.
* **Bias Monitor:** Real-time, or near real-time, intersectional bias detection and mitigation layer.
* **Context Manager:** Maintains conversation state, user profile (with consent), and retrieves relevant contextual information (e.g., from RAG).
* **RAG Retriever (Retrieval Augmented Generation):** Fetches relevant snippets from clinical guidelines, medical literature, and emergency protocols.

## 🧠 Specialized ML Services
* **Text NLP Service:** Symptom Extraction, Sentiment Analysis.
* **Image Analysis Service:** Initial Triage, Segmentation, VLM Description.
* **Audio Processing Service:** Speech-to-Text, Emotional Tone Analysis.
* **Time Series Service (Future):** Trend analysis for ECG, vital signs.
* **Risk Classifier Service:** Multi-modal Risk Scoring, X-Y Plot Coordinate for urgency/importance.

## 🚀 Message Queue & Cache
* **Celery Workers:** Asynchronous task distribution.
* **Redis Cache:** Session context, rate limiting, temporary data storage.
* **ChromaDB Vector Store (or similar):** RAG knowledge base, case report embeddings.

## 🗄️ Data Layer
* **Patient Data & Consent:** User Authentication, Profile Management, Consent Mechanisms.
* **EHR Integration (UK & India):** FHIR R4, GP Connect, ABDM standards.
* **PostgreSQL Database:** Structured storage for users, case reports, clinical questions, audit logs.
* **MinIO / S3 Compatible Object Storage:** Secure storage for uploaded files and generated reports.
* **Emergency Protocol Data:** Structured data and embeddings for 111/999 protocols.

## 🌐 Network Layer (Specialist Marketplace - Future v1.x)
* **Specialist Directory:** Registry of healthcare providers.
* **Doctor Availability:** Real-time status, scheduling.
* **Consultation Router:** Intelligent patient routing.

## 📄 PDF Report Generation
* **Standardized Case Reports:** Automated generation including transcript, AI analysis, risk scores, recommendations.
* **Secure Storage & Access:** Stored in MinIO/S3 with secure links.

## 🕵️ Logging & Monitoring
* **Comprehensive Audit Logs:** Anonymized/pseudonymized logs.
* **Metrics Dashboard:** Prometheus/Grafana pipeline for system health and usage.
* **Error Alerts & Exception Tracking:** Sentry integration.

*(Ref: PRD Sections, User Story)*

---
### Related Pages
* [v0 Prototype User Story: Chest Pain Triage](./v0-prototype-user-story.md)

---
[⬅️ Back to Home](./index.md)