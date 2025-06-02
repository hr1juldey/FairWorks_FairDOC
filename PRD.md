Okay, here's a comprehensive Product Requirements Document (PRD) for Fairdoc AI, incorporating the details from your provided documents and the user story.

# 🏥 Fairdoc AI: Product Requirements Document

**Version:** 0.1
**Date:** June 2, 2025
**Authors:** Gemini AI Assistant (based on user input)

## Table of Contents

1.  **🚀 Overview**
2.  **🎯 Vision & Goals**
3.  **👥 Target Users & Audience**
4.  **🧩 Core Features & Modules (v1.0 & Beyond)**
5.  **📖 User Story: v0 Prototype - Chest Pain Triage**
6.  **🔧 Technical Architecture**
      * Overall System Architecture
      * Frontend Layer
      * API Gateway Layer
      * AI Orchestration Layer
      * Specialized ML Services
      * Message Queue & Cache
      * Data Layer
      * Network Layer (Future)
      * PDF Report Generation
      * Logging & Monitoring
7.  **📦 Infrastructure & Deployment**
      * Node Specifications
      * Deployment Models
      * Cost Estimates per Node (Initial)
8.  **👩‍💻 Developer Guide & API Integration (v1.0 Target)**
      * Setup & Prerequisites
      * Core API Endpoints (v0 & v1)
      * SDKs & Libraries
      * Chat Platform Connectors
9.  **🗓️ Roadmap & Timeline**
      * Overall Project Roadmap
      * **📝 TODO.md: v0 Prototype (10-Hour Sprint)**
10. **💰 Financial Model & Sustainability**
      * Funding & Seed Model
      * Service Fees & Premiums
      * Subsidies & Grants
      * Value-Added Services
      * Cost Estimates & Break-even Analysis
11. **⚖️ Legal & Compliance**
      * Data Privacy & Security
      * Medical Device Regulations
      * Open-Source License
      * Clinical Content Licensing
      * Transparency & Ethics
12. **🤝 Contribution Guidelines**
13. **🎉 Acknowledgments**
14. **📚 References**

-----

## 1\. 🚀 Overview

Fairdoc AI is an open-source **Digital Public Infrastructure (DPI)** for healthcare, initially focusing on the **UK and India**. It's designed to provide secure, scalable, and interoperable medical triage and diagnostic support services via familiar messaging interfaces like WhatsApp, Signal, and Telegram, as well as a dedicated web UI.

Inspired by the success of Unified Payments Interface (UPI) and Open Network for Digital Commerce (ONDC) in India, Fairdoc aims to create a healthcare equivalent—the **Unified Health Interface (UHI)**. This platform will:

  * ⚖️ Provide **equitable access** to initial medical triage for all citizens, regardless of their geographical location or socioeconomic status.
  * 🏥 Promote **clinical best practices** by integrating established guidelines like NICE protocols in the UK and Ministry of Health and Family Welfare (MoHFW) / Indian Council of Medical Research (ICMR) standards in India.
  * 🌐 Foster an **ecosystem of innovation** by offering open APIs and SDKs for third-party developers, hospitals, insurance providers, pharmaceutical companies, and diagnostic services.
  * 💾 Operate under **open-source governance** (MIT License), promoting transparency, community collaboration, and rapid iteration.

This document outlines the requirements for Fairdoc AI, with a particular focus on developing a **v0 prototype within a 10-hour sprint** to demonstrate core triage functionality for a patient presenting with chest pain.

*(Ref: `README.md`, Prompt Overview)*

-----

## 2\. 🎯 Vision & Goals

### Vision

To democratize healthcare expertise through an AI-assisted triage and diagnostic support network that connects patients with the right care at the right time, while ensuring fairness, transparency, and robust human oversight. Healthcare should be a human right, not a privilege; technology should bridge gaps, not create them; AI should amplify human compassion, not replace it.

### Goals

  * **Bridge Healthcare Gaps:** Offer a readily accessible, baseline triage service via ubiquitous chat platforms, aiming to reduce preventable emergency visits, alleviate pressure on primary care, and improve outcomes from unnecessary in-person consultations, especially for underserved populations (e.g., addressing longer wait times for ethnic minorities and lack of specialist access for rural patients).
  * **Promote Evidence-Based Care:** Implement and continuously update triage and diagnostic support logic based on NICE Clinical Guidelines (e.g., CG95 for chest pain, CG108), MoHFW India protocols, and other credible medical sources for consistent and reliable decision support.
  * **Enable Health System Innovation:** Provide a robust suite of APIs and SDKs to empower telemedicine startups, hospitals, public health agencies, and other third-party applications to integrate Fairdoc’s triage, diagnostic support, and anonymized data services into their offerings.
  * **Ensure Sustainability & Public Good:** Operate primarily as a non-profit public utility. Funding will be sought through government grants, philanthropic contributions, minimal transaction fees for premium services, and value-added services for institutional partners, modeled after ONDC/UPI funding strategies.
  * **Champion Responsible AI:** Maintain ethical AI principles, including continuous bias monitoring and mitigation (addressing diagnostic bias against ethnicity, gender, age), data privacy, and ensuring human-in-the-loop for all critical medical decisions.

*(Ref: `README.md`, Prompt Vision & Goals)*

-----

## 3\. 👥 Target Users & Audience

  * **Primary Users (Patients):** Individuals in the UK and India seeking initial medical guidance or triage for their symptoms, accessible via common messaging apps.
  * **Secondary Users (Healthcare Professionals):**
      * Doctors and specialists using the platform for preliminary case information, patient routing, or second opinions (with patient consent).
      * Public health officials using anonymized data for epidemiological studies and resource allocation.
  * **Tertiary Users (Developers & Institutions):**
      * Third-party developers building health applications.
      * Hospitals and clinics integrating Fairdoc into their workflows.
      * Insurance companies for streamlining pre-authorizations or wellness programs.
      * Pharmaceutical companies and researchers for anonymized data insights (with ethical oversight).

-----

## 4\. 🧩 Core Features & Modules (v1.0 & Beyond)

This section outlines the comprehensive features. The **v0 Prototype** will implement a subset of these.

### 📱 Chat Frontend & Patient Interaction

  * **Multi-platform Connectors:** Seamless integration with WhatsApp, Telegram, Signal, and a dedicated Web UI (Mesop for v0).
  * **Bi-directional Streaming:** Real-time, responsive chat interaction between user and AI.
  * **File Upload & Handling:** Secure upload for images (X-rays, skin conditions), PDF lab reports, and potentially audio snippets (e.g., cough sounds).
  * **Multilingual Support:** English, Hindi, and major regional Indian languages. UK focus primarily on English, Welsh.
  * **Accessibility Features:** Adherence to WCAG guidelines for screen reader compatibility, adjustable text sizes, and clear navigation.
  * **Structured Questioning:** Guided symptom collection based on clinical protocols (e.g., NICE guidelines for specific conditions).

### 🔄 API Gateway

  * **Unified Entrypoint:** FastAPI-based REST/gRPC endpoint.
  * **Authentication & Authorization:** Secure access control, JWT-based for users and API keys for services. Rate limiting to prevent abuse.
  * **Version Routing:** Support for API versioning (e.g., `/api/v1/`, `/api/v2/`) for backward compatibility and phased rollouts.
  * **WebSocket Proxy:** For real-time chat communication.

### 🤖 AI Orchestration Engine

  * **Router Engine:** Intelligently determines which downstream AI/ML services or predefined clinical pathways to invoke based on initial user input and conversation context (e.g., symptom classifier, risk scorer, image analyzer).
  * **Model Selector:** Dynamically chooses the optimal AI model (e.g., local LLMs via Ollama like Gemma 4B, or specialized smaller models) based on task requirements, cost, VRAM availability, and desired accuracy/speed trade-off.
  * **Bias Monitor:** Real-time, or near real-time, intersectional bias detection and mitigation layer. Monitors for disparities in assessment outcomes based on protected characteristics (ethnicity, gender, age, socioeconomic indicators) and flags/adjusts as needed.
  * **Context Manager:** Maintains conversation state, user profile (with consent), and retrieves relevant contextual information (e.g., from RAG) to personalize and improve the accuracy of AI responses.
  * **RAG Retriever (Retrieval Augmented Generation):** Fetches relevant snippets from clinical guidelines (NICE, MoHFW), medical literature, and emergency protocols (e.g., `111.json`, `999.json` equivalents) to ground AI responses in evidence.

### 🧠 Specialized ML Services

  * **Text NLP Service:**
      * Symptom Extraction: Using models like DistilBERT, ClinicalBERT, or fine-tuned smaller LLMs to identify and normalize medical symptoms, entities, and negation from user text.
      * Sentiment Analysis: To gauge patient distress or urgency.
  * **Image Analysis Service:**
      * Initial Triage: Using models like EfficientNet-B3, MobileNet, or custom CNNs for basic analysis of X-rays, ECG images, dermatology photos.
      * Segmentation: SAM (Segment Anything Model) or SAM2.1 for identifying regions of interest in medical images.
      * VLM Description: Multi-modal models (e.g., Gemma 4B with vision capabilities) to describe labeled/segmented images.
  * **Audio Processing Service:**
      * Speech-to-Text: Whisper-Small/Base for transcribing user voice inputs or audio recordings (e.g., cough sounds).
      * Emotional Tone Analysis: From audio input.
  * **Time Series Service (Future):** LSTM/Transformer models for analyzing trends in ECG, vital signs if continuous data becomes available.
  * **Risk Classifier Service:**
      * Multi-modal Risk Scoring: XGBoost, LightGBM, or neural networks combining text, image, and structured data to produce a quantifiable risk score or urgency level.
      * X-Y Plot Coordinate: A specific output mapping importance (Y-axis: +1 important to -1 not important) and urgency (X-axis: +1 urgent to -1 not urgent) for critical cases.

### 🚀 Message Queue & Cache

  * **Celery Workers:** Asynchronous task distribution for computationally intensive operations like AI model inference, bias checks, EHR interactions, and PDF report generation.
  * **Redis Cache:**
      * Session Context: Storing active conversation states, user session data.
      * Rate Limiting: Tracking API request frequencies.
      * Temporary Data Storage: Short-lived data, RAG embeddings for quick retrieval.
  * **ChromaDB Vector Store (or similar):**
      * RAG Knowledge Base: Storing vector embeddings of clinical guidelines, FAQs, and emergency protocols for fast semantic search.
      * Case Report Embeddings: Storing embeddings of case report fields for matching against emergency protocols and EHR data.

### 🗄️ Data Layer

  * **Patient Data & Consent:**
      * User Authentication & Profile Management.
      * Granular consent mechanisms for data usage, EHR access, and specialist referral.
  * **EHR Integration (UK & India):**
      * UK: FHIR R4 client for GP Connect, NHS Digital APIs for patient record fetch/update (with explicit consent).
      * India: Adherence to ABDM (Ayushman Bharat Digital Mission) standards for UHI connectivity and health record exchange.
  * **PostgreSQL Database:**
      * Structured storage for user accounts, patient demographics (anonymized/pseudonymized where appropriate), Pydantic-defined case reports, clinical questions, AI assessment results, audit logs, and mappings for emergency protocols (e.g., `111.json`, `999.json` content).
  * **MinIO / S3 Compatible Object Storage:** Secure storage for user-uploaded files (images, PDFs) and system-generated PDF case reports. Each file will have a unique reference URL stored in the PostgreSQL database.
  * **Emergency Protocol Data:** Flattened tables or structured data from `111.json` (non-emergency advice) and `999.json` (emergency conditions) or their equivalents, with vector embeddings in ChromaDB for matching.

### 🌐 Network Layer (Specialist Marketplace - Future v1.x)

  * **Specialist Directory:** Registry of verified healthcare specialists, clinics, and telemedicine providers.
  * **Doctor Availability:** Real-time status updates, queue management, and appointment scheduling capabilities.
  * **Consultation Router:** Intelligent routing of high-risk patients or those requiring specialist consultation to available and appropriate providers (e.g., teleconsultation, ambulance dispatch based on urgency score and local emergency numbers).

### 📄 PDF Report Generation

  * **Standardized Case Reports:** Automated generation of PDF reports using libraries like ReportLab. Reports to include:
      * Patient-provided information and chat transcript.
      * AI-derived symptom analysis.
      * Uploaded image thumbnails (if any) and their VLM descriptions.
      * Risk scores, urgency assessment (including X-Y plot if applicable).
      * Recommended next steps based on guidelines.
      * Justification for assessment.
      * General, safe advice (LLM-generated and reviewed).
  * **Secure Storage & Access:** Reports stored in MinIO/S3, with secure links provided to patients and authorized healthcare professionals.

### 🕵️ Logging & Monitoring

  * **Comprehensive Audit Logs:** Anonymized/pseudonymized logging of all significant interactions, AI decisions, rationale, bias incidents, and data access.
  * **Metrics Dashboard:** Prometheus/Grafana pipeline for system health monitoring (CPU, GPU, memory usage), API performance, model inference times, usage patterns.
  * **Error Alerts & Exception Tracking:** Integration with Sentry or similar for real-time error reporting and debugging.

*(Ref: Prompt Core Features, User Story, `README.md`)*

-----

## 5\. 📖 User Story: v0 Prototype - Chest Pain Triage (10-Hour Sprint Focus)

**User:** A patient (e.g., "Alex," 45 years old) experiencing new-onset left chest pain, accompanied by coughing and wheezing. Alex is concerned and decides to use the Fairdoc service via a web interface (Mesop prototype).

**Pre-conditions:**

  * The Fairdoc v0 backend (FastAPI) is running.
  * The Mesop frontend is accessible.
  * Basic infrastructure (PostgreSQL, MinIO, Redis, Ollama with Gemma 4B and mxbai-embed-large) is set up and minimally configured.
  * Pre-programmed questions for chest pain (based on simplified NHS/NICE guidance) are available.
  * Mock `111.json` (non-emergency) and `999.json` (emergency) vector embeddings are pre-loaded in ChromaDB (or a simplified matching mechanism is in place).

**User Journey & Engineering Steps:**

1.  **Initiates Chat (Frontend: Mesop | Backend: FastAPI)**

      * **Alex:** Opens the Fairdoc Mesop web app and starts a new chat session.
      * **System (Mesop UI):** Displays a welcome message and a prompt to describe their main symptom.
      * **Alex:** Types "I have left chest pain with coughing and wheezing."
      * **Engineering (Frontend):** Mesop sends this initial message to the FastAPI backend (`/api/v0/chat/initiate`).

2.  **Initial Symptom Processing & Case Report Creation (Backend: FastAPI, Gemma 4B, Pydantic, PostgreSQL)**

      * **Engineering (Backend):**
          * The FastAPI endpoint receives the message.
          * A **CaseReport Pydantic model instance** is created with a unique `case_id`.
          * The initial message ("I have left chest pain with coughing and wheezing") is **silently processed by Gemma 4B (via Ollama)** to extract initial keywords/symptoms (e.g., "chest pain," "left side," "coughing," "wheezing") and populate relevant fields in the `CaseReport` Pydantic model.
          * The initial `CaseReport` (with `case_id` and extracted symptoms) is **saved to the PostgreSQL `case_reports` table**. Status: "PENDING\_QUESTIONS".

3.  **Guided Questioning (Frontend: Mesop | Backend: FastAPI)**

      * **Engineering (Backend):** Based on keywords like "chest pain," the backend retrieves a **pre-programmed set of questions** (simplified NICE guidance for chest pain first contact).
          * *Search Result for NHS NICE "chest pain first contact questions":* Typical questions involve:
              * **P**rovocation/Palliation: What makes it worse or better?
              * **Q**uality: Describe the pain (sharp, dull, pressure).
              * **R**adiation: Does it spread anywhere? (e.g., arm, jaw, back).
              * **S**everity: On a scale of 1-10, how bad is it?
              * **T**iming: When did it start? Is it constant or intermittent? How long does it last?
              * Associated symptoms: Shortness of breath, nausea, sweating, dizziness, palpitations, fever.
              * For cough/wheeze: Duration, sputum, triggers.
              * Relevant history: Previous heart problems, lung conditions, risk factors (smoking, diabetes, hypertension).
      * **System (Mesop UI):** Displays the first question, e.g., "Can you describe the pain? For example, is it sharp, dull, like pressure, or a burning feeling?"
      * **Alex:** Responds to each question sequentially.
      * **Engineering (Frontend & Backend):**
          * Mesop sends each answer to a FastAPI endpoint (e.g., `/api/v0/chat/continue`).
          * The backend appends the Q\&A to the chat transcript.
          * Gemma 4B continues to **silently listen and update the structured fields** in the `CaseReport` Pydantic model based on Alex's answers.
          * The `CaseReport` in PostgreSQL is updated after each significant answer or batch of answers.

4.  **Optional File Upload (Frontend: Mesop | Backend: FastAPI, MinIO, Celery)**

      * **System (Mesop UI):** After core questions, "Have you had any recent medical reports or a chest X-ray you'd like to share? (Optional)" with an upload button.
      * **Alex:** Chooses to upload a PDF of a recent (mock) ECG report and a JPG of a (mock) chest X-ray.
      * **Engineering (Frontend):** Mesop handles file selection and uploads it to a FastAPI endpoint (e.g., `/api/v0/upload_file`).
      * **Engineering (Backend - FastAPI):**
          * Receives the file.
          * Triggers a **Celery background task** (`upload_to_minio_task`).
      * **Engineering (Celery Worker - `upload_to_minio_task`):**
          * Uploads the file to a **MinIO bucket** (e.g., `fairdoc-casefiles`).
          * Gets the **MinIO URL/object key** for the file.
          * Updates the `CaseReport` in PostgreSQL: adds an entry to an `attachments` field (e.g., a JSON array like `[{"type": "pdf_report", "url": "minio_url_1", "status": "uploaded"}, {"type": "xray_image", "url": "minio_url_2", "status": "uploaded"}]`).
      * **System (Mesop UI):** Shows "File uploaded successfully. We will process it."

5.  **Case Finalization & Initial Notification (Backend: FastAPI, Celery)**

      * **System (Mesop UI):** "Thank you, Alex. We have your initial information. We are now processing your case. You will receive an update shortly."
      * **Engineering (Backend):** Once the standard questions are done and optional files are noted as uploaded, the `CaseReport` status in PostgreSQL is updated to "PROCESSING".
      * A Celery task (`patient_notification_task`) is triggered to (conceptually for v0) send Alex this notification.

6.  **Background Processing - Phase 1: Embedding & Basic Matching (Backend: Celery, mxbai-embed-large, Redis, ChromaDB/PostgreSQL)**

      * **Celery Worker (`process_case_report_task` - Step 1):**
          * Retrieves the finalized `CaseReport` from PostgreSQL.
          * Selected textual fields (chief complaint, key symptoms summary from Gemma 4B) are **vector embedded using `mxbai-embed-large` (via Ollama)**.
          * These embeddings are temporarily **cached in Redis** for quick access during this processing phase.
          * The case embeddings are **matched against pre-loaded vector embeddings** of `111.json` (non-emergency conditions/advice) and `999.json` (emergency conditions) stored in **ChromaDB** (which could use PostgreSQL as a backend or have its vectors mapped to PostgreSQL tables holding the actual text/rules).
          * Matching rule IDs or relevant snippets from `111.json`/`999.json` are added to the `CaseReport` in PostgreSQL (e.g., `matched_protocols: ["chest_pain_protocol_7", "breathing_difficulty_protocol_3"]`).
          * (Simplified for v0): EHR matching is skipped or mocked.

7.  **Background Processing - Phase 2: Multi-modal Analysis (Backend: Celery, OpenCV, SAM2.1 (mocked), Gemma 4B VLM)**

      * **Celery Worker (`process_case_report_task` - Step 2 - for each uploaded image like X-ray):**
          * Retrieves the image file from MinIO using the stored URL.
          * **(Mocked/Simplified for v0):**
              * OpenCV: Basic processing (e.g., load, check dimensions).
              * SAM2.1: Segmentation is conceptual/mocked. Assume a pre-labeled version or skip complex segmentation.
              * Special X-ray model: Not available for v0.
          * **Gemma 4B (VLM capabilities):** The (potentially mock-labeled) image is fed to Gemma 4B to generate a textual description (e.g., "Chest X-ray shows possible opacity in left lower lobe."). This description is added to the `CaseReport` (e.g., `image_analysis: [{"url": "minio_url_2", "description": "...", "status": "analyzed"}]`).
          * The analyzed image (or its labeled version) could be stored back in MinIO with a new link.
      * **(Simplified for v0):** PDF report text extraction (OCR if needed) and analysis is deferred or heavily simplified. If Alex uploaded a PDF, its textual content might be coarsely summarized by Gemma 4B.

8.  **Background Processing - Phase 3: Reasoning & Urgency Scoring (Backend: Celery, DeepSeek/Mistral-like model)**

      * **Celery Worker (`process_case_report_task` - Step 3):**
          * The comprehensive `CaseReport` (now including text Q\&A summaries, 111/999 matches, image descriptions) is fed to a **stronger reasoning LLM (e.g., DeepSeek R1 14B, Qwen4, or Mistral 13B - simulated via a smaller capable model like Gemma 7B if necessary for speed, or even a well-crafted prompt to Gemma 4B for v0).**
          * The LLM is asked **one critical question**: "Based on this case report, in a normalized X-Y plot where Y-axis is Important/Not-Important (+1 to -1) and X-axis is Urgent/Not-Urgent (+1 to -1), provide a plot coordinate (e.g., {'x': 0.8, 'y': 0.9})."
          * The returned score (e.g., `{'x': 0.8, 'y': 0.9}`) is saved to the `CaseReport` as `urgency_coordinates`. This score implicitly maps to the potential for harm if not addressed swiftly.

9.  **Background Processing - Phase 4: PDF Report Generation (Backend: Celery, ReportLab)**

      * **Celery Worker (`generate_pdf_report_task`):**
          * Retrieves the fully processed `CaseReport`.
          * Uses **ReportLab** to generate a PDF document containing:
              * Patient (Alex's) details (if provided, else anonymous).
              * Summary of chat Q\&A.
              * Key findings from Gemma 4B's initial processing.
              * VLM description of the X-ray.
              * The X-Y urgency/importance score.
              * A brief justification for the assessment (potentially drafted by the reasoning LLM).
          * The generated PDF is **uploaded to MinIO**, and its URL is added to the `CaseReport`. `status` becomes "REPORT\_GENERATED".

10. **Decision & Notification/Escalation (Backend: FastAPI/Celery)**

      * **Engineering (Backend - `final_disposition_task`):**
          * Based on the `urgency_coordinates` (e.g., if X \> 0.7 and Y \> 0.7), the system determines if the case is potentially fatal/highly urgent.
          * **If highly urgent:**
              * The PDF report is **flagged for immediate review by a (mock) doctor** from a `doctors_table` in PostgreSQL (for v0, this is a placeholder).
              * An alert is conceptually sent to this doctor.
              * Alex receives a notification: "Your case has been assessed as requiring urgent attention. We are forwarding your report to a healthcare professional. If you feel your condition is worsening rapidly, please contact emergency services at 999 (UK) / 108 (India)."
          * **If not immediately life-threatening (for v0, assume this path for simplicity of demo):**
              * The PDF report link is sent to Alex via the Mesop interface.
              * A message with **general, safe advice (drafted by the reasoning LLM** during step 8, e.g., "Based on your symptoms, it's important to monitor your condition. Rest, avoid strenuous activity. If pain worsens or you develop new severe symptoms like difficulty breathing, seek medical attention promptly.") is also sent.
      * **System (Mesop UI):** Alex sees the final message, the advice, and a link to download their PDF case report.
      * The `CaseReport` status in PostgreSQL is updated to "COMPLETED\_PATIENT\_NOTIFIED" or "COMPLETED\_ESCALATED".

**Pydantic Models (Illustrative for v0):**

```python
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import datetime

class Attachment(BaseModel):
    attachment_id: str # uuid
    case_id: str
    file_type: str # 'pdf_report', 'xray_image', 'ecg_image', etc.
    original_filename: Optional[str] = None
    minio_url: HttpUrl
    upload_timestamp: datetime.datetime
    analysis_status: str = "uploaded" # 'uploaded', 'processing', 'analyzed', 'error'
    analysis_description: Optional[str] = None # VLM output for images, OCR summary for PDFs

class ChatInteraction(BaseModel):
    timestamp: datetime.datetime
    source: str # 'patient', 'system', 'ai_triage'
    text: str
    intermediate_llm_summary: Optional[str] = None # Gemma 4B's summary

class CaseReportV0(BaseModel):
    case_id: str # uuid, primary key
    patient_id: Optional[str] = None # Link to a user table if exists
    creation_timestamp: datetime.datetime
    last_updated_timestamp: datetime.datetime
    status: str # 'PENDING_QUESTIONS', 'PROCESSING', 'AWAITING_UPLOAD_ANALYSIS', 'REPORT_GENERATED', 'COMPLETED_PATIENT_NOTIFIED', 'COMPLETED_ESCALATED', 'ERROR'
    
    initial_complaint: Optional[str] = None
    symptoms_extracted_by_llm: Optional[Dict[str, Any]] = None # from Gemma 4B silent listening
    chat_history: List[ChatInteraction] = []
    
    attachments: List[Attachment] = [] # List of MinIO URLs and metadata for uploaded files
    
    vector_embeddings_cached_redis: bool = False
    matched_emergency_protocols: Optional[List[str]] = None # IDs from 111/999.json matches
    
    # Results from reasoning LLM
    urgency_coordinates: Optional[Dict[str, float]] = None # {'x': 0.0, 'y': 0.0}
    reasoning_llm_justification: Optional[str] = None
    general_advice_from_llm: Optional[str] = None
    
    generated_pdf_report_url: Optional[HttpUrl] = None

class CaseReportDB(CaseReportV0): # For SQLAlchemy, if needed
    pass

class UserMessage(BaseModel):
    session_id: Optional[str] = None # Or link to case_id
    text: str
    attachments: Optional[List[Dict[str, str]]] = None # For new uploads within chat flow

class AIResponseMessage(BaseModel):
    session_id: str
    response_text: str
    next_questions_ids: Optional[List[str]] = None # To guide frontend
    request_file_upload: bool = False
    report_link: Optional[HttpUrl] = None
```

This user story provides a clear path for the 10-hour v0 prototype.

-----

## 6\. 🔧 Technical Architecture

*(Based on, `README.md`, `TODO.md` and user story. MermaidJS diagrams below)*

### Overall System Architecture (Conceptual v1.0)

```mermaid

---
config:
  theme: neo-dark
  layout: elk
---
flowchart LR
 subgraph subGraph0["💬 User Interaction Layer"]
        WA["📱 WhatsApp"]
        TG["🤖 Telegram"]
        SL["💬 Signal"]
        WebUI["🌐 Web UI<br>(Mesop v0)"]
  end
 subgraph subGraph1["🚪 Gateway & Orchestration"]
        GW["⚙️ FastAPI API Gateway<br>(Auth, Routing, WebSocket)"]
        AO["🤖 AI Orchestration Engine"]
        MS["🧠 Model Selector<br>(Ollama)"]
        BM["⚖️ Bias Monitor"]
        CM["📦 Context Manager<br>(RAG)"]
  end
 subgraph subGraph2["🔁 Core Services Layer"]
        VS["📚 ChromaDB Vector Store<br>(Clinical Guidelines, Case Embeds)"]
        MQ["🐇 Celery Task Queue"]
        Redis["🗃️ Redis Cache<br>(Session, Temp Data)"]
  end
 subgraph subGraph3["🧠 Specialized AI/ML Workers (via Celery)"]
        W_NLP["📜 Text NLP<br>(Gemma 4B, mxbai-embed)"]
        W_IMG["🖼️ Image Analysis<br>(OpenCV, SAM2.1, VLM)"]
        W_AUD["🔊 Audio Processing<br>(Future)"]
        W_RSK["⚕️ Risk Classifier<br>(DeepSeek/Mistral)"]
        W_PDF["📄 PDF Generator<br>(ReportLab)"]
  end
 subgraph subGraph4["💾 Data & Integration Layer"]
        DB["🗄️ PostgreSQL DB<br>Users, Cases, Protocols"]
        FS["📦 MinIO/S3 Storage<br>(Files, PDFs)"]
        EHR["🏥 NHS/ABDM EHR<br>(Future)"]
  end
 subgraph subGraph5["🧩 Monitoring & Logging"]
        LOG["🕵️ Central Logs<br>(ELK / Prometheus / Sentry)"]
  end
    WA --> GW
    TG --> GW
    SL --> GW
    WebUI --> GW
    GW --> AO & FS & LOG
    AO --> MS & BM & CM & MQ & LOG
    CM --> VS
    MQ --> Redis & LOG
    W_NLP --> MQ & DB
    W_IMG --> MQ & DB & FS
    W_AUD --> MQ
    W_RSK --> MQ & DB
    W_PDF --> MQ & DB & FS
    DB <--> VS
    EHR <--> AO
    DB --> LOG
     WA:::user
     WA:::user
     TG:::user
     TG:::user
     SL:::user
     SL:::user
     WebUI:::user
     WebUI:::user
     GW:::gateway
     GW:::gateway
     AO:::gateway
     AO:::gateway
     MS:::gateway
     MS:::gateway
     BM:::gateway
     BM:::gateway
     CM:::gateway
     CM:::gateway
     VS:::core
     VS:::core
     MQ:::core
     MQ:::core
     Redis:::core
     Redis:::core
     W_NLP:::aiml
     W_NLP:::aiml
     W_IMG:::aiml
     W_IMG:::aiml
     W_AUD:::aiml
     W_AUD:::aiml
     W_RSK:::aiml
     W_RSK:::aiml
     W_PDF:::aiml
     W_PDF:::aiml
     DB:::data
     DB:::data
     FS:::data
     FS:::data
     EHR:::data
     EHR:::data
     LOG:::monitor
     LOG:::monitor
    classDef user fill:#E0F7FA,stroke:#00796B,stroke-width:2px,color:#004D40
    classDef gateway fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef core fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92
    classDef aiml fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    classDef data fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef monitor fill:#ECEFF1,stroke:#37474F,stroke-width:2px,color:#263238
```

* **v0 Focus:** Basic FastAPI endpoints to handle chat, file uploads, and trigger backend processing for the user story. No complex auth for the 10hr sprint.

### AI Orchestration Layer

----

```mermaid
---
config:
  theme: neo-dark
  layout: elk
---
flowchart TB
 subgraph subGraph0["🤖 AI Orchestration - v0"]
        M["🔍 RAG Retriever<br>(ChromaDB: 111/999 only)"]
        L["📦 Context Manager<br>(Session / Case ID)"]
        K["⚖️ Bias Monitor<br>(Conceptual Only)"]
        J["🧠 Model Invoker<br>(Ollama Call)"]
        I["🚦 Request Handler<br>(FastAPI)"]
  end
    I --> J
    J --> K
    K --> L
    L --> M
     I:::ai
     J:::ai
     K:::ai
     L:::ai
     M:::ai
    classDef ai fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92


```

* **v0 Focus:** Direct calls to Ollama for Gemma 4B (text & VLM), mxbai-embed-large. Reasoning LLM calls also direct or simplified. Basic RAG via ChromaDB for emergency protocol matching. Bias monitor is conceptual.

### Specialized ML Services (Invoked via Celery for v0)

```mermaid
---
config:
  theme: neo-dark
  layout: elk
---
flowchart LR
 subgraph subGraph0["🧠 Specialized ML Services - v0 (via Celery)"]
        N["📜 Text NLP<br>(Gemma 4B - Summary)"]
        O["🖼️ Image Analysis<br>(OpenCV + Gemma 4B VLM)"]
        P["🔊 Audio Processing<br>(🚫 Not Available in v0)"]
        Q["📈 Time Series<br>(🚫 Not Available in v0)"]
        R["⚕️ Risk Classifier<br>(Reasoning LLM - X→Y Score)"]
        DUMMY[" "]
  end
    DUMMY --> N & O & P & Q & R
     DUMMY:::spacer
     N:::ml
     O:::ml
     P:::ml
     Q:::ml
     R:::ml
    classDef ml fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    classDef spacer fill:#FFFFFF,stroke:#FFFFFF

```

* **v0 Focus:**
  * Text NLP: Gemma 4B via Ollama for summarizing chat and extracting keywords. Mxbai-embed-large for embeddings.
  * Image Analysis: Basic OpenCV, Gemma 4B (VLM) for description. SAM2.1 is mocked.
  * Risk Classifier: Reasoning LLM (simulated if needed) for the X-Y plot coordinate.

### Message Queue & Cache

-----

```mermaid
---
config:
  layout: elk
  theme: neo-dark
---
flowchart TB
 subgraph subGraph0["📬 Message Queue & Cache - v0 Core (Async)"]
        U["📚 ChromaDB Vector Store<br>(111/999 Embeddings)"]
        T["🗃️ Redis Cache<br>(Sessions, Temp Embeddings)"]
        S["🐇 Celery Workers"]
  end
    S --> T
    T --> U
     S:::queue
     T:::queue
     U:::queue
    classDef queue fill:#F1F8E9,stroke:#689F38,stroke-width:2px,color:#33691E

```

* **v0 Focus:** Celery for background tasks (file processing, AI analysis, PDF generation). Redis for caching embeddings briefly. ChromaDB for pre-loaded emergency protocol vectors.

### Data Layer

-----

```mermaid
flowchart TB
  V["🏥 EHR Integration<br/>(🚫 Not Available in v0)"]:::data
  W["📑 FHIR R4 API<br/>(🚫 Not Available in v0)"]:::data
  X["🗄️ PostgreSQL DB<br/>(Users, Case Reports, Attachments)"]:::data
  Y["📦 MinIO Object Storage<br/>(Uploaded Files, PDF Reports)"]:::data

  subgraph "💾 Data Layer - Core Components (v0)"
    V --> W
    W --> X
    X --> Y
  end

  classDef data fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;

```

* **v0 Focus:** PostgreSQL for `case_reports` (using Pydantic models like `CaseReportV0`) and `attachments`. MinIO for storing uploaded files and the final PDF report.

### Network Layer (Specialist Marketplace)

  * Not in scope for v0. Future enhancement.

### PDF Report Generation

  * **v0 Focus:** A Celery task using ReportLab to create a basic PDF with key information from the `CaseReportV0` model.

### Logging & Monitoring

  * **v0 Focus:** Basic print statements and FastAPI/Celery default logging. Full Prometheus/Grafana/Sentry is out of scope for the 10-hour sprint.

-----

## 7\. 📦 Infrastructure & Deployment

*(Ref: Prompt, `Fairdoc_AI_v0 Prototype (10-Hour Sprint).md`)*

### Node Specifications (Target for Scaled Deployment)

  * **Minimum Production Node:**
      * 🖥️ CPU: 8 vCPUs (x86\_64)
      * 🧠 RAM: 32 GB DDR4
      * 💾 Storage: 1 TB NVMe SSD
      * 🎮 GPU: CUDA-enabled GPU ≥12 GB VRAM (e.g., NVIDIA RTX 3060/3080/4070, T4)
      * 🌐 Network: 1 Gbps Ethernet
  * **Local HPC for v0 Prototype (as per user context):**
      * CPU: AMD Ryzen 7 5700X (8c/16t)
      * RAM: 32 GB DDR4
      * GPU: NVIDIA RTX 3060 12 GB
      * Python: 3.12.10 venv

### Deployment Models

  * **Cloud (Target):** Kubernetes cluster (AWS/GCP/Azure) with GPU nodes (e.g., g4dn.2xlarge), control plane nodes (e.g., t3.medium), auto-scaling, managed Redis & PostgreSQL.
  * **On-Prem/Edge (Future):** Local servers in clinics.
  * **Hybrid (Future):** Core inference in cloud, lightweight tasks on edge.
  * **v0 Prototype:** Local deployment on the HPC using Docker Compose for services (PostgreSQL, Redis, MinIO, Ollama) and direct run for FastAPI/Celery/Mesop. The `Fairdoc_AI_v0 Prototype (10-Hour Sprint).md` mentions progress with Docker Compose for PostgreSQL, Redis, MinIO, ChromaDB. This setup should be leveraged/completed.

### Cost Estimates per Node (Illustrative for Cloud)

Certainly! Here's an updated and detailed infrastructure cost breakdown for Fairdoc, incorporating current AWS pricing and additional components:

---

## 💰 Fairdoc Infrastructure Cost Breakdown (Monthly Estimates)

| **Component**            | **Specification**                    | **Estimated Monthly Cost (USD)** | **Notes**                                                                                    |
| ------------------------ | ------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------- |
| **GPU Node**             | 8 vCPU, 32 GB RAM, NVIDIA T4 (16 GB) | \$378 – \$530                    | AWS g4dn.xlarge: \$0.526/hr (~~\$378/month); SageMaker: \$0.7364/hr (~~\$530/month)          |
| **CPU Node**             | 4 vCPU, 16 GB RAM                    | \$100 – \$150                    | Suitable for application server and Celery tasks                                             |
| **Redis Instance**       | 3 GB RAM, 2 vCPU                     | \$50 – \$100                     | AWS ElastiCache: \$0.032/hr for cache.t4g.small (\~\$23/month); higher for larger instances  |
| **PostgreSQL Instance**  | db.r5.large (2 vCPU, 16 GB RAM)      | \$200                            | AWS RDS pricing for db.r5.large                                                              |
| **ChromaDB**             | 2 vCPU, 8 GB RAM                     | \$100                            | Self-hosted or managed vector database                                                       |
| **MinIO/S3 Storage**     | Variable (storage + egress)          | \$50+                            | Based on usage; AWS S3 pricing applies                                                       |
| **Monitoring & Logging** | Prometheus, ELK Stack                | \$50 – \$100                     | Costs for data ingestion and storage                                                         |
| **Bandwidth**            | Variable                             | \$100+                           | Depends on data transfer volumes                                                             |
| **Total Estimated Cost** |                                      | **\$1,028 – \$1,330+**           | Excludes potential savings from reserved instances or spot pricing                           |

-----

**Additional Considerations:**

* **Cost Optimization:** Utilizing spot instances or reserved instances can lead to significant savings, especially for GPU nodes.
* **Scalability:** Costs may vary based on usage patterns, data transfer, and storage requirements.
* **Compliance and Security:** Ensure that all components meet necessary compliance standards (e.g., HIPAA, GDPR) and implement robust security measures.


-----

## 8\. 👩‍💻 Developer Guide & API Integration (v1.0 Target)

*(v0 will focus on achieving the user story functionality; full SDKs are for later)*

### Setup & Prerequisites (for v0 on Local HPC)

1.  **Clone Repo (Assume one exists/is created):**
    ```bash
    git clone <repository_url> fairdoc_v0
    cd fairdoc_v0
    ```
2.  **Python Environment:**
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    Key `requirements.txt` entries for v0: `fastapi`, `uvicorn`, `sqlalchemy` (for Pydantic-to-DB mapping if not using raw SQL), `psycopg2-binary`, `celery`, `redis`, `minio`, `ollama`, `reportlab`, `mesop`, `google-search-results` (for SerpApi if used for dynamic info, or just `requests`), `torch`, `transformers` (for mxbai if not solely relying on Ollama's embedding endpoint), `opencv-python`, `pydantic>=2.0`.
3.  **Infrastructure Services (Docker Compose - from `Fairdoc_AI_v0 Prototype (10-Hour Sprint).md`):**
      * Ensure `docker-compose.yml` is configured for:
          * PostgreSQL (e.g., `postgres:15`)
          * Redis (e.g., `redis:7`)
          * MinIO (e.g., `minio/minio`)
          * Ollama with Gemma 4B, Gemma 7B (or other reasoning model), mxbai-embed-large pulled.
          * ChromaDB (e.g., `chromadb/chroma`)
      * Start services: `docker-compose up -d`
      * Verify connections (as per `test_connections.py` mentioned in sprint doc).
4.  **Environment Variables (`.env` file):**
    ```env
    DATABASE_URL="postgresql://user:password@localhost:5432/fairdocdb_v0"
    REDIS_URL="redis://localhost:6379/0"
    MINIO_ENDPOINT="localhost:9000"
    MINIO_ACCESS_KEY="minioadmin"
    MINIO_SECRET_KEY="minioadmin"
    MINIO_BUCKET_NAME="fairdoc-v0-cases"
    OLLAMA_BASE_URL="http://localhost:11434" # If Ollama runs in Docker, ensure FastAPI can reach it.
    # Add other necessary configs
    ```
5.  **Run Backend (FastAPI):**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 
    ```
6.  **Run Celery Workers:**
    ```bash
    celery -A app.celery_worker.celery worker -l info 
    ```
7.  **Run Frontend (Mesop):**
    ```bash
    mesop app.mesop_app:app
    ```
    Access at `http://localhost:3210` (Mesop default) or as configured.

### Core API Endpoints (Focus for v0)

*(All prefixed with `/api/v0`)*

1.  **`POST /chat/initiate`**:
      * Description: Starts a new chat session for the patient, creates an initial case report.
      * Request Body: `{"initial_message": "I have chest pain..."}`
      * Response: `{"case_id": "uuid", "first_question": "Can you describe the pain?"}`
2.  **`POST /chat/continue`**:
      * Description: Sends subsequent patient responses, gets next question or status.
      * Request Body: `{"case_id": "uuid", "answer": "It's a dull ache."}`
      * Response: `{"case_id": "uuid", "next_question": "...", "status_update": "..."}` or `{"case_id": "uuid", "final_message": "...", "report_link": "..."}`
3.  **`POST /upload_file`**: (Potentially integrated into `/chat/continue` if Mesop handles upload and sends data/reference)
      * Description: Uploads a file (X-ray, PDF) related to a case.
      * Request: `multipart/form-data` with `case_id` and `file`.
      * Response: `{"file_id": "uuid", "minio_url": "...", "message": "File received, processing started."}`
      * *Alternatively, Mesop might upload directly to a pre-signed MinIO URL obtained from backend, then notify backend.*
4.  **`GET /case/{case_id}/status`**:
      * Description: Frontend polls this to get updates on background processing.
      * Response: `{"case_id": "uuid", "status": "PROCESSING_IMAGE_ANALYSIS", "progress_percent": 60}`
5.  **`GET /case/{case_id}/report`**:
      * Description: Retrieves the final PDF report link once ready.
      * Response: `{"case_id": "uuid", "report_url": "minio_url_to_pdf", "advice": "..."}`

*(These are simplified for the 10hr sprint. Full v1 endpoints from prompt for future reference.)*

### SDKs & Libraries (Future v1.0)

  * Python SDK: `pip install fairdoc-sdk`
  * JavaScript SDK: `npm install fairdoc-sdk`

### Chat Platform Connectors (Future v1.0)

  * WhatsApp (Twilio), Telegram Bot, Signal-CLI integration as outlined in prompt.

-----

## 9\. 🗓️ Roadmap & Timeline

### Overall Project Roadmap (from Prompt)

```mermaid
---
config:
  theme: neo-dark
  themeVariables:
    primaryColor: '#6200EE'
    primaryTextColor: '#FFFFFF'
    secondaryColor: '#03DAC6'
    secondaryTextColor: '#000000'
    tertiaryColor: '#018786'
    tertiaryTextColor: '#FFFFFF'
    background: '#121212'
    fontFamily: ''
    fontSize: 14px
    nodeBorder: '#BBBBBB'
    clusterBkg: '#1E1E1E'
    clusterBorder: '#BBBBBB'
  layout: elk
---
flowchart TB
 subgraph subGraph0["June 2025"]
        J1["2025-06-03: Project kickoff & team alignment"]
        J2["2025-06-10: Set up dev env & CI/CD pipelines"]
        J3["2025-06-17: Define core architecture & tech stack"]
        J4["2025-06-24: Begin implementing backend microservices"]
  end
 subgraph subGraph1["July 2025"]
        J5["2025-07-01: Integrate Ollama interface & test inference"]
        J6["2025-07-08: Develop chat UI skeleton (Mesop) & basic routing"]
        J7["2025-07-15: Implement NICE Protocol Engine prototype"]
        J8["2025-07-22: Set up ChromaDB & basic RAG retrieval"]
        J9["2025-07-29: Build file upload handler & MinIO/S3 integration"]
  end
 subgraph subGraph2["August 2025"]
        A1["2025-08-05: Complete chat UI with messaging & file upload"]
        A2["2025-08-12: Integrate session management & context propagation"]
        A3["2025-08-19: Wire protocol engine to chat UI (end-to-end flow)"]
        A4["2025-08-26: Develop PDF report generator (ReportLab)"]
  end
 subgraph subGraph3["September 2025"]
        S1["2025-09-02: Implement Celery task queue for background processing"]
        S2["2025-09-09: Integrate Redis cache for session & temp data"]
        S3["2025-09-16: Link Celery tasks to ML services (NLP, risk classifier)"]
        S4["2025-09-23: Conduct internal alpha testing (chat + triage)"]
        S5["2025-09-30: Fix alpha-phase critical bugs & refine workflows"]
  end
 subgraph subGraph4["October 2025"]
        O1["2025-10-07: Prepare beta release candidate; stabilize features"]
        O2["2025-10-14: Launch beta to selected partners & gather feedback"]
        O3["2025-10-21: Iterate on chat UX & adjust protocol logic per feedback"]
        O4["2025-10-28: Enhance security (rate limiting, input sanitization)"]
  end
 subgraph subGraph5["November 2025"]
        N1["2025-11-04: Improve scalability (DB optimization, caching)"]
        N2["2025-11-11: Refine analytics & monitoring (Prometheus, ELK)"]
        N3["2025-11-18: Finalize user management module (roles, permissions)"]
        N4["2025-11-25: Conduct partner pilot testing & collect metrics"]
  end
 subgraph subGraph6["December 2025"]
        D1["2025-12-02: Implement mobile UI wrappers (React Native/Flutter)"]
        D2["2025-12-09: Integrate with pilot EHR systems (mock endpoints)"]
        D3["2025-12-16: Localize UI components (language resource files)"]
        D4["2025-12-23: Conduct end-to-end performance benchmarking"]
        D5["2025-12-30: Holiday-period bug fixes & documentation updates"]
  end
 subgraph subGraph7["January 2026"]
        Y1["2026-01-06: Finalize mobile support & publish iOS/Android builds"]
        Y2["2026-01-13: Launch public marketing site & documentation portal"]
        Y3["2026-01-20: Conduct security audit & compliance review"]
        Y4["2026-01-27: Optimize AI inference pipeline (batching, caching)"]
  end
 subgraph subGraph8["February 2026"]
        F1["2026-02-03: Expand RAG capabilities (ingest more guidelines)"]
        F2["2026-02-10: Add telehealth scheduling & appointment API"]
        F3["2026-02-17: Implement localization for EU/US (dates, formats)"]
        F4["2026-02-24: Conduct usability testing with international users"]
  end
 subgraph subGraph9["March 2026"]
        M1["2026-03-03: Roll out v1.1 with enhanced AI risk tools"]
        M2["2026-03-10: Generate comprehensive PDF reports"]
        M3["2026-03-17: Integrate third-party telemedicine providers"]
        M4["2026-03-24: Perform cross-region load testing (EU, US)"]
        M5["2026-03-31: Fix internationalization bugs & performance tweaks"]
  end
 subgraph subGraph10["April 2026"]
        R1["2026-04-07: Launch pilot in EU & US healthcare networks"]
        R2["2026-04-14: Implement continuous bias monitoring/remediation"]
        R3["2026-04-21: Develop analytics dashboard for partner insights"]
        R4["2026-04-28: Integrate insurance/payment gateways (open APIs)"]
  end
 subgraph subGraph11["May 2026"]
        M01["2026-05-05: Conduct final UAT & QA cycle"]
        M02["2026-05-12: Optimize data pipelines & archival strategies"]
        M03["2026-05-19: Complete documentation (developer, operator, user)"]
        M04["2026-05-26: Prepare v2.0 planning & roadmap definition"]
  end
 subgraph subGraph12["June 2026"]
        JU["2026-06-02: One-year retrospective & community roadmap kickoff"]
  end
    J4 --> J5
    J9 --> A1
    A4 --> S1
    S5 --> O1
    O4 --> N1
    N4 --> D1
    D5 --> Y1
    Y4 --> F1
    F4 --> M1
    M5 --> R1
    R4 --> M01
     J1:::default
     J2:::default
     J3:::default
     J4:::default
     J5:::default
     J6:::default
     J7:::default
     J8:::default
     J9:::default
     A1:::default
     A2:::default
     A3:::default
     A4:::default
     S1:::default
     S2:::default
     S3:::default
     S4:::default
     S5:::default
     O1:::default
     O2:::default
     O3:::default
     O4:::default
     N1:::default
     N2:::default
     N3:::default
     N4:::default
     D1:::default
     D2:::default
     D3:::default
     D4:::default
     D5:::default
     Y1:::default
     Y2:::default
     Y3:::default
     Y4:::default
     F1:::default
     F2:::default
     F3:::default
     F4:::default
     M1:::default
     M2:::default
     M3:::default
     M4:::default
     M5:::default
     R1:::default
     R2:::default
     R3:::default
     R4:::default
     M01:::default
     M02:::default
     M03:::default
     M04:::default
     JU:::default

```

-----

### 📝 TODO.md: v0 Prototype (10-Hour Sprint) - Chest Pain User Story

**Goal:** Build a working MVP for the chest pain triage flow: Patient chat (Mesop) → Case report creation (PostgreSQL) → Basic AI assessment (Ollama: Gemma 4B text/VLM, mxbai-embed, reasoning LLM) & optional file upload (MinIO) → Simple PDF report generation (ReportLab).

**Team Size:** 1 developer
**Environment:** Local HPC (RTX 3060 12GB, 5700X, 32GB RAM, Python 3.12)
**Constraint:** 10 hours

-----

**Phase 1: Backend Foundation & Core Logic (Est. 4 hours)**

  * **Hour 1: Project Setup & Basic Models (60 mins)**
      * [ ] Initialize FastAPI project structure (`app/main.py`, `app/models.py`, `app/core_logic.py`, `app/celery_worker.py`).
      * [ ] Define core Pydantic models for `CaseReportV0`, `Attachment`, `ChatInteraction` (as per User Story section).
      * [ ] Define corresponding SQLAlchemy models (or use Pydantic with a simple ORM helper) for PostgreSQL tables: `case_reports`, `attachments`.
      * [ ] Setup basic database connection logic (e.g., in `app/database.py`). Create tables.
      * [ ] Basic `.env` file with DB, Redis, MinIO, Ollama URLs.
      * [ ] `requirements.txt` with essential libraries.
  * **Hour 2: API Endpoints & Initial Chat Flow (60 mins)**
      * [ ] Implement FastAPI endpoint: `POST /api/v0/chat/initiate`.
          * Receives initial message.
          * Creates `CaseReportV0` instance.
          * (Mocked) Calls Gemma 4B (text) via Ollama to summarize initial complaint and store in Pydantic model.
          * Saves to `case_reports` table in PostgreSQL.
          * Returns `case_id` and first pre-programmed chest pain question.
      * [ ] Implement FastAPI endpoint: `POST /api/v0/chat/continue`.
          * Receives `case_id` and `answer`.
          * Appends Q\&A to a list in the `CaseReportV0` model (or separate chat log table).
          * (Mocked) Calls Gemma 4B to update structured fields in `CaseReportV0`.
          * Updates `case_reports` table.
          * Returns next pre-programmed question or "upload prompt".
      * [ ] Store \~5-7 pre-programmed chest pain questions (based on NICE research) in the backend.
  * **Hour 3: MinIO Integration & File Upload Logic (60 mins)**
      * [ ] Implement FastAPI endpoint: `POST /api/v0/upload_file` (or integrate into `/chat/continue`).
          * Receives `case_id` and `file`.
          * Triggers Celery task `upload_to_minio_task`.
      * [ ] Implement Celery task `upload_to_minio_task`:
          * Connects to MinIO.
          * Uploads file to specified bucket (e.g., `fairdoc-v0-cases`).
          * Gets MinIO URL/object key.
          * Creates `Attachment` record in PostgreSQL linked to `case_id`.
          * Updates `CaseReportV0` with attachment info.
      * [ ] Basic MinIO client setup.
  * **Hour 4: Celery Setup & Basic Background Tasks (60 mins)**
      * [ ] Configure Celery with Redis as broker/backend.
      * [ ] Implement basic structure for `process_case_report_task` (Celery).
          * Placeholder for vector embedding, 111/999 matching.
          * Placeholder for image analysis.
          * Placeholder for reasoning LLM.
      * [ ] Implement basic structure for `generate_pdf_report_task` (Celery).
      * [ ] Implement patient notification Celery task (mocked: prints to console).

-----

**Phase 2: Mesop Frontend & AI Integration (Est. 4 hours)**

  * **Hour 5: Basic Mesop Chat UI (60 mins)**
      * [ ] Create a simple Mesop app (`app/mesop_app.py`).
      * [ ] UI for displaying chat messages (user and system).
      * [ ] Input field for user to type messages.
      * [ ] "Send" button to call `/api/v0/chat/initiate` and `/api/v0/chat/continue`.
  * **Hour 6: Mesop File Upload & Displaying Questions (60 mins)**
      * [ ] Add file upload component in Mesop UI (using `me.uploader` if suitable or basic HTML input).
      * [ ] On file selection, Mesop calls `/api/v0/upload_file` (or its integrated version).
      * [ ] Ensure Mesop UI correctly displays sequential questions from backend.
  * **Hour 7: Integrating Ollama Calls (Gemma 4B Text & mxbai-embed) (60 mins)**
      * [ ] Flesh out Gemma 4B (text) call in `/chat/initiate` and `/chat/continue` (actual Ollama API call).
      * [ ] In `process_case_report_task` (Celery):
          * Implement vector embedding of case text using `mxbai-embed-large` via Ollama.
          * Store embeddings in Redis (temporarily) or directly use for ChromaDB.
      * [ ] (Simplified) ChromaDB: Pre-load 2-3 mock emergency/non-emergency vector embeddings. Implement basic cosine similarity matching against case embeddings. Update `CaseReportV0`.
  * **Hour 8: Integrating Image Analysis (Mocked/Simplified VLM) & Reasoning LLM (60 mins)**
      * [ ] In `process_case_report_task` (Celery) for image attachments:
          * Retrieve file from MinIO.
          * (Mocked/Simplified) Use Gemma 4B (VLM capabilities via Ollama) to get a text description. If direct VLM is tricky in time, use a fixed description or a simple text prompt to Gemma 4B *about* an image.
          * Update `CaseReportV0` with image description.
      * [ ] In `process_case_report_task` (Celery):
          * Call a reasoning LLM (Gemma 7B or even Gemma 4B with a good prompt via Ollama) with the case summary.
          * Prompt for the X-Y urgency/importance coordinate.
          * Prompt for a brief justification and general advice.
          * Update `CaseReportV0`.

-----

**Phase 3: PDF Generation & Final Polish (Est. 2 hours)**

  * **Hour 9: PDF Report Generation (ReportLab) (60 mins)**
      * [ ] Implement Celery task `generate_pdf_report_task`:
          * Retrieve fully processed `CaseReportV0`.
          * Use ReportLab to generate a simple PDF with: Case ID, Initial Complaint, Key Symptoms Summary (from Gemma 4B), X-ray description (from VLM), Urgency Coordinates, Justification, General Advice.
          * Upload PDF to MinIO.
          * Update `CaseReportV0` with PDF URL.
  * **Hour 10: End-to-End Flow Testing & Final Touches (60 mins)**
      * [ ] Test the entire user story flow from Mesop chat to PDF report link.
      * [ ] Implement the final decision logic (mocked doctor notification vs. patient report).
      * [ ] Display final advice and PDF link in Mesop UI.
      * [ ] Basic error handling (e.g., if Ollama fails, return a graceful message).
      * [ ] Quick code cleanup and comments.

-----

**Risk Mitigation (from `Fairdoc_AI_v0 Prototype (10-Hour Sprint).md` adapted for this sprint):**

  * **If AI Models Don't Work / Too Slow:** Use rule-based scoring for urgency, pre-programmed responses/PDF content. Mock Ollama calls heavily.
  * **If Mesop is Too Complex:** Fall back to extremely basic HTML forms + FastAPI or even just Postman testing of backend logic.
  * **If File Upload Fails:** Skip file processing for demo; use text-only assessment.
  * **If Time Runs Short:** Focus on core chat flow and mock AI responses. Skip PDF generation, use JSON responses in Mesop. Prioritize backend logic over perfect UI.

*(This 10-hour plan is ambitious. Realistic completion will depend on developer familiarity with tools. Focus on a demonstrable "happy path" for the user story.)*

-----

## 10\. 💰 Financial Model & Sustainability

*(Ref: Prompt)*

### Funding & Seed Model

  * **Initial Grants & Loans:**
      * India: Seek ₹100 Cr from government initiatives (e.g., Digital India, MeitY, Health Ministry grants), Development Bank Loans (SIDBI, NABARD).
      * UK: Seek £10 M from UK government (e.g., NHSX, Innovate UK), research councils.
  * **NGO & Foundational Partnerships:** Collaborate with organizations like WHO, Bill & Melinda Gates Foundation, Wellcome Trust for pilot programs and research funding.
  * **Venture Philanthropy / Impact Investors:** Attract funding from organizations focused on social impact and public good.

### Service Fees & Premiums

  * **Core Triage (Public Good):** Basic chat-based triage will be **free for individual users** in India and the UK, subsidized by government/grants.
  * **Premium AI Triage (B2B):**
      * Private Providers/Hospitals: Per-assessment fee for advanced multi-modal AI triage (e.g., India: ₹30–₹100/assessment; UK: £2-£5/assessment).
  * **Specialist Network Commission (Future):** A small commission (e.g., 5-15%) on teleconsultations successfully booked and completed via the Fairdoc specialist marketplace.
  * **SaaS for Institutions:**
      * Annual subscription fee for hospitals, clinic chains, and insurers for integrating Fairdoc into their workflows, dashboards, and potentially EHRs.
      * India: e.g., ₹5 Lakh/year per hospital network.
      * UK: e.g., £10k-£20k/year per NHS Trust or private hospital group, depending on scale and features.

### Subsidies & Grants (Ongoing)

  * **India UHI Participation:** Potential subsidies for supporting the Unified Health Interface, GPU infrastructure for AI, and deployment in rural/underserved areas (e.g., ₹50 Cr).
  * **UK NHS Digital / AI Lab Grants:** Ongoing grants for clinical validation, integration with NHS systems, and research on AI in healthcare (e.g., £5 M/year).
  * **Research Grants:** Joint applications with academic partners for research into AI ethics, bias mitigation, and clinical efficacy.

### Value-Added Services (B2B)

  * **Anonymized Analytics & Insights:** Provide aggregated, anonymized, and de-identified health trend data and insights to public health agencies, research institutions, and pharmaceutical companies (with strict ethical oversight and data privacy compliance). E.g., India: ₹20 Lakh/data package.
  * **Insurance Partner Referrals (Ethical):** With user consent, facilitate referrals to insurance partners for relevant health plans or services, potentially earning a small, transparent referral fee.
  * **Clinical Trial Support:** Platform features to assist in identifying and (with consent) recruiting eligible participants for clinical trials.

### Cost Estimates & Break-even Analysis (Illustrative)

*(Using cost estimates from Infrastructure section, highly simplified)*

**Assumptions:**

  * Targeting 1 active operational zone (e.g., a major city or region) initially.
  * Monthly operational cost per zone: \~$1,200 (infra) + staff + R\&D + compliance. Let's assume total ops cost of $20,000/month initially for a lean team and one zone ($240k/year).

**Revenue Projections (Year 2-3):**

  * **Free Triages:** 100,000/month (covered by government/grant subsidy equivalent to operational cost for this service).
  * **Premium Triages:** 20,000/month @ average $2 net revenue = $40,000/month.
  * **Institutional SaaS:** 10 hospitals/institutions @ average $1,000/month ($10k/year each for simplified example) = $10,000/month.
  * **Total Projected Revenue (excluding core subsidy):** $50,000/month.

**Break-even:**

  * If monthly revenue is $50,000 and ops costs are $20,000 (excluding major R\&D scaling), the model shows profitability for premium services.
  * The "free core triage" is the public good aspect, whose costs need to be covered by grants/subsidies. The $200k deficit mentioned in the prompt's example likely refers to the gap if *all* services were initially run without sufficient premium uptake or full subsidy coverage.
  * **Key to sustainability:** Secure foundational government/grant funding for the public good aspect, then scale B2B services to cross-subsidize and fund further innovation. Transparency in annual reports on usage, costs, funding sources, and impact is crucial.

-----

## 11\. ⚖️ Legal & Compliance

*(Ref: Prompt, `README.md`)*

### Data Privacy & Security

  * **UK/EU Compliance:**
      * **UK GDPR** and Data Protection Act 2018.
      * Principles: Data minimization, lawful basis for processing (primarily consent for personal health data), purpose limitation, accuracy, storage limitation, integrity & confidentiality, accountability.
      * User Rights: Right of access, rectification, erasure ("right to be forgotten"), restriction of processing, data portability, right to object.
      * **NHS Digital Data Security and Protection Toolkit (DSPT)** compliance.
  * **India Compliance:**
      * **Digital Personal Data Protection Act (DPDPA), 2023.**
      * Similar principles to GDPR: consent-based processing for personal data, purpose limitation, data minimization.
      * Requirements for Data Fiduciaries and Data Processors.
      * Data Breach notification requirements.
      * Ayushman Bharat Digital Mission (ABDM) standards for health data interoperability and security.
  * **General Security Measures:**
      * **Encryption:** AES-256 for data at rest (PostgreSQL TDE, MinIO server-side encryption). TLS 1.3 for data in transit. End-to-end encryption for chat messages where platform allows (Signal), client-side encryption considerations for sensitive data before storage.
      * **Access Control:** Role-Based Access Control (RBAC) for backend systems and databases.
      * **Audit Logging:** Comprehensive and immutable audit trails for data access, modifications, and system interactions. (As per Logging & Monitoring section).
      * **De-identification/Pseudonymization:** For data used in analytics and research.
      * **ISO 27001** Information Security Management & **SOC 2 Type II** compliance (aspirational targets for mature platform).
  * **HIPAA (For potential US Expansion):**
      * Awareness of requirements: Business Associate Agreements (BAAs) with partners, technical/physical/administrative safeguards.

### Medical Device Regulations

  * **UK MHRA (Medicines and Healthcare products Regulatory Agency):**
      * Classification: Software as a Medical Device (SaMD). Likely Class IIa if it provides information used to make decisions with diagnosis or therapeutic purposes.
      * Requirements: Clinical Safety Case (DCB0129/DCB0160 standards), CE marking (or UKCA marking post-Brexit). Appoint UK Responsible Person.
  * **India CDSCO (Central Drugs Standard Control Organisation):**
      * Registration as Software as a Medical Device (SMD) under Medical Device Rules, 2017. Classification will depend on risk (likely Class B or C).
  * **FDA (US - for future expansion):**
      * Potential for 510(k) submission for clinical decision support software, depending on claims and risk.

### Open-Source License

  * **📜 MIT License** (as specified in prompt). This allows permissive use, modification, and distribution, fostering collaboration.

<!-- end list -->

```
MIT License

Copyright (c) 2024-2025 Fairdoc AI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Clinical Content Licensing

  * **NICE Guidelines (UK):**
      * Utilize NICE Syndication API where possible.
      * Adhere to NICE licensing terms: typically for non-commercial use within specific contexts, requires attribution, display of NICE logo. For a DPI, specific agreements might be needed.
  * **MoHFW / ICMR Protocols (India):**
      * Many guidelines are in the public domain or available under Creative Commons licenses (e.g., CC-BY).
      * Ensure proper citations, linkbacks to original sources, and adherence to any specific terms of use.
  * **Disclaimer:** Fairdoc AI is an assistive tool and **NOT** a replacement for professional medical advice or diagnosis. All AI-generated recommendations for critical care paths must be reviewed and approved by qualified healthcare professionals. The system is designed to augment, not replace, clinical judgment. *(Ref: README.md)*

### Transparency & Ethics

  * **Open Governance Charter (Proposed):** Establish a public board including clinicians, AI developers, ethicists, patient representatives, and public health officials.
  * **Quarterly Transparency Reports:** Publicly available reports (e.g., on GitHub or dedicated website) covering usage statistics, performance metrics, bias audit results, funding, and governance decisions.
  * **Bias Monitoring & Mitigation:**
      * Continuous fairness audits of AI models against various demographic groups (age, gender, ethnicity, socioeconomic status where ethically sourced). Publish anonymized metrics.
      * Implement techniques for bias detection (e.g., Aequitas, IBM AI Fairness 360 concepts) and mitigation (pre-processing, in-processing, post-processing).
      * Annual external audit by an independent AI ethics board or academic partners (e.g., research partners like University of Oxford, Alan Turing Institute). *(Ref: README.md)*

-----

## 12\. 🤝 Contribution Guidelines

*(Ref: Prompt)*

### Get Started 🔨

1.  **Fork & Clone:**
    ```bash
    git clone https://github.com/fairworks/fairdoc-core.git # (Example URL)
    cd fairdoc-core
    git checkout -b feature/your-brilliant-feature
    ```
2.  **Setup Environment:** Follow `README.md` or `DEVELOPER_GUIDE.md` for detailed setup (Python version, dependencies, Docker, environment variables).
    ```bash
    pip install -r requirements.txt
    # Potentially: docker-compose up -d
    ```
3.  **Review Documentation:** Familiarize yourself with `CONTRIBUTING.md` (this section, expanded) and `CODE_OF_CONDUCT.md`.

### Issue Tracking 🐛

  * Use the GitHub Issues tracker for bugs, feature requests, documentation improvements, etc.
  * Check existing issues before creating a new one.
  * Use descriptive titles and provide clear steps to reproduce for bugs.
  * Label issues appropriately (e.g., `bug`, `feature`, `documentation`, `frontend`, `backend`, `ai`, `ops`, `good first issue`).

### Development Workflow 🛠️

  * **Branching Strategy (GitFlow Inspired):**
      * `main`: Stable, production-ready code. Protected branch. Merges only from `release` or `hotfix` branches.
      * `develop`: Latest integrated features. Base for new feature branches.
      * `feature/<short-description>`: For new features or improvements. Branched from `develop`.
      * `release/<version>`: For preparing a new production release. Branched from `develop`. Bug fixes and release-specific changes.
      * `hotfix/<issue-number>`: For critical bugs in production. Branched from `main`, merged back to `main` and `develop`.
  * **Code Style & Quality:**
      * Follow PEP 8 for Python code.
      * Use pre-commit hooks (e.g., Black for formatting, Flake8 for linting, Mypy for type checking). Configuration will be in `pyproject.toml` or `.pre-commit-config.yaml`.
      * Write clear, commented code where logic is complex.
  * **Commit Messages:** Write meaningful commit messages (e.g., "feat: Implement user registration endpoint", "fix: Correct calculation in risk score module").

### Testing Requirements 🧪

  * **Unit Tests:** New code should aim for \>90% unit test coverage. Use `pytest`. Test individual functions, classes, and modules.
  * **Integration Tests:** Test interactions between components (e.g., API endpoint → AI service → Database).
  * **Bias Tests:** For AI models, include tests to evaluate performance across different demographic subgroups to identify and mitigate potential biases. (Frameworks and datasets to be defined).
  * **Async Testing:** Use `pytest-asyncio` for testing asynchronous FastAPI endpoints and Celery tasks.
  * Run tests locally before pushing: `pytest`

### Pull Request (PR) Process 📬

1.  **Push Changes:** Push your feature branch to your fork on GitHub.
    ```bash
    git push origin feature/your-brilliant-feature
    ```
2.  **Create PR:** Submit a Pull Request against the `develop` branch of the main Fairdoc repository.
3.  **PR Template:** Fill out the PR template:
      * Clear description of changes.
      * Reference to the issue number(s) it resolves (e.g., "Closes \#123").
      * Summary of testing performed.
      * Screenshots or GIFs if UI changes are involved.
      * Any specific instructions for reviewers.
4.  **Code Review:**
      * At least one core contributor/maintainer will review the PR.
      * Address review comments and push updates to your feature branch.
      * Automated checks (CI pipeline: linting, tests) must pass.
5.  **Merge:** Once approved and all checks pass, a maintainer will merge the PR.

### Why Contribute? 🌟

  * **Impact:** Contribute to building a vital piece of Digital Public Infrastructure that can improve healthcare access and equity for millions.
  * **Learn & Grow:** Work with cutting-edge AI/ML technologies, cloud infrastructure, and engage in a complex, real-world problem domain.
  * **Recognition:** Contributors will be acknowledged in project documentation, website, and potentially annual reports.
  * **Research Opportunities:** Potential to collaborate with esteemed academic partners (Imperial College London, University of Oxford, Cambridge University, Alan Turing Institute) on research aspects.
  * **Community:** Become part of a global community dedicated to responsible AI, open-source development, and digital health equity.

-----

## 13\. 🎉 Acknowledgments

*(From `README.md`)*

### 🏆 Special Thanks

  * **NHS Digital:** For FHIR R4 guidance, API access, and clinical pathway insights.
  * **DevPost RAI Hackathon:** For promoting responsible AI in healthcare and inspiring initial concepts.
  * **OpenAI & Anthropic:** For advancing conversational AI capabilities that inform possibilities.
  * **Ollama Team:** For enabling accessible local LLM deployment, crucial for experimentation and privacy-preserving options.
  * **ChromaDB (and other Vector DBs):** For efficient vector storage and retrieval solutions vital for RAG.
  * **The wider Open Source Community:** For the countless libraries and tools that make a project like Fairdoc feasible.

### 🔬 Research Partners (Existing & Aspirational)

  * **Imperial College London:** Clinical validation studies.
  * **University of Oxford:** Bias mitigation research.
  * **Cambridge University:** Healthcare AI ethics framework.
  * **Alan Turing Institute:** Responsible AI development methodologies.
  * **Indian Institutes of Technology (IITs) & AIIMS (All India Institutes of Medical Sciences):** For collaboration on adapting and validating for the Indian context.

### 💡 Inspiration

> *"Healthcare should be a human right, not a privilege. Technology should bridge gaps, not create them. AI should amplify human compassion, not replace it."*

\<div style="text-align: center"\>**Built with ❤️ for healthcare equity and responsible AI**\</div\>

-----

## 14\. 📚 References

*(From Deep Research)*

  * **Digital Public Infrastructure (DPI):**
      * Stanford Social Innovation Review – Digital Public Infrastructure for the Developing World
      * World Economic Forum – Building inclusive digital public infrastructure
      * MOSIP.io, Beckn Protocol
  * **India's Digital Stack:**
      * IBM – What is ONDC?
      * Economic Times – Is ONDC’s gamble paying off?
      * BIS (Bank for Int’l Settlements) – Lessons from India’s UPI
      * Moneycontrol – The real cost of UPI…
      * Ayushman Bharat Digital Mission (ABDM) & Unified Health Interface (UHI) documentation.
  * **Healthcare & AI Guidelines:**
      * NICE – Syndication Service & API Guide, Clinical Guidelines (CG95 - Chest Pain, etc.)
      * MoHFW India & ICMR - Standard Treatment Guidelines.
      * WHO - Ethics and governance of artificial intelligence for health.
  * **Open Source & Community:**
      * Digital Square – Global Goods for Health
  * **Technical & Cloud:**
      * AWS EC2 Pricing (g4dn.xlarge, g4dn.2xlarge, etc.)
      * Google Cloud, Azure pricing for comparable GPU and compute instances.
      * FastAPI, Celery, Ollama, ChromaDB, ReportLab, Mesop documentation.
  * **Responsible AI Toolkits:**
      * Aequitas, IBM AI Fairness 360, Google Responsible AI Toolkit, Microsoft Responsible AI.
  * **Data Protection:**
      * UK ICO (Information Commissioner's Office) - Guide to GDPR, Data Protection Act 2018.
      * India's Digital Personal Data Protection Act, 2023 text.

-----
