# 📖 User Story: v0 Prototype - Chest Pain Triage (10-Hour Sprint Focus)

**User:** A patient (e.g., "Alex," 45 years old) experiencing new-onset left chest pain, accompanied by coughing and wheezing. Alex is concerned and decides to use the Fairdoc service via a web interface (Mesop prototype).

**Pre-conditions:**
* The Fairdoc v0 backend (FastAPI) is running.
* The Mesop frontend is accessible.
* Basic infrastructure (PostgreSQL, MinIO, Redis, Ollama with Gemma 4B and mxbai-embed-large) is set up and minimally configured.
* Pre-programmed questions for chest pain (based on simplified NHS/NICE guidance) are available.
* Mock `111.json` (non-emergency) and `999.json` (emergency) vector embeddings are pre-loaded in ChromaDB (or a simplified matching mechanism is in place).

## User Journey & Engineering Steps:

1.  **Initiates Chat (Frontend: Mesop | Backend: FastAPI)**
    * **Alex:** Opens the Fairdoc Mesop web app and starts a new chat session.
    * **System (Mesop UI):** Displays a welcome message and a prompt to describe their main symptom.
    * **Alex:** Types "I have left chest pain with coughing and wheezing."
    * **Engineering (Frontend):** Mesop sends this initial message to the FastAPI backend (`/api/v0/chat/initiate`).

2.  **Initial Symptom Processing & Case Report Creation (Backend: FastAPI, Gemma 4B, Pydantic, PostgreSQL)**
    * **Engineering (Backend):**
        * The FastAPI endpoint receives the message.
        * A **CaseReport Pydantic model instance** is created with a unique `case_id`.
        * The initial message is **silently processed by Gemma 4B (via Ollama)** to extract initial keywords/symptoms and populate relevant fields in the `CaseReport` Pydantic model.
        * The initial `CaseReport` is **saved to the PostgreSQL `case_reports` table**. Status: "PENDING_QUESTIONS".

3.  **Guided Questioning (Frontend: Mesop | Backend: FastAPI)**
    * **Engineering (Backend):** Based on keywords, the backend retrieves a **pre-programmed set of questions** (simplified NICE guidance).
        * *NHS NICE "chest pain first contact questions" typically involve: PQRST (Provocation/Palliation, Quality, Radiation, Severity, Timing), associated symptoms, relevant history.*
    * **System (Mesop UI):** Displays the first question.
    * **Alex:** Responds sequentially.
    * **Engineering (Frontend & Backend):** Mesop sends answers; backend appends to transcript, Gemma 4B updates `CaseReport`, which is saved to PostgreSQL.

4.  **Optional File Upload (Frontend: Mesop | Backend: FastAPI, MinIO, Celery)**
    * **System (Mesop UI):** Prompts for optional report/X-ray upload.
    * **Alex:** Uploads a PDF and a JPG.
    * **Engineering (Frontend):** Mesop uploads to FastAPI endpoint.
    * **Engineering (Backend - FastAPI):** Triggers Celery task `upload_to_minio_task`.
    * **Engineering (Celery Worker - `upload_to_minio_task`):** Uploads to MinIO, gets URL, updates `CaseReport` in PostgreSQL with attachment info.
    * **System (Mesop UI):** Shows "File uploaded successfully."

5.  **Case Finalization & Initial Notification (Backend: FastAPI, Celery)**
    * **System (Mesop UI):** "Thank you... We are now processing your case."
    * **Engineering (Backend):** `CaseReport` status updated to "PROCESSING". Celery task (`patient_notification_task`) (conceptually) sends notification.

6.  **Background Processing - Phase 1: Embedding & Basic Matching (Backend: Celery, mxbai-embed-large, Redis, ChromaDB/PostgreSQL)**
    * **Celery Worker (`process_case_report_task` - Step 1):** Retrieves `CaseReport`. Textual fields embedded using `mxbai-embed-large`. Embeddings cached in Redis. Matched against `111.json`/`999.json` embeddings in ChromaDB. Matches added to `CaseReport`. (EHR matching mocked/skipped for v0).

7.  **Background Processing - Phase 2: Multi-modal Analysis (Backend: Celery, OpenCV, SAM2.1 (mocked), Gemma 4B VLM)**
    * **Celery Worker (`process_case_report_task` - Step 2 - for each image):** Retrieves image from MinIO. OpenCV for basic processing. SAM2.1 mocked. Gemma 4B (VLM) generates textual description. Description added to `CaseReport`. (PDF text extraction simplified/deferred for v0).

8.  **Background Processing - Phase 3: Reasoning & Urgency Scoring (Backend: Celery, DeepSeek/Mistral-like model)**
    * **Celery Worker (`process_case_report_task` - Step 3):** Comprehensive `CaseReport` fed to reasoning LLM (simulated via Gemma 7B/4B if needed for v0). LLM provides X-Y urgency/importance plot coordinate. Score saved to `CaseReport`.

9.  **Background Processing - Phase 4: PDF Report Generation (Backend: Celery, ReportLab)**
    * **Celery Worker (`generate_pdf_report_task`):** Retrieves processed `CaseReport`. ReportLab generates PDF (transcript summary, VLM description, urgency score, justification, general advice). PDF uploaded to MinIO, URL added to `CaseReport`. Status: "REPORT_GENERATED".

10. **Decision & Notification/Escalation (Backend: FastAPI/Celery)**
    * **Engineering (Backend - `final_disposition_task`):** Based on `urgency_coordinates`.
        * **If highly urgent:** Flagged for (mock) doctor review. Alert (conceptual) sent to doctor. Alex notified of urgency and escalation.
        * **If not life-threatening (v0 path):** PDF report link and general advice (from reasoning LLM) sent to Alex via Mesop.
    * **System (Mesop UI):** Alex sees final message, advice, PDF link.
    * `CaseReport` status updated (e.g., "COMPLETED_PATIENT_NOTIFIED").

## Pydantic Models (Illustrative for v0):

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
    
    attachments: List[Attachment] = [] 
    
    vector_embeddings_cached_redis: bool = False
    matched_emergency_protocols: Optional[List[str]] = None 
    
    urgency_coordinates: Optional[Dict[str, float]] = None 
    reasoning_llm_justification: Optional[str] = None
    general_advice_from_llm: Optional[str] = None
    
    generated_pdf_report_url: Optional[HttpUrl] = None

class UserMessage(BaseModel):
    session_id: Optional[str] = None 
    text: str
    attachments: Optional[List[Dict[str, str]]] = None 

class AIResponseMessage(BaseModel):
    session_id: str
    response_text: str
    next_questions_ids: Optional[List[str]] = None 
    request_file_upload: bool = False
    report_link: Optional[HttpUrl] = None

```
