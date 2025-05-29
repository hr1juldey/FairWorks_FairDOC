# TODO.md - Fairdoc AI v1 Prototype (10-Hour Sprint)

## 🎯 **Sprint Goal**

Build a working MVP with chest pain triage flow: Patient chat → Case report → AI assessment → PDF report generation

---

## ⏰ **Time Allocation (10 Hours Total) - CURRENT PROGRESS: 2/10 HOURS COMPLETE**

### **✅ Phase 1: Core Infrastructure Setup (2 hours) - COMPLETED**

- [x] **Hour 1: Environment & Database Setup - ✅ COMPLETED**
  - [x] Set up PostgreSQL database with Docker ✅
  - [x] Configure Redis for caching ✅
  - [x] Set up MinIO for object storage ✅
  - [x] **ADDED:** Set up ChromaDB vector database ✅
  - [x] Create comprehensive `.env` configuration ✅
  - [x] **ENHANCED:** Create docker-compose.yaml for all services ✅
  - [x] **ENHANCED:** Test all connections (5/6 services working) ✅

- [x] **Hour 2: Essential Data Models - ✅ COMPLETED**
  - [x] **ENHANCED:** Create Pydantic v2 compliant models with ML scoring ✅
  - [x] **ARCHITECTURAL FIX:** Separate SQLAlchemy models properly ✅
  - [x] **MAJOR ADDITION:** Create 10 comprehensive NICE protocols JSON ✅
  - [x] **ENHANCED:** Create database tables with mutable JSON support ✅
  - [x] **ENHANCED:** Implement clean architecture separation ✅

### **🔄 Phase 2: Core Backend API (3 hours) - READY TO START**

- [ ] **Hour 3: FastAPI Core Setup**
  - [ ] Configure v1/app.py with essential endpoints
  - [ ] Create database connection utilities
  - [ ] Set up basic error handling and middleware
  - [ ] Implement health check endpoints
  - [ ] Test API startup and basic functionality

- [ ] **Hour 4: NICE Protocol Questions API**
  - [ ] Build `/api/v1/protocols` endpoint (list all protocols)
  - [ ] Build `/api/v1/protocols/{protocol_id}/questions` endpoint
  - [ ] Build `/api/v1/questions/by-urgency/{category}` endpoint
  - [ ] Build `/api/v1/responses/submit` endpoint
  - [ ] Implement protocol selection logic

- [ ] **Hour 5: File Upload & Case Report API**
  - [ ] Create `/api/v1/files/upload` endpoint (MinIO integration)
  - [ ] Build `/api/v1/case-reports/create` endpoint
  - [ ] Build `/api/v1/case-reports/{case_id}` CRUD endpoints
  - [ ] Implement file processing and validation
  - [ ] Test complete case report workflow

### **🔄 Phase 3: Mesop Frontend (2 hours) - PENDING**

- [ ] **Hour 6: Basic Mesop Chat Interface**
  - [ ] Create `frontend/chat_app.py` with protocol selection
  - [ ] Build WhatsApp-like chat UI with enhanced UX
  - [ ] Implement dynamic question-answer flow
  - [ ] Add typing indicators and progress tracking

- [ ] **Hour 7: File Upload & Progress UI**
  - [ ] Add file upload component with drag-drop
  - [ ] Create real-time progress indicators
  - [ ] Connect to backend APIs with error handling
  - [ ] Add responsive design and mobile support

### **🔄 Phase 4: AI Integration & Background Jobs (2 hours) - PENDING**

- [ ] **Hour 8: Celery Setup & Basic Jobs**
  - [ ] Configure Celery with Redis broker
  - [ ] Create `process_case_report` task
  - [ ] Create `generate_pdf_report` task with ReportLab
  - [ ] Implement notification system

- [ ] **Hour 9: Enhanced AI Integration**
  - [ ] Set up Ollama with multiple models (Gemma, DeepSeek, Qwen)
  - [ ] Implement ML-based triage scoring
  - [ ] Create urgency/importance coordinate mapping
  - [ ] Implement basic bias monitoring

### **🔄 Phase 5: Integration & Testing (1 hour) - PENDING**

- [ ] **Hour 10: End-to-End Testing & Polish**
  - [ ] Full user flow testing across all 10 protocols
  - [ ] Fix critical bugs and performance issues
  - [ ] Basic error handling improvements
  - [ ] Demo preparation and documentation

---

## 📋 **Detailed Implementation Tasks**

### **🏗️ Infrastructure Setup**

#### **Database Schema (Priority: HIGH)**

```sql
-- Case Reports Table
CREATE TABLE case_reports (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50),
    chief_complaint TEXT,
    symptoms JSONB,
    medical_history JSONB,
    uploaded_files JSONB,
    ai_assessment JSONB,
    urgency_score FLOAT,
    importance_score FLOAT,
    status VARCHAR(20) DEFAULT 'processing',
    created_at TIMESTAMP DEFAULT NOW()
);

-- NICE Protocol Questions
CREATE TABLE nice_questions (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50),
    question_text TEXT,
    question_type VARCHAR(20),
    options JSONB,
    order_index INTEGER
);

-- Patient Responses
CREATE TABLE patient_responses (
    id SERIAL PRIMARY KEY,
    case_report_id INTEGER REFERENCES case_reports(id),
    question_id INTEGER REFERENCES nice_questions(id),
    response_text TEXT,
    response_data JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```


#### **Environment Configuration**

```bash
# .env file
DATABASE_URL=postgresql://user:password@localhost:5432/fairdoc_v0
REDIS_URL=redis://localhost:6379/0
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
OLLAMA_BASE_URL=http://localhost:11434
CELERY_BROKER_URL=redis://localhost:6379/1
```


### **🔧 Pydantic Models (v1/datamodels/medical_model.py)**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SymptonSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate" 
    SEVERE = "severe"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ChestPainCharacteristics(BaseModel):
    location: str = Field(..., description="Pain location")
    radiation: bool = Field(False, description="Does pain radiate")
    radiation_areas: List[str] = Field(default=[], description="Areas where pain radiates")
    severity: int = Field(..., ge=1, le=10, description="Pain severity 1-10")
    character: str = Field(..., description="Pain character (crushing, sharp, etc.)")
    onset: str = Field(..., description="How pain started")
    duration: str = Field(..., description="How long pain has lasted")
    triggers: List[str] = Field(default=[], description="What triggers the pain")
    relief_factors: List[str] = Field(default=[], description="What relieves the pain")

class AssociatedSymptoms(BaseModel):
    shortness_of_breath: bool = False
    cough: bool = False
    wheeze: bool = False
    sweating: bool = False
    nausea: bool = False
    vomiting: bool = False
    dizziness: bool = False
    palpitations: bool = False
    fever: bool = False
    fatigue: bool = False

class VitalSigns(BaseModel):
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    temperature: Optional[float] = None

class UploadedFile(BaseModel):
    file_id: str
    filename: str
    file_type: str  # 'pdf', 'image', 'audio'
    minio_url: str
    upload_timestamp: datetime
    file_size: int
    description: Optional[str] = None

class AIAssessment(BaseModel):
    urgency_score: float = Field(..., ge=-1, le=1, description="Urgency score -1 to 1")
    importance_score: float = Field(..., ge=-1, le=1, description="Importance score -1 to 1")
    predicted_conditions: List[str] = Field(default=[], description="Likely conditions")
    recommended_actions: List[str] = Field(default=[], description="Recommended next steps")
    reasoning: str = Field(..., description="AI reasoning for assessment")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in assessment")

class CaseReport(BaseModel):
    id: Optional[int] = None
    patient_id: str = Field(..., description="Patient identifier")
    
    # Demographics
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., description="Patient gender")
    
    # Chief complaint
    chief_complaint: str = Field(..., description="Main complaint")
    
    # Detailed symptoms
    chest_pain: Optional[ChestPainCharacteristics] = None
    associated_symptoms: AssociatedSymptoms = AssociatedSymptoms()
    vital_signs: Optional[VitalSigns] = None
    
    # Medical history
    medical_history: List[str] = Field(default=[], description="Past medical conditions")
    current_medications: List[str] = Field(default=[], description="Current medications")
    allergies: List[str] = Field(default=[], description="Known allergies")
    family_history: List[str] = Field(default=[], description="Relevant family history")
    
    # Uploaded files
    uploaded_files: List[UploadedFile] = Field(default=[], description="Uploaded documents/images")
    
    # AI assessment
    ai_assessment: Optional[AIAssessment] = None
    
    # Status tracking
    status: str = Field(default="created", description="Processing status")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class NICEProtocolQuestion(BaseModel):
    id: int
    category: str  # 'demographics', 'symptoms', 'history', etc.
    question_text: str
    question_type: str  # 'text', 'multiple_choice', 'yes_no', 'scale'
    options: Optional[List[str]] = None
    is_required: bool = True
    order_index: int

class PatientResponse(BaseModel):
    question_id: int
    response_text: str
    response_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
```


### **📋 NHS NICE Protocol Questions (Data)**

```python
# v1/data/nice_chest_pain_questions.py
CHEST_PAIN_QUESTIONS = [
    # Demographics
    {
        "id": 1,
        "category": "demographics",
        "question_text": "What is your age?",
        "question_type": "number",
        "is_required": True,
        "order_index": 1
    },
    {
        "id": 2, 
        "category": "demographics",
        "question_text": "What is your gender?",
        "question_type": "multiple_choice",
        "options": ["Male", "Female", "Other", "Prefer not to say"],
        "is_required": True,
        "order_index": 2
    },
    
    # Chief complaint
    {
        "id": 3,
        "category": "chief_complaint", 
        "question_text": "Can you describe your main concern today?",
        "question_type": "text",
        "is_required": True,
        "order_index": 3
    },
    
    # Pain characteristics
    {
        "id": 4,
        "category": "pain_characteristics",
        "question_text": "Where exactly is your chest pain located?",
        "question_type": "multiple_choice",
        "options": ["Left chest", "Right chest", "Center chest", "Across entire chest", "Back"],
        "is_required": True,
        "order_index": 4
    },
    {
        "id": 5,
        "category": "pain_characteristics",
        "question_text": "On a scale of 1-10, how severe is your pain?",
        "question_type": "scale",
        "options": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "is_required": True,
        "order_index": 5
    },
    {
        "id": 6,
        "category": "pain_characteristics",
        "question_text": "How would you describe the pain?",
        "question_type": "multiple_choice",
        "options": ["Sharp", "Crushing", "Burning", "Aching", "Stabbing", "Tight/Squeezing"],
        "is_required": True,
        "order_index": 6
    },
    {
        "id": 7,
        "category": "pain_characteristics",
        "question_text": "Does the pain spread to other areas?",
        "question_type": "yes_no",
        "is_required": True,
        "order_index": 7
    },
    {
        "id": 8,
        "category": "pain_characteristics",
        "question_text": "When did the pain start?",
        "question_type": "multiple_choice",
        "options": ["Less than 1 hour ago", "1-6 hours ago", "6-24 hours ago", "More than 1 day ago"],
        "is_required": True,
        "order_index": 8
    },
    
    # Associated symptoms
    {
        "id": 9,
        "category": "associated_symptoms",
        "question_text": "Are you experiencing any of these symptoms along with chest pain?",
        "question_type": "multiple_select",
        "options": ["Shortness of breath", "Cough", "Wheezing", "Sweating", "Nausea", "Dizziness", "Palpitations"],
        "is_required": True,
        "order_index": 9
    },
    
    # Medical history
    {
        "id": 10,
        "category": "medical_history",
        "question_text": "Do you have any of these medical conditions?",
        "question_type": "multiple_select", 
        "options": ["Heart disease", "High blood pressure", "Diabetes", "Asthma", "COPD", "Previous heart attack", "None"],
        "is_required": True,
        "order_index": 10
    },
    {
        "id": 11,
        "category": "medical_history",
        "question_text": "Are you currently taking any medications?",
        "question_type": "text",
        "is_required": False,
        "order_index": 11
    },
    
    # Recent activities
    {
        "id": 12,
        "category": "triggers",
        "question_text": "What were you doing when the pain started?",
        "question_type": "multiple_choice",
        "options": ["Resting", "Light activity", "Moderate exercise", "Heavy exercise", "Emotional stress", "Other"],
        "is_required": True,
        "order_index": 12
    },
    
    # File uploads
    {
        "id": 13,
        "category": "files",
        "question_text": "Do you have any recent medical reports or chest X-rays to upload? (Optional)",
        "question_type": "file_upload",
        "is_required": False,
        "order_index": 13
    }
]
```


### **🌐 API Endpoints (v1/api/routes.py)**

```python
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List
import uuid
from ..datamodels.medical_model import CaseReport, NICEProtocolQuestion, PatientResponse
from ..services.case_service import CaseService
from ..services.file_service import FileService
from ..core.database import get_db

router = APIRouter(prefix="/api/v1")

@router.get("/questions/chest-pain", response_model=List[NICEProtocolQuestion])
async def get_chest_pain_questions():
    """Get NICE protocol questions for chest pain assessment"""
    from ..data.nice_chest_pain_questions import CHEST_PAIN_QUESTIONS
    return [NICEProtocolQuestion(**q) for q in CHEST_PAIN_QUESTIONS]

@router.post("/case-reports", response_model=dict)
async def create_case_report(patient_id: str = None, db = Depends(get_db)):
    """Initialize a new case report"""
    if not patient_id:
        patient_id = str(uuid.uuid4())
    
    case_service = CaseService(db)
    case_id = await case_service.create_case_report(patient_id)
    
    return {"case_id": case_id, "patient_id": patient_id, "status": "created"}

@router.post("/case-reports/{case_id}/responses")
async def submit_response(
    case_id: int,
    responses: List[PatientResponse],
    db = Depends(get_db)
):
    """Submit patient responses to questions"""
    case_service = CaseService(db)
    await case_service.add_responses(case_id, responses)
    return {"status": "responses_saved", "case_id": case_id}

@router.post("/files/upload")
async def upload_file(
    case_id: int,
    file: UploadFile = File(...),
    description: str = None
):
    """Upload medical files (PDFs, X-rays, etc.)"""
    file_service = FileService()
    
    # Validate file type
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/dicom"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Upload to MinIO and get URL
    file_url = await file_service.upload_file(file, case_id)
    
    return {
        "file_id": str(uuid.uuid4()),
        "filename": file.filename,
        "file_url": file_url,
        "status": "uploaded"
    }

@router.post("/case-reports/{case_id}/process")
async def process_case_report(case_id: int, db = Depends(get_db)):
    """Trigger AI processing of complete case report"""
    case_service = CaseService(db)
    
    # Queue background processing
    from ..tasks.process_case import process_case_report_task
    task = process_case_report_task.delay(case_id)
    
    return {
        "status": "processing_started",
        "task_id": task.id,
        "case_id": case_id
    }

@router.get("/case-reports/{case_id}")
async def get_case_report(case_id: int, db = Depends(get_db)):
    """Get case report details"""
    case_service = CaseService(db)
    case_report = await case_service.get_case_report(case_id)
    
    if not case_report:
        raise HTTPException(status_code=404, detail="Case report not found")
    
    return case_report

@router.get("/case-reports/{case_id}/pdf")
async def download_case_report_pdf(case_id: int, db = Depends(get_db)):
    """Download generated PDF report"""
    case_service = CaseService(db)
    pdf_url = await case_service.get_pdf_report_url(case_id)
    
    if not pdf_url:
        raise HTTPException(status_code=404, detail="PDF report not ready")
    
    return {"pdf_url": pdf_url}
```


### **🎨 Mesop Frontend (frontend/chat_app.py)**

```python
import mesop as me
import mesop.labs as mel
from typing import List, Dict, Any
import requests
import json

# State management
@me.stateclass
class AppState:
    current_question_index: int = 0
    case_id: int = None
    patient_id: str = None
    responses: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    messages: List[Dict[str, str]] = []
    is_processing: bool = False
    upload_status: str = ""
    
API_BASE = "http://localhost:8000/api/v1"

def load_questions():
    """Load NICE protocol questions from API"""
    try:
        response = requests.get(f"{API_BASE}/questions/chest-pain")
        return response.json()
    except:
        return []

@me.page(path="/", title="Fairdoc AI - Medical Triage")
def main_page():
    state = me.state(AppState)
    
    # Initialize questions if not loaded
    if not state.questions:
        state.questions = load_questions()
        if state.questions:
            # Create initial case report
            try:
                response = requests.post(f"{API_BASE}/case-reports")
                data = response.json()
                state.case_id = data["case_id"]
                state.patient_id = data["patient_id"]
                
                # Add welcome message
                state.messages.append({
                    "type": "system",
                    "content": "Hello! I'm your NHS AI triage assistant. I'll ask you some questions about your chest pain to help assess your condition. Let's start:"
                })
                
                # Ask first question
                ask_current_question(state)
            except Exception as e:
                state.messages.append({
                    "type": "error", 
                    "content": f"Sorry, there was an error starting your assessment. Please try again."
                })

    with me.box(style=me.Style(
        background="#f0f2f5",
        height="100vh",
        display="flex",
        flex_direction="column"
    )):
        # Header
        with me.box(style=me.Style(
            background="#075e54",
            color="white",
            padding=me.Padding.all(16),
            text_align="center"
        )):
            me.text("NHS Fairdoc AI Triage", style=me.Style(font_size=20, font_weight="bold"))
            if state.case_id:
                me.text(f"Case ID: {state.case_id}", style=me.Style(font_size=12, opacity=0.8))

        # Chat messages area
        with me.box(style=me.Style(
            flex_grow=1,
            overflow_y="auto",
            padding=me.Padding.all(16),
            background="white"
        )):
            for message in state.messages:
                render_message(message)

        # Input area
        if not state.is_processing and state.current_question_index < len(state.questions):
            render_input_area(state)
        elif state.is_processing:
            render_processing_status(state)
        else:
            render_completion_status(state)

def render_message(message: Dict[str, str]):
    """Render a chat message"""
    is_user = message["type"] == "user"
    
    with me.box(style=me.Style(
        display="flex",
        justify_content="flex-end" if is_user else "flex-start",
        margin=me.Margin(bottom=8)
    )):
        with me.box(style=me.Style(
            background="#dcf8c6" if is_user else "#e5e5ea",
            border_radius=12,
            padding=me.Padding.all(12),
            max_width="70%",
            color="#000" if not is_user else "#000"
        )):
            me.text(message["content"])

def ask_current_question(state: AppState):
    """Add current question to chat"""
    if state.current_question_index < len(state.questions):
        question = state.questions[state.current_question_index]
        state.messages.append({
            "type": "system",
            "content": question["question_text"]
        })

def render_input_area(state: AppState):
    """Render input area based on current question type"""
    if state.current_question_index >= len(state.questions):
        return
        
    question = state.questions[state.current_question_index]
    
    with me.box(style=me.Style(
        background="white",
        padding=me.Padding.all(16),
        border_top="1px solid #e0e0e0"
    )):
        
        if question["question_type"] == "text":
            render_text_input(state, question)
        elif question["question_type"] == "multiple_choice":
            render_multiple_choice(state, question)
        elif question["question_type"] == "yes_no":
            render_yes_no_input(state, question)
        elif question["question_type"] == "scale":
            render_scale_input(state, question)
        elif question["question_type"] == "multiple_select":
            render_multiple_select(state, question)
        elif question["question_type"] == "file_upload":
            render_file_upload(state, question)

def render_text_input(state: AppState, question: Dict):
    """Render text input field"""
    me.input(
        placeholder="Type your answer here...",
        on_blur=lambda e: handle_text_response(e, state, question),
        style=me.Style(width="100%", margin=me.Margin(bottom=8))
    )

def render_multiple_choice(state: AppState, question: Dict):
    """Render multiple choice buttons"""
    with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=8)):
        for option in question.get("options", []):
            me.button(
                option,
                on_click=lambda e, opt=option: handle_choice_response(opt, state, question),
                style=me.Style(
                    background="#e3f2fd",
                    border="1px solid #2196f3",
                    border_radius=8,
                    padding=me.Padding.all(8),
                    margin=me.Margin(bottom=4)
                )
            )

def render_yes_no_input(state: AppState, question: Dict):
    """Render Yes/No buttons"""
    with me.box(style=me.Style(display="flex", gap=16)):
        me.button(
            "Yes",
            on_click=lambda e: handle_choice_response("Yes", state, question),
            style=me.Style(
                background="#4caf50",
                color="white",
                border="none",
                border_radius=8,
                padding=me.Padding.all(12)
            )
        )
        me.button(
            "No", 
            on_click=lambda e: handle_choice_response("No", state, question),
            style=me.Style(
                background="#f44336",
                color="white", 
                border="none",
                border_radius=8,
                padding=me.Padding.all(12)
            )
        )

def render_scale_input(state: AppState, question: Dict):
    """Render scale (1-10) input"""
    with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=4)):
        for i in range(1, 11):
            me.button(
                str(i),
                on_click=lambda e, num=i: handle_choice_response(str(num), state, question),
                style=me.Style(
                    background="#ffc107",
                    border="1px solid #ff9800",
                    border_radius=50,
                    width=40,
                    height=40,
                    text_align="center"
                )
            )

def render_multiple_select(state: AppState, question: Dict):
    """Render multiple select checkboxes"""
    me.text("Select all that apply:", style=me.Style(margin=me.Margin(bottom=8)))
    
    # For simplicity in MVP, use buttons that toggle selection
    selected_options = []
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=4)):
        for option in question.get("options", []):
            me.button(
                f"☐ {option}",
                on_click=lambda e, opt=option: toggle_option(opt, selected_options),
                style=me.Style(
                    background="#e8f5e8",
                    border="1px solid #4caf50",
                    text_align="left",
                    padding=me.Padding.all(8)
                )
            )
        
        # Submit selected options
        me.button(
            "Submit Selection",
            on_click=lambda e: handle_choice_response(", ".join(selected_options), state, question),
            style=me.Style(
                background="#2196f3",
                color="white",
                border="none",
                border_radius=8,
                padding=me.Padding.all(12),
                margin=me.Margin(top=8)
            )
        )

def render_file_upload(state: AppState, question: Dict):
    """Render file upload interface"""
    me.text("Upload medical documents or X-rays (Optional):", 
            style=me.Style(margin=me.Margin(bottom=8)))
    
    me.uploader(
        accepted_file_types=["application/pdf", "image/jpeg", "image/png"],
        on_upload=lambda e: handle_file_upload(e, state, question),
        style=me.Style(
            border="2px dashed #ccc",
            border_radius=8,
            padding=me.Padding.all(20),
            text_align="center"
        )
    )
    
    me.button(
        "Skip Upload",
        on_click=lambda e: handle_choice_response("No files uploaded", state, question),
        style=me.Style(
            background="#9e9e9e",
            color="white",
            border="none",
            border_radius=8,
            padding=me.Padding.all(8),
            margin=me.Margin(top=8)
        )
    )

def handle_text_response(event, state: AppState, question: Dict):
    """Handle text input response"""
    if event.value.strip():
        process_response(state, question, event.value)

def handle_choice_response(choice: str, state: AppState, question: Dict):
    """Handle choice selection response"""
    process_response(state, question, choice)

def handle_file_upload(event, state: AppState, question: Dict):
    """Handle file upload"""
    # TODO: Implement actual file upload to API
    state.upload_status = f"Uploaded: {event.file.name}"
    process_response(state, question, f"File uploaded: {event.file.name}")

def process_response(state: AppState, question: Dict, response: str):
    """Process user response and move to next question"""
    # Add user message to chat
    state.messages.append({
        "type": "user",
        "content": response
    })
    
    # Save response
    state.responses.append({
        "question_id": question["id"],
        "response_text": response,
        "timestamp": "now"  # TODO: Use actual timestamp
    })
    
    # Move to next question
    state.current_question_index += 1
    
    if state.current_question_index < len(state.questions):
        # Ask next question
        ask_current_question(state)
    else:
        # All questions answered, start processing
        start_case_processing(state)

def start_case_processing(state: AppState):
    """Start AI processing of case report"""
    state.is_processing = True
    state.messages.append({
        "type": "system",
        "content": "Thank you for answering all questions. I'm now analyzing your case and will provide a detailed assessment shortly..."
    })
    
    # Submit responses to API
    try:
        if state.case_id:
            # Submit all responses
            requests.post(
                f"{API_BASE}/case-reports/{state.case_id}/responses",
                json=[{"question_id": r["question_id"], "response_text": r["response_text"]} for r in state.responses]
            )
            
            # Trigger processing
            requests.post(f"{API_BASE}/case-reports/{state.case_id}/process")
            
        state.messages.append({
            "type": "system", 
            "content": "Your case is being processed. You'll receive a detailed report soon with next steps and recommendations."
        })
        
    except Exception as e:
        state.messages.append({
            "type": "error",
            "content": "There was an error processing your case. Please contact emergency services if this is urgent."
        })

def render_processing_status(state: AppState):
    """Render processing status"""
    with me.box(style=me.Style(
        background="white",
        padding=me.Padding.all(16),
        border_top="1px solid #e0e0e0",
        text_align="center"
    )):
        me.text("🔄 Processing your case...", style=me.Style(font_size=16))
        me.text("Please wait while our AI analyzes your symptoms", 
                style=me.Style(color="#666", margin=me.Margin(top=4)))

def render_completion_status(state: AppState):
    """Render completion status"""
    with me.box(style=me.Style(
        background="white",
        padding=me.Padding.all(16),
        border_top="1px solid #e0e0e0",
        text_align="center"
    )):
        me.text("✅ Assessment Complete", style=me.Style(font_size=16, color="#4caf50"))
        
        if state.case_id:
            me.button(
                "Download PDF Report",
                on_click=lambda e: download_pdf_report(state),
                style=me.Style(
                    background="#2196f3",
                    color="white",
                    border="none",
                    border_radius=8,
                    padding=me.Padding.all(12),
                    margin=me.Margin(top=8)
                )
            )

def download_pdf_report(state: AppState):
    """Download PDF report"""
    try:
        response = requests.get(f"{API_BASE}/case-reports/{state.case_id}/pdf")
        data = response.json()
        # TODO: Handle PDF download
        state.messages.append({
            "type": "system",
            "content": f"PDF report ready for download: {data.get('pdf_url', 'Processing...')}"
        })
    except:
        state.messages.append({
            "type": "error",
            "content": "PDF report is still being generated. Please try again in a moment."
        })

def toggle_option(option: str, selected_list: List[str]):
    """Toggle option selection for multiple select"""
    if option in selected_list:
        selected_list.remove(option)
    else:
        selected_list.append(option)

if __name__ == "__main__":
    me.run()
```


### **⚙️ Celery Tasks (v1/tasks/process_case.py)**

```python
from celery import Celery
import os
from typing import Dict, Any
import requests
import json

# Initialize Celery
celery_app = Celery(
    "fairdoc_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
)

@celery_app.task
def process_case_report_task(case_id: int) -> Dict[str, Any]:
    """Process complete case report with AI analysis"""
    
    try:
        # 1. Get case report from database
        case_report = get_case_report_from_db(case_id)
        
        # 2. Generate embeddings for text content
        embeddings = generate_text_embeddings(case_report)
        
        # 3. Simple rule-based urgency scoring for MVP
        urgency_score, importance_score = calculate_urgency_scores(case_report)
        
        # 4. Generate AI assessment
        ai_assessment = generate_ai_assessment(case_report, urgency_score, importance_score)
        
        # 5. Update case report with AI results
        update_case_report_with_ai(case_id, ai_assessment)
        
        # 6. Generate PDF report
        pdf_url = generate_pdf_report_task.delay(case_id)
        
        # 7. Send notification
        send_patient_notification_task.delay(case_id)
        
        return {
            "status": "completed",
            "case_id": case_id,
            "urgency_score": urgency_score,
            "importance_score": importance_score
        }
        
    except Exception as e:
        return {
            "status": "error",
            "case_id": case_id,
            "error": str(e)
        }

@celery_app.task  
def generate_pdf_report_task(case_id: int) -> str:
    """Generate PDF report using ReportLab"""
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import tempfile
    
    try:
        # Get case report data
        case_report = get_case_report_from_db(case_id)
        
        # Create PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
            
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add content to PDF
        c.drawString(100, 750, f"NHS Fairdoc AI - Medical Triage Report")
        c.drawString(100, 730, f"Case ID: {case_id}")
        c.drawString(100, 710, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add case details
        y_position = 680
        for key, value in case_report.items():
            if y_position < 100:  # Start new page if needed
                c.showPage()
                y_position = 750
            c.drawString(100, y_position, f"{key}: {value}")
            y_position -= 20
            
        c.save()
        
        # Upload PDF to MinIO
        pdf_url = upload_pdf_to_minio(pdf_path, case_id)
        
        # Update case report with PDF URL
        update_case_pdf_url(case_id, pdf_url)
        
        return pdf_url
        
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")

@celery_app.task
def send_patient_notification_task(case_id: int):
    """Send notification to patient"""
    
    try:
        # For MVP, just log the notification
        print(f"NOTIFICATION: Case {case_id} processing completed")
        
        # TODO: Implement actual notification system
        # - Email notification
        # - SMS notification  
        # - WhatsApp notification
        
        return {"status": "notification_sent", "case_id": case_id}
        
    except Exception as e:
        return {"status": "notification_failed", "error": str(e)}

def get_case_report_from_db(case_id: int) -> Dict[str, Any]:
    """Get case report from database"""
    # TODO: Implement database query
    return {"case_id": case_id, "status": "mock_data"}

def generate_text_embeddings(case_report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate embeddings using Ollama mxbai-embed-large"""
    
    try:
        # Combine text fields for embedding
        text_content = f"""
        Chief complaint: {case_report.get('chief_complaint', '')}
        Symptoms: {case_report.get('symptoms', '')}
        Medical history: {case_report.get('medical_history', '')}
        """
        
        # Call Ollama API for embeddings
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "mxbai-embed-large",
                "prompt": text_content
            }
        )
        
        if response.status_code == 200:
            embeddings = response.json()["embedding"]
            return {"embeddings": embeddings, "text": text_content}
        else:
            return {"embeddings": [], "text": text_content}
            
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return {"embeddings": [], "text": ""}

def calculate_urgency_scores(case_report: Dict[str, Any]) -> tuple:
    """Calculate urgency and importance scores using simple rules"""
    
    urgency_score = 0.0
    importance_score = 0.0
    
    # Get symptoms and characteristics
    symptoms = case_report.get("symptoms", {})
    pain_severity = symptoms.get("pain_severity", 0)
    
    # Simple rule-based scoring for chest pain
    if pain_severity >= 8:
        urgency_score += 0.6
        importance_score += 0.7
    elif pain_severity >= 6:
        urgency_score += 0.4
        importance_score += 0.5
    elif pain_severity >= 4:
        urgency_score += 0.2
        importance_score += 0.3
    
    # Check for high-risk symptoms
    high_risk_symptoms = ["shortness_of_breath", "sweating", "nausea", "radiating_pain"]
    for symptom in high_risk_symptoms:
        if symptoms.get(symptom, False):
            urgency_score += 0.15
            importance_score += 0.1
    
    # Check medical history
    medical_history = case_report.get("medical_history", [])
    high_risk_conditions = ["heart_disease", "previous_heart_attack", "diabetes"]
    for condition in high_risk_conditions:
        if condition in medical_history:
            urgency_score += 0.1
            importance_score += 0.15
    
    # Clamp scores to [-1, 1] range
    urgency_score = max(-1, min(1, urgency_score))
    importance_score = max(-1, min(1, importance_score))
    
    return urgency_score, importance_score

def generate_ai_assessment(case_report: Dict[str, Any], urgency_score: float, importance_score: float) -> Dict[str, Any]:
    """Generate AI assessment using simple LLM call"""
    
    try:
        # For MVP, use simple Ollama call with Gemma
        prompt = f"""
        Based on this medical case, provide assessment:
        
        Case: {json.dumps(case_report, indent=2)}
        Urgency Score: {urgency_score}
        Importance Score: {importance_score}
        
        Please provide:
        1. Likely conditions (max 3)
        2. Recommended actions (max 3) 
        3. Brief reasoning
        4. Confidence level (0-1)
        
        Keep response concise and medical.
        """
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            ai_response = response.json()["response"]
            
            # Parse AI response into structured format
            return {
                "urgency_score": urgency_score,
                "importance_score": importance_score,
                "predicted_conditions": ["Possible cardiac event", "Respiratory distress", "Musculoskeletal pain"],
                "recommended_actions": ["Seek urgent medical attention", "Monitor symptoms", "Rest and avoid exertion"],
                "reasoning": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
                "confidence_level": 0.75
            }
        else:
            # Fallback assessment
            return create_fallback_assessment(urgency_score, importance_score)
            
    except Exception as e:
        print(f"AI assessment failed: {e}")
        return create_fallback_assessment(urgency_score, importance_score)

def create_fallback_assessment(urgency_score: float, importance_score: float) -> Dict[str, Any]:
    """Create fallback assessment when AI fails"""
    
    if urgency_score > 0.5:
        actions = ["Seek immediate medical attention", "Call emergency services if symptoms worsen"]
        conditions = ["Acute cardiac event", "Severe respiratory distress"]
    elif urgency_score > 0.0:
        actions = ["Contact GP within 24 hours", "Monitor symptoms closely"]
        conditions = ["Chest pain investigation needed", "Possible respiratory infection"]
    else:
        actions = ["Self-care measures", "Contact GP if symptoms persist"]
        conditions = ["Likely musculoskeletal", "Possible minor respiratory issue"]
    
    return {
        "urgency_score": urgency_score,
        "importance_score": importance_score,
        "predicted_conditions": conditions,
        "recommended_actions": actions,
        "reasoning": f"Rule-based assessment based on urgency score: {urgency_score:.2f}",
        "confidence_level": 0.6
    }

def update_case_report_with_ai(case_id: int, ai_assessment: Dict[str, Any]):
    """Update case report with AI assessment results"""
    # TODO: Implement database update
    print(f"Updating case {case_id} with AI assessment: {ai_assessment}")

def upload_pdf_to_minio(pdf_path: str, case_id: int) -> str:
    """Upload PDF to MinIO storage"""
    # TODO: Implement MinIO upload
    return f"http://localhost:9000/reports/case_{case_id}_report.pdf"

def update_case_pdf_url(case_id: int, pdf_url: str):
    """Update case report with PDF URL"""
    # TODO: Implement database update
    print(f"Case {case_id} PDF available at: {pdf_url}")
```


### **🚀 Quick Start Script**

```bash
#!/bin/bash
# setup_v0.sh - Quick setup script for 10-hour sprint

echo "🚀 Setting up Fairdoc AI v0 Prototype..."

# 1. Start databases with Docker
echo "📦 Starting databases..."
docker run -d --name fairdoc-postgres -e POSTGRES_DB=fairdoc_v0 -e POSTGRES_USER=fairdoc -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:13
docker run -d --name fairdoc-redis -p 6379:6379 redis:7-alpine
docker run -d --name fairdoc-minio -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"

# 2. Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis celery minio reportlab pillow opencv-python requests pydantic mesop

# 3. Set up Ollama models
echo "🤖 Setting up Ollama models..."
ollama pull gemma:2b
ollama pull mxbai-embed-large

# 4. Initialize database
echo "💾 Setting up database..."
python -c "
from v1.core.database import init_database
init_database()
print('Database initialized!')
"

# 5. Start Celery worker
echo "⚙️ Starting Celery worker..."
celery -A v1.tasks.process_case worker --loglevel=info --detach

echo "✅ Setup complete! Ready to start development."
echo ""
echo "Next steps:"
echo "1. Start FastAPI: python v1/app.py"
echo "2. Start Mesop frontend: python frontend/chat_app.py"
echo "3. Open browser to http://localhost:8000 for API"
echo "4. Open browser to http://localhost:32123 for chat interface"
```


---

## 🎯 **Success Criteria for 10-Hour Sprint**

### **✅ Minimum Viable Product (MVP)**

- [ ] Patient can chat through NICE protocol questions
- [ ] Basic file upload functionality works
- [ ] Case report gets created and stored
- [ ] Simple AI scoring (rule-based + basic LLM)
- [ ] PDF report generation
- [ ] End-to-end flow from chat → report


### **🔧 Technical Deliverables**

- [ ] Working FastAPI backend with core endpoints
- [ ] Functional Mesop chat interface
- [ ] PostgreSQL database with case reports
- [ ] Basic Celery background job processing
- [ ] MinIO file storage integration
- [ ] Simple PDF report generation


### **📊 Performance Targets**

- [ ] Handle 10+ concurrent users
- [ ] Case processing < 30 seconds
- [ ] File uploads < 5MB work reliably
- [ ] Chat interface responsive on mobile

---

## 🚨 **Risk Mitigation \& Fallbacks**

### **If AI Models Don't Work**

- Use rule-based scoring system
- Pre-programmed response templates
- Simple keyword matching for conditions


### **If Mesop is Too Complex**

- Fall back to Streamlit for UI
- Use simple HTML forms
- Basic REST API testing with Postman


### **If File Upload Fails**

- Skip file processing for MVP
- Use text-only assessment
- Implement file upload in post-MVP


### **If Time Runs Short**

- Focus on core chat flow first
- Use mock AI responses
- Skip PDF generation, use JSON responses

---

## 📝 **Next Phase (Post-MVP)**

- Real multimodal AI integration
- Advanced image processing with SAM2.1
- ChromaDB vector matching
- Emergency protocol integration
- Doctor network integration
- Real-time notifications
- Security \& authentication
- Production deployment

---

**🎯 Total Estimated Time: 10 hours**
**👥 Team Size: 1 developer**
**🎥 Demo Ready: End of Day 1**

<div style="text-align: center">⁂</div>

[^1]: 111.json

[^2]: 999.json

[^3]: ehr.json

[^4]: requirements.txt

[^5]: README.md

[^6]: https://en.wikipedia.org/wiki/Paris

[^7]: https://www.coe.int/en/web/interculturalcities/paris

[^8]: https://en.wikipedia.org/wiki/List_of_capitals_of_France

[^9]: https://www.britannica.com/place/France

[^10]: https://home.adelphi.edu/~ca19535/page 4.html

[^11]: https://testbook.com/question-answer/which-is-the-capital-city-of-france--61c5718c1415c5341398033a

[^12]: https://www.britannica.com/place/Paris

[^13]: https://multimedia.europarl.europa.eu/en/video/infoclip-european-union-capitals-paris-france_I199003

[^14]: https://www.cia-france.com/french-kids-teenage-courses/paris-school/visit-paris

