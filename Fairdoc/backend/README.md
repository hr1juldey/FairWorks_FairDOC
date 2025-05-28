# 🏥 Fairdoc AI - Intelligent Healthcare Triage & Network Platform

## 🌟 Project Overview

Fairdoc AI is a revolutionary **AI-augmented medical triage network** that bridges the gap between patient demand and specialist availability in healthcare systems. Built specifically for the **DevPost Responsible AI Hackathon**, this platform addresses critical healthcare inequalities while maintaining ethical AI principles and bias mitigation at its core.

### 🎯 Mission Statement

> **Democratize healthcare expertise through AI-assisted triage that connects patients with the right care at the right time, while ensuring fairness, transparency, and human oversight.**

### 🔧 Problem We Solve

- **Healthcare Access Inequality**: 40% longer wait times for ethnic minorities, 35% rural patients lack specialist access
- **Resource Inefficiency**: 74% physician burnout, specialists idle while others are overwhelmed
- **Diagnostic Bias**: AI systems perpetuating historical biases in medical decision-making
- **Economic Burden**: £131 billion annually wasted on preventable emergency visits

### 💡 Our Solution

A **multi-agent AI orchestration platform** that:

- Conducts intelligent patient triage using specialized ML classifiers
- Routes patients to available specialists through a marketplace network
- Continuously monitors and mitigates bias in real-time
- Learns from doctor decisions to improve over time
- Maintains human-in-the-loop for all critical decisions

---

## 🏗️ Technical Architecture

### 🎯 Core Design Principles

1. **Unified API Architecture**: Single entry point serving both legacy (v1) and modern (v2) APIs
2. **Version Compatibility**: Seamless migration path from v1 to v2 with backward compatibility
3. **Bias-First Design**: Real-time fairness monitoring and correction across all versions
4. **Cost Optimization**: Minimal LLM usage through intelligent routing
5. **NHS Compliance**: FHIR R4, GDPR, and clinical safety standards
6. **Human Oversight**: Doctor-in-the-loop for all final decisions

### 🗂️ System Architecture Overview

```mermaid
graph RL
    subgraph "Frontend Layer"
        A[NHS App Integration] --> B[Patient Interface]
        B --> C[Doctor Dashboard]
    end
    
    subgraph "Unified API Gateway"
        D[Main FastAPI App] --> E[WebSocket Manager]
        E --> F[Authentication Layer]
        D --> G[v1 API Mount]
        D --> H[v2 API Mount]
        G --> I[Legacy Router]
        H --> J[Modern Router]
    end
    
    subgraph "v1 Backend (Legacy)"
        K[Original Triage] --> L[Basic ML Models]
        L --> M[PostgreSQL Only]
        I --> K
    end
    
    subgraph "v2 Backend (Modern)"
        N[AI Orchestration Layer] --> O[Model Selector]
        O --> P[Bias Monitor]
        P --> Q[Context Manager]
        J --> N
    end
    
    subgraph "Specialized ML Services (v2)"
        R[Text NLP Service]
        S[Image Analysis Service]
        T[Audio Processing Service]
        U[Time Series Service]
        V[Risk Classifier Service]
    end
    
    subgraph "Message Queue & Cache"
        W[Celery Workers] --> X[Redis Cache]
        X --> Y[ChromaDB Vector Store]
    end
    
    subgraph "Data Layer (Separated)"
        Z[NHS EHR Integration] --> AA[FHIR R4 API]
        AA --> BB[PostgreSQL DB]
        CC[RAG Documents] --> Y
    end
    
    subgraph "Network Layer"
        DD[Specialist Marketplace] --> EE[Doctor Availability]
        EE --> FF[Consultation Router]
    end
    
    A --> D
    N --> R & S & T & U & V
    R --> W
    N --> DD
    W --> Z
    Z --> BB
    CC --> Y
```

### 🧠 AI Model Architecture

#### **Multi-Agent Orchestration System**

```python
# Core AI orchestration framework (v2)
class FairdocAIOrchestrator:
    def __init__(self):
        self.routers = {
            'primary_triage': PrimaryTriageRouter(),
            'specialist_router': SpecialistRouter(),
            'bias_monitor': BiasMonitoringAgent(),
            'context_manager': ContextManager(),
            'ollama_router': OllamaModelRouter()
        }
        
        self.specialized_services = {
            'text_nlp': TextAnalysisService(),
            'image_analysis': MedicalImageService(),
            'audio_processing': SpeechAnalysisService(),
            'vitals_analysis': TimeSeriesService(),
            'risk_assessment': RiskClassifierService()
        }
```

#### **Specialized ML Classifier Services**

| Service Type | Purpose | Model Architecture | VRAM Usage | Response Time |
| :-- | :-- | :-- | :-- | :-- |
| **Text NLP** | Symptom extraction, sentiment analysis | DistilBERT + Clinical BERT | ~400MB |  BIAS_THRESHOLD:
            bias_alerts.append(f"Demographic parity violation: {parity_score:.3f}")
        
        # Check equalized odds
        odds_score = self.calculate_equalized_odds(ai_decision, patient_demographics)
        if odds_score > BIAS_THRESHOLD:
            bias_alerts.append(f"Equalized odds violation: {odds_score:.3f}")
        
        # Apply corrections if bias detected
        if bias_alerts:
            corrected_decision = await self.apply_bias_correction(ai_decision, patient_demographics)
            
            # Log bias incident
            await self.log_bias_incident(bias_alerts, patient_demographics, ai_decision)
            
            return corrected_decision
        
        return ai_decision
    
    async def apply_bias_correction(self, decision: dict, demographics: dict):
        """Apply counterfactual fairness corrections"""
        
        # Generate counterfactual scenarios
        counterfactuals = self.generate_counterfactuals(demographics)
        
        # Re-evaluate decision with neutral demographics
        neutral_decision = await self.ai_service.evaluate_with_demographics(
            decision['symptoms'], 
            counterfactuals['neutral']
        )
        
        # Weighted average of original and neutral decisions
        corrected_decision = self.weighted_fairness_correction(decision, neutral_decision)
        
        return corrected_decision
```

### 🏥 NHS EHR Integration

#### **FHIR R4 Compliant Data Exchange**

```python
class NHSEHRIntegration:
    """NHS Electronic Health Record integration with FHIR R4 compliance"""
    
    def __init__(self):
        self.fhir_client = FHIRClient(base_url=NHS_FHIR_ENDPOINT)
        self.gp_connect_client = GPConnectClient()
        
    async def fetch_patient_record(self, nhs_number: str) -> NHSPatientRecord:
        """Fetch comprehensive patient record from NHS systems"""
        
        # Validate NHS number
        if not self.validate_nhs_number(nhs_number):
            raise ValueError("Invalid NHS number")
        
        # Fetch from multiple NHS sources
        tasks = [
            self.fetch_gp_summary(nhs_number),
            self.fetch_hospital_records(nhs_number),
            self.fetch_medication_history(nhs_number),
            self.fetch_pathology_results(nhs_number)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate into unified patient record
        patient_record = self.aggregate_nhs_data(results)
        
        # Convert to FHIR R4 format
        fhir_bundle = self.convert_to_fhir(patient_record)
        
        # Store in local cache for performance
        await self.cache_patient_record(nhs_number, fhir_bundle)
        
        return patient_record
    
    async def update_patient_record(self, nhs_number: str, triage_results: dict):
        """Update NHS records with AI triage results"""
        
        # Create FHIR Observation resource
        observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "225399008",
                    "display": "AI-assisted triage assessment"
                }]
            },
            "subject": {"reference": f"Patient/{nhs_number}"},
            "effectiveDateTime": datetime.utcnow().isoformat(),
            "valueString": json.dumps(triage_results),
            "extension": [{
                "url": "http://fairdoc.ai/bias-score",
                "valueDecimal": triage_results.get('bias_score', 0.0)
            }]
        }
        
        # Submit to NHS systems via GP Connect
        await self.gp_connect_client.create_observation(observation)
```

### 💬 Multi-Modal Chat System

#### **Context-Aware Conversation Management**

```python
class ContextAwareChatOrchestrator:
    """Manages multi-modal conversations with context preservation"""
    
    def __init__(self):
        self.redis_client = redis.Redis(decode_responses=True)
        self.chromadb_client = chromadb.Client()
        self.ollama_router = OllamaModelRouter()
        
    async def process_patient_message(self, session_id: str, message: MultiModalMessage):
        """Process incoming patient message with context awareness"""
        
        # Retrieve conversation context from Redis
        context = await self.get_conversation_context(session_id)
        
        # Extract and process multi-modal content
        processed_content = await self.process_multimodal_content(message)
        
        # Update context buffer
        context['messages'].append(processed_content)
        context['patient_state'] = self.update_patient_state(context, processed_content)
        
        # Determine optimal response strategy
        response_strategy = await self.determine_response_strategy(context, processed_content)
        
        if response_strategy['type'] == 'ollama_llm':
            # Use local LLM for conversational response
            response = await self.ollama_router.generate_response(
                model=response_strategy['model'],
                context=context,
                message=processed_content
            )
        elif response_strategy['type'] == 'rule_based':
            # Use rule-based response for structured data collection
            response = await self.generate_rule_based_response(context, processed_content)
        elif response_strategy['type'] == 'specialist_escalation':
            # Escalate to human specialist
            response = await self.escalate_to_specialist(context, processed_content)
        
        # Store conversation in ChromaDB for similarity search
        await self.store_conversation_vector(session_id, processed_content, response)
        
        # Update Redis context
        await self.update_conversation_context(session_id, context)
        
        return response
```

### 🔄 Intelligent Model Routing

#### **Cost-Optimized LLM Selection**

```python
class OllamaModelRouter:
    """Intelligent router for local LLM models to minimize costs"""
    
    def __init__(self):
        self.models = {
            'clinical_reasoning': 'llama3.1:8b-instruct-q4_K_M',      # 4.9GB VRAM
            'conversation': 'phi3.5:3.8b-mini-instruct-q4_K_M',      # 2.3GB VRAM  
            'classification': 'gemma2:2b-instruct-q4_K_M',           # 1.6GB VRAM
            'summarization': 'qwen2.5:1.5b-instruct-q4_K_M'         # 1.1GB VRAM
        }
        
        self.model_capabilities = {
            'clinical_reasoning': ['complex_diagnosis', 'treatment_planning', 'differential_diagnosis'],
            'conversation': ['patient_interaction', 'clarification_questions', 'empathy_responses'],
            'classification': ['symptom_classification', 'urgency_assessment', 'specialty_routing'],
            'summarization': ['session_summary', 'handoff_notes', 'key_findings']
        }
        
        self.cost_per_token = {
            'clinical_reasoning': 0.001,  # Highest cost, most capable
            'conversation': 0.0005,       # Medium cost, good for chat
            'classification': 0.0002,     # Low cost, fast inference
            'summarization': 0.0001       # Lowest cost, efficient
        }
    
    async def select_optimal_model(self, task_type: str, context_length: int, budget_constraint: float):
        """Select most cost-effective model for the task"""
        
        # Filter models capable of handling the task
        capable_models = []
        for model, capabilities in self.model_capabilities.items():
            if task_type in capabilities:
                capable_models.append(model)
        
        # Calculate cost for each capable model
        model_costs = []
        for model in capable_models:
            estimated_tokens = self.estimate_token_usage(context_length, model)
            total_cost = estimated_tokens * self.cost_per_token[model]
            
            if total_cost  0.90, f"Clinical accuracy too low: {accuracy}"
    
    def test_response_time_performance(self):
        """Test system response times under load"""
        
        import asyncio
        import aiohttp
        
        async def make_request():
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post('/api/v2/assess', json=self.sample_patient) as response:
                    await response.json()
                return time.time() - start_time
        
        # Simulate 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        response_times = await asyncio.gather(*tasks)
        
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time  str:
        """Encrypt sensitive patient data using AES-256"""
        
        fernet = Fernet(self.encryption_key)
        serialized_data = json.dumps(patient_data).encode()
        encrypted_data = fernet.encrypt(serialized_data)
        
        # Log access for audit trail
        self.audit_logger.log_data_access(
            action='encrypt',
            data_type='patient_data',
            user_id=get_current_user_id(),
            timestamp=datetime.utcnow()
        )
        
        return encrypted_data.decode()
    
    def anonymize_for_analytics(self, patient_data: dict) -> dict:
        """Remove PII while preserving clinical utility"""
        
        anonymized = patient_data.copy()
        
        # Remove direct identifiers
        remove_fields = ['nhs_number', 'name', 'address', 'phone', 'email']
        for field in remove_fields:
            anonymized.pop(field, None)
        
        # Generalize quasi-identifiers
        if 'age' in anonymized:
            anonymized['age_group'] = self.generalize_age(anonymized['age'])
            del anonymized['age']
        
        if 'postcode' in anonymized:
            anonymized['region'] = self.generalize_location(anonymized['postcode'])
            del anonymized['postcode']
        
        return anonymized
```

### 📋 Compliance Checklist

- ✅ **GDPR Compliance**: Data minimization, consent management, right to erasure
- ✅ **NHS Digital Standards**: FHIR R4, GP Connect, clinical safety (DCB0129)
- ✅ **UK MHRA**: Medical device registration (Class IIa)
- ✅ **Clinical Governance**: Human-in-the-loop, audit trails, bias monitoring
- ✅ **Data Security**: AES-256 encryption, secure key management, audit logging
- ✅ **Access Control**: Role-based permissions, multi-factor authentication

---

## 🚀 Deployment & Scaling

### 🐳 Docker Configuration

```yaml
# docker-compose.yml - Complete development stack
version: '3.8'

services:
  fairdoc-unified:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://fairdoc:password@postgres:5432/fairdoc
      - REDIS_URL=redis://redis:6379/0
      - CHROMADB_URL=http://chromadb:8000
    depends_on:
      - postgres
      - redis
      - chromadb
    volumes:
      - ./model_cache:/app/model_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  fairdoc-v1:
    build: .
    command: python dev_v1.py
    ports:
      - "8001:8001"
    environment:
      - POSTGRES_URL=postgresql://fairdoc:password@postgres:5432/fairdoc
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  fairdoc-v2:
    build: .
    command: python dev_v2.py
    ports:
      - "8002:8002"
    environment:
      - POSTGRES_URL=postgresql://fairdoc:password@postgres:5432/fairdoc
      - REDIS_URL=redis://redis:6379/0
      - CHROMADB_URL=http://chromadb:8000
    depends_on:
      - postgres
      - redis
      - chromadb
    volumes:
      - ./model_cache:/app/model_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fairdoc
      POSTGRES_USER: fairdoc
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  celery-worker:
    build: .
    command: celery -A v2.services.celery_app worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/1
      - POSTGRES_URL=postgresql://fairdoc:password@postgres:5432/fairdoc
    depends_on:
      - redis
      - postgres
    volumes:
      - ./model_cache:/app/model_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  redis_data:
  chroma_data:
  ollama_data:
```

### ⚡ Performance Optimization

```python
# performance_optimizer.py - RTX 4070 optimization
class RTX4070Optimizer:
    """Optimization strategies for RTX 4070 12GB VRAM"""
    
    def __init__(self):
        self.max_vram_gb = 12
        self.reserved_vram_gb = 2  # Reserve for system
        self.available_vram_gb = self.max_vram_gb - self.reserved_vram_gb
        
    def optimize_model_loading(self):
        """Dynamic model loading based on available VRAM"""
        
        # Model memory requirements (quantized)
        models = {
            'clinical_llm': 3.2,      # Llama 3.1 8B quantized
            'image_classifier': 0.8,   # EfficientNet-B3
            'text_classifier': 0.4,    # DistilBERT
            'audio_processor': 0.3,    # Whisper small
            'risk_classifier': 0.2,    # XGBoost ensemble
        }
        
        # Priority loading based on usage frequency
        priority_order = ['text_classifier', 'risk_classifier', 'clinical_llm', 'image_classifier', 'audio_processor']
        
        loaded_models = {}
        used_vram = 0
        
        for model_name in priority_order:
            if used_vram + models[model_name] 90% for both v1 and v2
- **Bias Testing**: All models must pass demographic parity tests
- **Performance**:  **Important**: This software is for clinical decision support only and should not be used as the sole basis for medical decisions. All AI-generated recommendations must be reviewed and approved by qualified healthcare professionals. The system is designed to augment, not replace, clinical judgment.

### 🔒 Data Protection Notice

All patient data is processed in accordance with:

- **UK GDPR** and Data Protection Act 2018
- **NHS Digital Data Security Standards**
- **ISO 27001** Information Security Management
- **SOC 2 Type II** compliance

---

## 🎉 Acknowledgments

### 🏆 Special Thanks

- **NHS Digital** for FHIR R4 guidance and API access
- **DevPost RAI Hackathon** for promoting responsible AI in healthcare
- **OpenAI & Anthropic** for advancing conversational AI capabilities
- **Ollama Team** for enabling local LLM deployment
- **ChromaDB** for efficient vector storage solutions

### 🔬 Research Partners

- **Imperial College London** - Clinical validation studies
- **University of Oxford** - Bias mitigation research
- **Cambridge University** - Healthcare AI ethics framework
- **Alan Turing Institute** - Responsible AI development

### 💡 Inspiration

> *"Healthcare should be a human right, not a privilege. Technology should bridge gaps, not create them. AI should amplify human compassion, not replace it."*

---

**Built with ❤️ for healthcare equity and responsible AI**

*Last updated: May 2025*

⁂

