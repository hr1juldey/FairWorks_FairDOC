# Fairdoc AI Triage System v2 Backend Testing Plan

## Overview: File-by-File, Module-by-Module Testing Strategy

Given your architectural commitment to modularity (1 file, 1 purpose, ≤300 lines), we'll test each component in isolation before building up to integration testing.

## Testing Structure

```text

src/tests/
├── unit/                    # Individual component testing
│   ├── core/               # Core infrastructure tests
│   ├── models/             # Data model validation
│   ├── services/           # Business logic testing
│   ├── api/                # API endpoint testing
│   └── utils/              # Utility function testing
├── integration/            # Cross-component testing
│   ├── service_flows/      # Service interaction testing
│   ├── data_flows/         # Data pipeline testing
│   └── external_deps/      # External dependency testing
└── e2e/                    # End-to-end scenarios
    ├── user_scenarios/     # User journey testing
    └── medical_workflows/  # Medical triage workflows
```

## Phase 1: Unit Testing Plan (File-by-File)

### Week 1: Core Infrastructure Testing

#### Day 1: Core Configuration Testing

**Files to Test:**

- `src/app2/core/config_v2.py`
- `src/app2/core/database_v2.py`
- `src/app2/core/dependencies_v2.py`

**Test Categories:**

```python
# tests/unit/core/test_config_v2.py
class TestConfigV2:
    def test_environment_variable_loading(self):
        """Test configuration loads from environment correctly"""
        pass
    
    def test_default_values(self):
        """Test default configuration values are sensible"""
        pass
    
    def test_validation_rules(self):
        """Test configuration validation catches invalid values"""
        pass

# tests/unit/core/test_database_v2.py
class TestDatabaseV2:
    def test_connection_pool_creation(self):
        """Test database connection pool initializes"""
        pass
    
    def test_health_check(self):
        """Test database health check functionality"""
        pass
    
    def test_connection_retry_logic(self):
        """Test database reconnection on failure"""
        pass

# tests/unit/core/test_dependencies_v2.py
class TestDependenciesV2:
    def test_dependency_injection(self):
        """Test FastAPI dependency injection works"""
        pass
    
    def test_service_lifecycle(self):
        """Test service initialization and cleanup"""
        pass
```


#### Day 2: Data Models Testing

**Files to Test:**

- `src/app2/models/database/nice_protocols.py`
- `src/app2/models/database/conversation_state.py`
- `src/app2/models/database/gold_standards.py`

**Test Categories:**

```python
# tests/unit/models/database/test_nice_protocols.py
class TestNICEProtocols:
    def test_model_creation(self):
        """Test NICEProtocol model creation with valid data"""
        pass
    
    def test_field_validation(self):
        """Test model field validation rules"""
        pass
    
    def test_json_field_handling(self):
        """Test JSON field serialization/deserialization"""
        pass
    
    def test_seed_data_validity(self):
        """Test NICE_SEED_DATA is valid and complete"""
        pass

# tests/unit/models/database/test_conversation_state.py
class TestConversationState:
    def test_state_transitions(self):
        """Test valid conversation state transitions"""
        pass
    
    def test_context_versioning(self):
        """Test context versioning functionality"""
        pass
    
    def test_serialization(self):
        """Test conversation state serialization"""
        pass
```


#### Day 3: Schema Validation Testing

**Files to Test:**

- `src/app2/models/schemas/medical_triage.py`
- `src/app2/models/schemas/multiturn_chat.py`

**Test Categories:**

```python
# tests/unit/models/schemas/test_medical_triage.py
class TestMedicalTriageSchemas:
    def test_medical_outcome_enum(self):
        """Test MedicalOutcome enum values are complete"""
        pass
    
    def test_triage_request_validation(self):
        """Test medical triage request validation"""
        pass
    
    def test_response_schema_completeness(self):
        """Test response schemas include all required fields"""
        pass

# tests/unit/models/schemas/test_multiturn_chat.py
class TestMultiturnChatSchemas:
    def test_chat_request_validation(self):
        """Test multiturn chat request validation"""
        pass
    
    def test_conversation_id_format(self):
        """Test conversation ID format validation"""
        pass
    
    def test_stakeholder_type_validation(self):
        """Test stakeholder type validation"""
        pass
```


### Week 2: Service Layer Testing

#### Day 4-5: DSPy Services Testing

**Files to Test:**

- `src/app2/services/dspy/medical_agent.py`
- `src/app2/services/dspy/question_generator.py`
- `src/app2/services/dspy/evaluation_optimizer.py`

**Test Categories:**

```python
# tests/unit/services/dspy/test_medical_agent.py
class TestMedicalAgent:
    def test_agent_initialization(self):
        """Test DSPy medical agent initializes correctly"""
        pass
    
    def test_signature_validation(self):
        """Test MedicalTriageSignature is properly defined"""
        pass
    
    def test_process_turn_basic(self):
        """Test basic turn processing with valid input"""
        pass
    
    def test_process_turn_edge_cases(self):
        """Test turn processing with edge cases"""
        pass
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        pass
    
    def test_red_flag_detection(self):
        """Test red flag symptom detection"""
        pass

# tests/unit/services/dspy/test_question_generator.py
class TestQuestionGenerator:
    def test_question_generation_quality(self):
        """Test generated questions are medically appropriate"""
        pass
    
    def test_conversation_context_usage(self):
        """Test questions use conversation context appropriately"""
        pass
    
    def test_nice_protocol_integration(self):
        """Test questions align with NICE protocols"""
        pass

# tests/unit/services/dspy/test_evaluation_optimizer.py
class TestEvaluationOptimizer:
    def test_optimization_setup(self):
        """Test DSPy optimization configuration"""
        pass
    
    def test_gold_standard_loading(self):
        """Test gold standard examples load correctly"""
        pass
    
    def test_metric_calculation(self):
        """Test evaluation metrics calculation"""
        pass
```


#### Day 6-7: Context and Chat Services Testing

**Files to Test:**

- `src/app2/services/context/redis_queue.py`
- `src/app2/services/context/nice_lookup.py`
- `src/app2/services/chat/stakeholder_router.py`
- `src/app2/services/chat/emergency_handler.py`
- `src/app2/services/chat/persistence_handler.py`

**Test Categories:**

```python
# tests/unit/services/context/test_redis_queue.py
class TestConversationQueue:
    def test_redis_connection(self):
        """Test Redis connection establishment"""
        pass
    
    def test_conversation_creation(self):
        """Test new conversation creation"""
        pass
    
    def test_state_persistence(self):
        """Test conversation state persistence"""
        pass
    
    def test_conversation_updates(self):
        """Test conversation turn updates"""
        pass
    
    def test_expiration_handling(self):
        """Test conversation expiration logic"""
        pass

# tests/unit/services/context/test_nice_lookup.py
class TestNICELookupService:
    def test_protocol_search(self):
        """Test NICE protocol search functionality"""
        pass
    
    def test_symptom_matching(self):
        """Test symptom to protocol matching"""
        pass
    
    def test_relevance_scoring(self):
        """Test protocol relevance scoring"""
        pass

# tests/unit/services/chat/test_stakeholder_router.py
class TestStakeholderRouter:
    def test_route_generation(self):
        """Test message route generation"""
        pass
    
    def test_emergency_routing(self):
        """Test emergency case routing"""
        pass
    
    def test_stakeholder_validation(self):
        """Test stakeholder type validation"""
        pass
```


### Week 3: API and Utilities Testing

#### Day 8-9: API Endpoint Testing

**Files to Test:**

- `src/app2/api/v2/endpoints/multiturn_chat.py`
- `src/app2/api/v2/endpoints/admin_dashboard.py`
- `src/app2/api/v2/endpoints/evaluation_metrics.py`
- `src/app2/api/v2/router_v2.py`

**Test Categories:**

```python
# tests/unit/api/v2/endpoints/test_multiturn_chat.py
class TestMultiturnChatEndpoint:
    def test_chat_endpoint_basic(self):
        """Test basic chat endpoint functionality"""
        pass
    
    def test_new_conversation_creation(self):
        """Test new conversation initialization"""
        pass
    
    def test_existing_conversation_continuation(self):
        """Test continuing existing conversations"""
        pass
    
    def test_error_handling(self):
        """Test error handling and response codes"""
        pass
    
    def test_background_task_scheduling(self):
        """Test background task scheduling"""
        pass
    
    def test_response_schema_compliance(self):
        """Test response matches expected schema"""
        pass

# tests/unit/api/v2/endpoints/test_admin_dashboard.py
class TestAdminDashboard:
    def test_metrics_endpoint(self):
        """Test admin metrics endpoint"""
        pass
    
    def test_conversation_analytics(self):
        """Test conversation analytics"""
        pass
    
    def test_access_control(self):
        """Test admin access control"""
        pass
```


#### Day 10: Utilities Testing

**Files to Test:**

- `src/app2/utils/datetime_utils.py`
- `src/app2/utils/outcome_mapper.py`

**Test Categories:**

```python
# tests/unit/utils/test_datetime_utils.py
class TestDatetimeUtils:
    def test_timezone_handling(self):
        """Test timezone conversion utilities"""
        pass
    
    def test_timestamp_formatting(self):
        """Test timestamp formatting functions"""
        pass
    
    def test_duration_calculations(self):
        """Test conversation duration calculations"""
        pass

# tests/unit/utils/test_outcome_mapper.py
class TestOutcomeMapper:
    def test_medical_outcome_mapping(self):
        """Test medical outcome classification"""
        pass
    
    def test_confidence_threshold_handling(self):
        """Test confidence threshold logic"""
        pass
    
    def test_edge_case_outcomes(self):
        """Test edge case outcome handling"""
        pass
```


## Phase 2: Integration Testing Plan

### Week 4: Service Integration Testing

#### Day 11-12: Core Service Flows

**Test Categories:**

```python
# tests/integration/service_flows/test_medical_reasoning_flow.py
class TestMedicalReasoningFlow:
    def test_dspy_to_redis_integration(self):
        """Test DSPy agent output stored in Redis correctly"""
        pass
    
    def test_nice_lookup_integration(self):
        """Test NICE protocol lookup integrates with reasoning"""
        pass
    
    def test_conversation_state_updates(self):
        """Test conversation state updates after reasoning"""
        pass

# tests/integration/service_flows/test_safety_pipeline.py
class TestSafetyPipeline:
    def test_input_validation_flow(self):
        """Test input validation through safety checks"""
        pass
    
    def test_output_validation_flow(self):
        """Test output validation before user delivery"""
        pass
    
    def test_emergency_escalation_flow(self):
        """Test emergency case escalation pipeline"""
        pass
```


#### Day 13-14: Data Flow Testing

**Test Categories:**

```python
# tests/integration/data_flows/test_conversation_lifecycle.py
class TestConversationLifecycle:
    def test_full_conversation_flow(self):
        """Test complete conversation from creation to completion"""
        pass
    
    def test_context_preservation(self):
        """Test context preservation across conversation turns"""
        pass
    
    def test_state_transitions(self):
        """Test valid state transitions through conversation"""
        pass

# tests/integration/data_flows/test_medical_data_pipeline.py
class TestMedicalDataPipeline:
    def test_medical_input_processing(self):
        """Test medical data processing pipeline"""
        pass
    
    def test_evidence_bundle_generation(self):
        """Test medical evidence bundle creation"""
        pass
    
    def test_audit_trail_creation(self):
        """Test complete audit trail generation"""
        pass
```


### Week 5: External Dependencies Testing

#### Day 15-16: External Integration Testing

**Test Categories:**

```python
# tests/integration/external_deps/test_redis_integration.py
class TestRedisIntegration:
    def test_redis_failover(self):
        """Test system behavior when Redis fails"""
        pass
    
    def test_redis_performance_under_load(self):
        """Test Redis performance with high conversation volume"""
        pass
    
    def test_data_consistency(self):
        """Test data consistency in Redis operations"""
        pass

# tests/integration/external_deps/test_database_integration.py
class TestDatabaseIntegration:
    def test_postgresql_persistence(self):
        """Test PostgreSQL data persistence"""
        pass
    
    def test_transaction_handling(self):
        """Test database transaction management"""
        pass
    
    def test_connection_pool_behavior(self):
        """Test database connection pool under load"""
        pass
```


## Phase 3: End-to-End Testing Plan

### Week 6: User Scenario Testing

#### Day 17-18: Medical Workflow Testing

**Test Categories:**

```python
# tests/e2e/medical_workflows/test_headache_triage.py
class TestHeadacheTriage:
    def test_simple_headache_scenario(self):
        """Test simple headache triage workflow"""
        pass
    
    def test_emergency_headache_scenario(self):
        """Test emergency headache detection and routing"""
        pass
    
    def test_chronic_headache_scenario(self):
        """Test chronic headache management workflow"""
        pass

# tests/e2e/medical_workflows/test_chest_pain_triage.py
class TestChestPainTriage:
    def test_cardiac_emergency_detection(self):
        """Test cardiac emergency detection and escalation"""
        pass
    
    def test_non_cardiac_chest_pain(self):
        """Test non-cardiac chest pain assessment"""
        pass
    
    def test_ambiguous_chest_pain(self):
        """Test handling of ambiguous chest pain symptoms"""
        pass
```


#### Day 19-20: User Journey Testing

**Test Categories:**

```python
# tests/e2e/user_scenarios/test_complete_user_journeys.py
class TestCompleteUserJourneys:
    def test_new_user_onboarding(self):
        """Test new user first interaction"""
        pass
    
    def test_returning_user_experience(self):
        """Test returning user conversation continuity"""
        pass
    
    def test_multi_session_conversations(self):
        """Test conversations spanning multiple sessions"""
        pass
    
    def test_concurrent_user_handling(self):
        """Test system handling multiple concurrent users"""
        pass

# tests/e2e/user_scenarios/test_error_recovery.py
class TestErrorRecovery:
    def test_service_failure_recovery(self):
        """Test user experience during service failures"""
        pass
    
    def test_invalid_input_handling(self):
        """Test handling of invalid user inputs"""
        pass
    
    def test_session_timeout_recovery(self):
        """Test recovery from session timeouts"""
        pass
```


## Implementation Strategy

### Test Infrastructure Setup

```python
# conftest.py - Shared test fixtures
@pytest.fixture
async def test_app():
    """Create test FastAPI application"""
    pass

@pytest.fixture
async def test_redis():
    """Create test Redis connection"""
    pass

@pytest.fixture
async def test_db():
    """Create test database session"""
    pass

@pytest.fixture
def mock_dspy_agent():
    """Mock DSPy medical agent for testing"""
    pass

@pytest.fixture
def sample_medical_data():
    """Sample medical data for testing"""
    pass
```


### Test Execution Order

1. **Unit Tests**: Run for each file independently
2. **Integration Tests**: Run after all unit tests pass
3. **E2E Tests**: Run after integration tests pass
4. **Performance Tests**: Run on dedicated test environment
5. **Security Tests**: Run with specialized security testing tools

### Success Criteria

- **Unit Tests**: 90%+ code coverage, all tests pass
- **Integration Tests**: All service interactions work correctly
- **E2E Tests**: All user scenarios complete successfully
- **Performance Tests**: Response times < 3 seconds, handles 100+ concurrent users
- **Security Tests**: No critical vulnerabilities, all safety measures effective


### Test Automation

```yaml
# .github/workflows/test.yml
name: Fairdoc V2 Testing Pipeline
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/app2
  
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
      postgres:
        image: postgres:15
    steps:
      - name: Run integration tests
        run: pytest tests/integration/ -v
  
  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - name: Run E2E tests
        run: pytest tests/e2e/ -v
```

This comprehensive testing strategy ensures that each component works correctly in isolation before testing their interactions, ultimately validating the complete system with realistic medical scenarios. The modular approach aligns with your architectural principles and makes it easy to identify and fix issues at the appropriate level.

Are you ready to begin with Phase 1: Unit Testing? I can start by creating the specific test files for the core infrastructure components.

<div style="text-align: center">⁂</div>

[^1]: v0.2.6-Stable.md

[^2]: v0.2.6-Stable.md

[^3]: a-practical-guide-to-building-agents.pdf

[^4]: engineering_v2.md

[^5]: engineering.md

[^6]: solution.md

[^7]: Fairdoc-AI-Triage-System_-Detailed-47-Day_roadmap.md

[^8]: TODO.md

[^9]: brief_v2.md

[^10]: 30-Day-Implementation-Plan.md

