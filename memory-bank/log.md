# Log File Content

```log
(fairdoc-ai-triage) riju279@dream-machine:~/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage$ uv run pytest src/tests/poc/ -vvv -s --log-cli-level=INFO --tb=short
=================================================================================================================================== test session starts ===================================================================================================================================
platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0 -- /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage
configfile: pyproject.toml
plugins: mock-3.14.1, asyncio-1.1.0, Faker-37.4.2, anyio-4.9.0, cov-6.2.1, xdist-3.8.0, langsmith-0.4.8
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... 
----------------------------------------------------------------------------------------------------------------------------------- live log collection -----------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: GET https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json "HTTP/1.1 200 OK"
2025-08-23 03:54:04 [info     ] Found 4 models from environment
2025-08-23 03:54:04 [debug    ] Discovered models from ollama: ['hf.co/unsloth/medgemma-4b-it-GGUF:Q8_K_XL', 'hf.co/unsloth/medgemma-4b-it-GGUF', 'qwen3:8b', 'qwen3', 'qwen2.5-coder:7b', 'qwen2.5-coder', 'gpt-oss:20b', 'gpt-oss', 'mistral:7b', 'mistral', 'deepseek-r1:8b', 'deepseek-r1', 'gemma3n:e4b', 'gemma3n']
2025-08-23 03:54:04 [info     ] Found 14 models from ollama   
2025-08-23 03:54:04 [debug    ] ✅ Created LLM instance: gemma3n
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with default: gemma3n
2025-08-23 03:54:04 [info     ] ✅ DSPy Provider initialized with 14 models
2025-08-23 03:54:04 [info     ] Default model: gemma3n        
2025-08-23 03:54:04 [debug    ] ✅ Created LLM instance: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:54:04 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:54:04 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
collecting 13 items                                                                                                                                                                                                                                                                       2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:54:04 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:54:04 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
collected 50 items                                                                                                                                                                                                                                                                        

src/tests/poc/test_conversation_orchestrator.py::test_conversation_initialization 2025-08-23 03:54:04 [info     ] 🤖 Initializing shared DSPy configuration for test session...
2025-08-23 03:54:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Shared DSPy configured. Default model: gemma3n
2025-08-23 03:54:04 [info     ] 🔥 Pre-warming LLM models for test session...
2025-08-23 03:54:04 [info     ] 🔥 Starting LLM warmup          method=rest_api model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] 🌐 Warming up via REST API     

------------------------------------------------------------------------------------------------------------------------------------- live log setup --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1740 HTTP Request: GET http://localhost:11434/api/tags "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1740 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2025-08-23 03:54:04 [info     ] ✅ LLM warmup completed successfully method=rest_api model=gemma3n:e4b time_seconds=0.23378682136535645
2025-08-23 03:54:04 [info     ] ✅ Model 'gemma3n:e4b' warmed up successfully in 0.23s
2025-08-23 03:54:04 [info     ] 🎯 1/1 models warmed up successfully
2025-08-23 03:54:04 [info     ] ✅ DSPy prepared for async context with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy prepared for async context with model: gemma3n:e4b
2025-08-23 03:54:04 [debug    ] NICE indices built             categories=12 symptoms=160
2025-08-23 03:54:04 [info     ] Enhanced NICE Lookup Service initialized protocol_count=120
2025-08-23 03:54:04 [info     ] Legacy NICE Lookup Service initialized with enhanced backend
2025-08-23 03:54:04 [info     ] ✅ DSPy prepared for async context with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy prepared for async context with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:54:04 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:54:04 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ DSPy prepared for async context with model: gemma3n:e4b
2025-08-23 03:54:04 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 03:54:04 [info     ] ✅ V2 Conversation Queue initialized
2025-08-23 03:54:04 [info     ] ✅ Chat Orchestrator ready     
2025-08-23 03:54:04 [info     ] 🩺 Processing conversation turn conversation_id=UUID('954960c9-459e-4734-8263-a14d04958290') stakeholder=patient user_id=test_patient_001
2025-08-23 03:54:04 [info     ] 🔄 Started new conversation     conversation_id=conv_test_patient_001_1755901444 user_id=test_patient_001
2025-08-23 03:54:04 [info     ] 🔄 Conversation ID overridden   new_id=954960c9-459e-4734-8263-a14d04958290 old_id=conv_test_patient_001_1755901444
2025-08-23 03:54:04 [info     ] Protocol matched               confidence=0.8 protocol_code=PE_EMERGENCY
-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:54:22 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
2025-08-23 03:54:22 [info     ] 📨 Message routed               conversation_id=954960c9-459e-4734-8263-a14d04958290 from_stakeholder=patient medical_outcome=emergency routes_count=2
2025-08-23 03:54:22 [info     ] ✅ Conversation turn processed  context_turns=1 conversation_id=954960c9-459e-4734-8263-a14d04958290 outcome=emergency turn=2

🩺 CONVERSATION 954960c9 - Turn 2
👤 Patient: I have severe chest pain
🤖 Agent: None
📊 Assessment: EMERGENCY (95% confident about: emergency classification (found 3 red flags), clinical reasoning accuracy)
🚩 Red Flags: Severe chest pain, Severe chest pain, Suspected Pulmonary Embolism (PE)
💭 Reasoning: 1. **Identify the primary concern:** Severe chest pain is a serious symptom requiring immediate eval...
✅ Conversation Complete: emergency
------------------------------------------------------------
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_context_continuity 2025-08-23 03:54:22 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:22 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:22 [warning  ] ⚠️ DSPy configuration returned False, but continuing with initialization
2025-08-23 03:54:22 [debug    ] NICE indices built             categories=12 symptoms=160
2025-08-23 03:54:22 [info     ] Enhanced NICE Lookup Service initialized protocol_count=120
2025-08-23 03:54:22 [info     ] Legacy NICE Lookup Service initialized with enhanced backend
2025-08-23 03:54:22 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:22 [error    ] ❌ Failed to configure DSPy via centralized provider
2025-08-23 03:54:22 [warning  ] ⚠️ Question generator initialization issue: DSPy configuration failed
2025-08-23 03:54:22 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:22 [error    ] ❌ Failed to initialize medical agent: Failed to configure DSPy
2025-08-23 03:54:22 [info     ] ✅ V2 Conversation Queue initialized
2025-08-23 03:54:22 [info     ] ✅ Chat Orchestrator ready     
2025-08-23 03:54:22 [info     ] 🩺 Processing conversation turn conversation_id=UUID('13bf4b6e-b888-4a33-8176-f9623474a8de') stakeholder=patient user_id=test_patient
2025-08-23 03:54:22 [error    ] ❌ Medical agent not initialized
2025-08-23 03:54:22 [info     ] 🩺 Processing conversation turn conversation_id=UUID('13bf4b6e-b888-4a33-8176-f9623474a8de') stakeholder=patient user_id=test_patient
2025-08-23 03:54:22 [error    ] ❌ Medical agent not initialized
FAILED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_stage_progression 2025-08-23 03:54:22 [info     ] ✅ DSPy configured with model: gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_urgency_escalation 2025-08-23 03:54:35 [info     ] ✅ DSPy configured with model: gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_optimization[mipro] 2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_optimization[copro] 2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_optimization[bootstrap] 2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_redis_state_management 2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [warning  ] ⚠️ DSPy configuration returned False, but continuing with initialization
2025-08-23 03:54:55 [debug    ] NICE indices built             categories=12 symptoms=160
2025-08-23 03:54:55 [info     ] Enhanced NICE Lookup Service initialized protocol_count=120
2025-08-23 03:54:55 [info     ] Legacy NICE Lookup Service initialized with enhanced backend
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy via centralized provider
2025-08-23 03:54:55 [warning  ] ⚠️ Question generator initialization issue: DSPy configuration failed
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to initialize medical agent: Failed to configure DSPy
2025-08-23 03:54:55 [info     ] ✅ V2 Conversation Queue initialized
2025-08-23 03:54:55 [info     ] ✅ Chat Orchestrator ready     
2025-08-23 03:54:55 [info     ] 🩺 Processing conversation turn conversation_id=UUID('e6d78c9a-7763-424f-9a10-d03899304b3f') stakeholder=patient user_id=test_patient
2025-08-23 03:54:55 [error    ] ❌ Medical agent not initialized
FAILED
src/tests/poc/test_conversation_orchestrator.py::test_context_engineering_versioning ✅ Context versioning immutability working correctly
PASSED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_completion_logic PASSED
src/tests/poc/test_conversation_orchestrator.py::test_stakeholder_routing 2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [warning  ] ⚠️ DSPy configuration returned False, but continuing with initialization
2025-08-23 03:54:55 [debug    ] NICE indices built             categories=12 symptoms=160
2025-08-23 03:54:55 [info     ] Enhanced NICE Lookup Service initialized protocol_count=120
2025-08-23 03:54:55 [info     ] Legacy NICE Lookup Service initialized with enhanced backend
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy via centralized provider
2025-08-23 03:54:55 [warning  ] ⚠️ Question generator initialization issue: DSPy configuration failed
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to initialize medical agent: Failed to configure DSPy
2025-08-23 03:54:55 [info     ] ✅ V2 Conversation Queue initialized
2025-08-23 03:54:55 [info     ] ✅ Chat Orchestrator ready     
2025-08-23 03:54:55 [info     ] 🩺 Processing conversation turn conversation_id=UUID('f316ce18-3821-40b1-a24d-71468fbc3570') stakeholder=patient user_id=patient_001
2025-08-23 03:54:55 [error    ] ❌ Medical agent not initialized
FAILED
src/tests/poc/test_conversation_orchestrator.py::test_emergency_escalation_workflow 2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [warning  ] ⚠️ DSPy configuration returned False, but continuing with initialization
2025-08-23 03:54:55 [debug    ] NICE indices built             categories=12 symptoms=160
2025-08-23 03:54:55 [info     ] Enhanced NICE Lookup Service initialized protocol_count=120
2025-08-23 03:54:55 [info     ] Legacy NICE Lookup Service initialized with enhanced backend
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy via centralized provider
2025-08-23 03:54:55 [warning  ] ⚠️ Question generator initialization issue: DSPy configuration failed
2025-08-23 03:54:55 [error    ] ❌ Failed to configure DSPy: dspy.settings.configure(...) can only be called from the same async task that called it first. Please use `dspy.context(...)` in other async tasks instead.
2025-08-23 03:54:55 [error    ] ❌ Failed to initialize medical agent: Failed to configure DSPy
2025-08-23 03:54:55 [info     ] ✅ V2 Conversation Queue initialized
2025-08-23 03:54:55 [info     ] ✅ Chat Orchestrator ready     
2025-08-23 03:54:55 [info     ] 🩺 Processing conversation turn conversation_id=UUID('e457bfff-9e04-4c2d-83b0-56e83dfff7e0') stakeholder=patient user_id=emergency_patient
2025-08-23 03:54:55 [error    ] ❌ Medical agent not initialized
FAILED
src/tests/poc/test_conversation_orchestrator.py::test_conversation_metrics_tracking PASSED
src/tests/poc/test_medical_agent_optimization.py::test_medical_agent_basic_triage 2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:55 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:54:55 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:54:55 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:54:55 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:54:55 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:54:55 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:55:07 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_medical_reasoning_separation 2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:55:07 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:55:07 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:55:07 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_emergency_detection_accuracy 2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:55:07 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:55:07 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:55:07 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:55:07 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:55:18 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:55:36 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:55:50 [info     ] 🔄 Medical turn processed       confidence=100 outcome=emergency thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:56:07 [info     ] 🔄 Medical turn processed       confidence=100 outcome=emergency thinking_captured=True turn=4
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_conversation_completion_logic 2025-08-23 03:56:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:56:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:56:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:56:07 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:56:07 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:56:07 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:56:07 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:56:07 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:56:07 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:56:18 [info     ] 🔄 Medical turn processed       confidence=100 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:04 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='mild headache from work' urgency_level=low
2025-08-23 03:57:04 [info     ] 🔄 Medical turn processed       confidence=95 outcome=inconclusive thinking_captured=True turn=2
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_model_evaluation_against_gold_standards 2025-08-23 03:57:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 03:57:04 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 03:57:04 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 03:57:04 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 03:57:04 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 03:57:04 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 03:57:04 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b

------------------------------------------------------------------------------------------------------------------------------------- live log setup --------------------------------------------------------------------------------------------------------------------------------------
2025-08-23 03:57:04 [info     ] 🧪 Starting DSPy module evaluation limit=10
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:04 [info     ] 🗄️ V2 Database engine created  pool_size=10
2025-08-23 03:57:04 [info     ] 📊 V2 Session factory created  
2025-08-23 03:57:04 [debug    ] 📊 V2 Database session created 
2025-08-23 03:57:04,901 INFO sqlalchemy.engine.Engine select pg_catalog.version()
INFO     sqlalchemy.engine.Engine:base.py:1842 select pg_catalog.version()
2025-08-23 03:57:04,901 INFO sqlalchemy.engine.Engine [raw sql] ()
INFO     sqlalchemy.engine.Engine:base.py:1842 [raw sql] ()
2025-08-23 03:57:04,929 INFO sqlalchemy.engine.Engine select current_schema()
INFO     sqlalchemy.engine.Engine:base.py:1842 select current_schema()
2025-08-23 03:57:04,930 INFO sqlalchemy.engine.Engine [raw sql] ()
INFO     sqlalchemy.engine.Engine:base.py:1842 [raw sql] ()
2025-08-23 03:57:04,933 INFO sqlalchemy.engine.Engine show standard_conforming_strings
INFO     sqlalchemy.engine.Engine:base.py:1842 show standard_conforming_strings
2025-08-23 03:57:04,933 INFO sqlalchemy.engine.Engine [raw sql] ()
INFO     sqlalchemy.engine.Engine:base.py:1842 [raw sql] ()
2025-08-23 03:57:04 [debug    ] 🔌 Database connection established
2025-08-23 03:57:04 [debug    ] 📤 Database connection checked out from pool
2025-08-23 03:57:04,935 INFO sqlalchemy.engine.Engine BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:2698 BEGIN (implicit)
2025-08-23 03:57:04,942 INFO sqlalchemy.engine.Engine SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
INFO     sqlalchemy.engine.Engine:base.py:1842 SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
2025-08-23 03:57:04,942 INFO sqlalchemy.engine.Engine [generated in 0.00023s] (50,)
INFO     sqlalchemy.engine.Engine:base.py:1842 [generated in 0.00023s] (50,)
2025-08-23 03:57:05,004 INFO sqlalchemy.engine.Engine ROLLBACK
INFO     sqlalchemy.engine.Engine:base.py:2701 ROLLBACK
2025-08-23 03:57:05 [debug    ] 🔒 V2 Database session closed  
2025-08-23 03:57:05 [info     ] 📋 Loaded evaluation examples   count=8
  0%|                                                                                                                                                                                                                                                                | 0/8 [00:00<?, ?it/s]2025-08-23 03:57:05 [info     ] 🔄 Conversation reset          
2025-08-23 03:57:05 [info     ] 🔄 Conversation reset          
2025-08-23 03:57:05 [info     ] 🔄 Conversation reset          
2025-08-23 03:57:05 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:29 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:39 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:45 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=4
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:57:53 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=4
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 1 (0.0%):  12%|██████████████████████████▉                                                                                                                                                                                            | 1/8 [01:10<08:15, 70.78s/it]2025-08-23 03:58:15 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:58:23 [info     ] 🔄 Medical turn processed       confidence=85 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 2 (0.0%):  25%|█████████████████████████████████████████████████████▊                                                                                                                                                                 | 2/8 [01:25<03:48, 38.10s/it]2025-08-23 03:58:31 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:58:39 [info     ] 🔄 Medical turn processed       confidence=100 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:59:02 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 3 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [02:01<03:04, 36.86s/it]2025-08-23 03:59:06 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:59:12 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 4 (0.0%):  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 4/8 [02:11<01:45, 26.45s/it]2025-08-23 03:59:16 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 5 (0.0%):  62%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                | 5/8 [02:24<01:03, 21.29s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 6 (0.0%):  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                     | 6/8 [02:31<00:33, 16.60s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 03:59:44 [info     ] 🔄 Medical turn processed       confidence=90 outcome=routine thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:00:05 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:00:24 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='Buy herbal Viagra at 90% discount!!!' urgency_level=low
2025-08-23 04:00:24 [info     ] 🔄 Medical turn processed       confidence=100 outcome=inconclusive thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 7 (0.0%):  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 7/8 [03:21<00:27, 27.65s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 8 (0.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:28<00:00, 26.04s/it]
2025/08/23 04:00:33 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 8 (0.0%)
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:00:33 [info     ] ✅ DSPy evaluation completed    accuracy=EvaluationResult(score=0.0, results=<list of 8 results>) examples=8
FAILED
------------------------------------------------------------------------------------------------------------------------------------ live log teardown ------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"

src/tests/poc/test_medical_agent_optimization.py::test_dspy_optimizer_strategies[bootstrap] 2025-08-23 04:00:33 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:00:33 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 04:00:33 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 04:00:33 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 04:00:33 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 04:00:33 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:00:33 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:00:33 [info     ] ⚙️ Starting DSPy optimization  iterations=1 optimizer=bootstrap
2025-08-23 04:00:33 [debug    ] 📊 V2 Database session created 

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
ERROR    sqlalchemy.pool.impl.AsyncAdaptedQueuePool:base.py:378 Exception terminating connection <AdaptedConnection <asyncpg.connection.Connection object at 0x73fc80125c60>>
Traceback (most recent call last):
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/pool/base.py", line 374, in _close_connection
    self._dialect.do_terminate(connection)
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 1130, in do_terminate
    dbapi_connection.terminate()
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 907, in terminate
    self.await_(asyncio.shield(self._connection.close(timeout=2)))
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 132, in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 196, in greenlet_spawn
    value = await result
            ^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1504, in close
    await self._protocol.close(timeout)
  File "asyncpg/protocol/protocol.pyx", line 627, in close
  File "asyncpg/protocol/protocol.pyx", line 660, in asyncpg.protocol.protocol.BaseProtocol._request_cancel
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1673, in _cancel_current_command
    self._cancellations.add(self._loop.create_task(self._cancel(waiter)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 435, in create_task
    self._check_closed()
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 520, in _check_closed
    raise RuntimeError('Event loop is closed')
RuntimeError: Event loop is closed
2025-08-23 04:00:33 [error    ] ❌ V2 Database session error    error="Task <Task pending name='Task-74' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:00:33 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:00:33 [warning  ] ⚠️ Database load failed, using seed data error="Task <Task pending name='Task-74' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
2025-08-23 04:00:33 [info     ] 📋 Loaded evaluation examples   count=8
WARNING  root:base_module.py:134 Failed to deep copy attribute 'medical_agent' of EvaluationProgram, falling back to shallow copy or reference copy.
WARNING  root:base_module.py:134 Failed to deep copy attribute 'medical_agent' of EvaluationProgram, falling back to shallow copy or reference copy.
WARNING  root:base_module.py:134 Failed to deep copy attribute 'medical_agent' of EvaluationProgram, falling back to shallow copy or reference copy.
WARNING  root:base_module.py:134 Failed to deep copy attribute 'medical_agent' of EvaluationProgram, falling back to shallow copy or reference copy.
  0%|                                                                                                                                                                                                                                                                | 0/8 [00:00<?, ?it/s]2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Emergency Chest Pain - STEMI', 'description': '58-year-old male with classic ST-elevation MI symptoms requiring immediate PCI', 'primary_symptom': 'chest_pain', 'expected_outcome': 'emergency_route_to_doctor', 'patient_age': 58, 'patient_gender': 'male', 'expected_red_flags': ['crushing_chest_pain', 'left_arm_radiation', 'severe_sweating'], 'should_escalate': True, 'conversation_dialogue': [{'turn': 1, 'user_message': 'I have severe crushing chest pain going to my left arm', 'agent_question': 'When did this start and are you experiencing sweating or nausea?', 'expected_classification': 'inconclusive', 'red_flags_detected': ['crushing_chest_pain', 'left_arm_radiation'], 'confidence_score': 65}, {'turn': 2, 'user_message': "Started 45 minutes ago, I'm sweating heavily and feel very nauseous", 'agent_question': 'EMERGENCY - Call 108 immediately. You need emergency medical attention now.', 'expected_classification': 'emergency', 'red_flags_detected': ['crushing_chest_pain', 'left_arm_radiation', 'severe_sweating'], 'confidence_score': 95}], 'relevant_protocols': ['CG95_CHEST_PAIN', 'NG136_MI'], 'minimum_confidence_threshold': 90.0, 'expected_turn_count': 2, 'max_acceptable_turns': 3, 'created_by': 'dr_smith_cardiologist', 'clinical_notes': 'Classic STEMI presentation requiring immediate PCI'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Routine Headache Assessment', 'description': '32-year-old female with tension headache, suitable for self-care', 'primary_symptom': 'headache', 'expected_outcome': 'self_care_advice', 'patient_age': 32, 'patient_gender': 'female', 'expected_red_flags': [], 'should_escalate': False, 'conversation_dialogue': [{'turn': 1, 'user_message': 'I have a headache that started this morning', 'agent_question': 'Can you describe the pain - is it throbbing, tight, or sharp?', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 40}, {'turn': 2, 'user_message': "It's like a tight band around my head, not too severe", 'agent_question': 'Any nausea, vision changes, or neck stiffness?', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 50}, {'turn': 3, 'user_message': 'No, just the tight feeling. I had similar headaches before', 'agent_question': 'COMPLETE - This sounds like tension headache. Rest, hydration, and paracetamol should help.', 'expected_classification': 'self_care', 'red_flags_detected': [], 'confidence_score': 85}], 'relevant_protocols': ['NG127_HEADACHE'], 'minimum_confidence_threshold': 75.0, 'expected_turn_count': 3, 'max_acceptable_turns': 4, 'created_by': 'dr_jones_gp', 'clinical_notes': 'Typical tension headache pattern'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Emergency Respiratory Distress', 'description': '65-year-old female with severe asthma exacerbation requiring immediate treatment', 'primary_symptom': 'breathing_difficulty', 'expected_outcome': 'emergency_route_to_doctor', 'patient_age': 65, 'patient_gender': 'female', 'expected_red_flags': ['severe_dyspnea', 'silent_chest', 'exhaustion'], 'should_escalate': True, 'conversation_dialogue': [{'turn': 1, 'user_message': "I can't breathe properly, struggling to speak", 'agent_question': 'When did this breathing difficulty start and do you have asthma?', 'expected_classification': 'inconclusive', 'red_flags_detected': ['severe_dyspnea'], 'confidence_score': 70}, {'turn': 2, 'user_message': 'Yes asthma, started 2 hours ago, getting worse, used inhaler many times', 'agent_question': 'EMERGENCY - Call 108 now. This sounds like a severe asthma attack.', 'expected_classification': 'emergency', 'red_flags_detected': ['severe_dyspnea', 'exhaustion'], 'confidence_score': 92}], 'relevant_protocols': ['NG80_ASTHMA_EXAC'], 'minimum_confidence_threshold': 85.0, 'expected_turn_count': 2, 'max_acceptable_turns': 2, 'created_by': 'dr_respiratory_expert', 'clinical_notes': 'Severe asthma exacerbation with poor response to bronchodilators'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Case 4 :  Gradual Onset Jaw Pain – Potential Angina', 'description': '45-year-old male security guard with exertional jaw tightness, eventually found to have coronary artery disease', 'primary_symptom': 'jaw_pain', 'expected_outcome': 'routine_doctor_consultation', 'patient_age': 45, 'patient_gender': 'male', 'expected_red_flags': ['exertional_chest_equivalent'], 'should_escalate': False, 'conversation_dialogue': [{'turn': 1, 'user_message': 'Recently I notice a dull ache in my lower jaw when I climb office stairs', 'agent_question': 'Does the pain appear with exertion and ease with rest?', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 55}, {'turn': 2, 'user_message': 'Yes, it goes away within minutes of sitting down', 'agent_question': 'This could be a heart-related chest-pain equivalent. I recommend routine cardiology review and a treadmill test.', 'expected_classification': 'routine_doctor', 'red_flags_detected': ['exertional_chest_equivalent'], 'confidence_score': 88}], 'relevant_protocols': ['CG95_CHEST_PAIN'], 'minimum_confidence_threshold': 80.0, 'expected_turn_count': 2, 'max_acceptable_turns': 3, 'created_by': 'dr_cardio_india', 'clinical_notes': 'Jaw discomfort precipitated by exertion flagged as possible stable angina.'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
 50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                            | 4/8 [00:00<00:00, 34.18it/s]2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Case 5 :  Fever With Retro-Orbital Pain – Suspected Dengue', 'description': '27-year-old female software engineer in Bengaluru during monsoon, mild warning signs noted', 'primary_symptom': 'high_fever', 'expected_outcome': 'need_more_questions', 'patient_age': 27, 'patient_gender': 'female', 'expected_red_flags': [], 'should_escalate': False, 'conversation_dialogue': [{'turn': 1, 'user_message': 'I have 102-degree fever and pain behind my eyes since yesterday', 'agent_question': 'Any rash, abdominal pain, or bleeding? Have you checked platelet count?', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 60}, {'turn': 2, 'user_message': 'No bleeding but slight nausea, didn’t test platelets yet', 'agent_question': 'Dengue is common now – please visit lab for CBC and dengue NS1 test so we can decide next steps.', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 78}], 'relevant_protocols': ['MOHFW_DENGUE_2024'], 'minimum_confidence_threshold': 80.0, 'expected_turn_count': 2, 'max_acceptable_turns': 4, 'created_by': 'dr_public_health', 'clinical_notes': 'Need labs to risk-stratify possible dengue without current red-flag leakage signs.'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Case 6 :  Sudden One-Sided Weakness – Hyperacute Stroke', 'description': '62-year-old female with slurred speech within 30 minutes – eligible for thrombolysis', 'primary_symptom': 'weakness', 'expected_outcome': 'emergency_route_to_doctor', 'patient_age': 62, 'patient_gender': 'female', 'expected_red_flags': ['FAST_positive', 'slurred_speech', 'arm_drift'], 'should_escalate': True, 'conversation_dialogue': [{'turn': 1, 'user_message': 'My left arm feels heavy and words are jumbled', 'agent_question': 'TIME-CRITICAL – Call 108 and get to nearest stroke-ready hospital. When did this start?', 'expected_classification': 'emergency', 'red_flags_detected': ['FAST_positive', 'slurred_speech'], 'confidence_score': 97}], 'relevant_protocols': ['NICE_CVA_2024'], 'minimum_confidence_threshold': 90.0, 'expected_turn_count': 1, 'max_acceptable_turns': 1, 'created_by': 'dr_neuro_telestroke', 'clinical_notes': 'Meets FAST criteria; window <60 min – IV alteplase candidate.'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Case 7 :  Itchy Red Rash After New Antibiotic – Mild Allergic Reaction', 'description': '23-year-old male developed urticarial rash after first dose of amoxicillin', 'primary_symptom': 'rash', 'expected_outcome': 'self_care_advice', 'patient_age': 23, 'patient_gender': 'male', 'expected_red_flags': [], 'should_escalate': False, 'conversation_dialogue': [{'turn': 1, 'user_message': 'I’ve got itchy wheals on arms after starting an antibiotic', 'agent_question': 'Any breathing difficulty, lip swelling, or dizziness?', 'expected_classification': 'inconclusive', 'red_flags_detected': [], 'confidence_score': 50}, {'turn': 2, 'user_message': 'No, just hives', 'agent_question': 'Stop the antibiotic, take cetirizine 10 mg, observe. See GP if rash worsens.', 'expected_classification': 'self_care', 'red_flags_detected': [], 'confidence_score': 85}], 'relevant_protocols': ['CG183_DRUG_ALLERGY'], 'minimum_confidence_threshold': 80.0, 'expected_turn_count': 2, 'max_acceptable_turns': 3, 'created_by': 'dr_derm_gp', 'clinical_notes': 'Simple urticaria without anaphylaxis features.'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
2025/08/23 04:00:33 ERROR dspy.teleprompt.bootstrap: Failed to run or to evaluate example Example({'gold_standard': {'title': 'Case 8 :  Product Promotion Message', 'description': 'Spam content unrelated to health', 'primary_symptom': 'advertisement', 'expected_outcome': 'spam_or_irrelevant', 'patient_age': 0, 'patient_gender': 'unspecified', 'expected_red_flags': [], 'should_escalate': False, 'conversation_dialogue': [{'turn': 1, 'user_message': 'Buy herbal Viagra at 90% discount!!!', 'agent_question': 'This channel is for medical queries only. Promotional content is not accepted.', 'expected_classification': 'spam', 'red_flags_detected': [], 'confidence_score': 30}], 'relevant_protocols': [], 'minimum_confidence_threshold': 30.0, 'expected_turn_count': 1, 'max_acceptable_turns': 1, 'created_by': 'system_filter', 'clinical_notes': 'Filtered marketing spam.'}}) (input_keys={'gold_standard'}) with <bound method OptimizationProgram._medical_accuracy_metric of evaluation_program.accuracy_module.accuracy_evaluator.predict = Predict(StringSignature(predicted_outcome, expected_outcome, confidence_score -> reasoning, is_correct, accuracy_reasoning
    instructions='Evaluate medical accuracy against gold standards'
    predicted_outcome = Field(annotation=str required=True json_schema_extra={'desc': "Model's predicted medical outcome", '__dspy_field_type': 'input', 'prefix': 'Predicted Outcome:'})
    expected_outcome = Field(annotation=str required=True json_schema_extra={'desc': 'Gold standard expected outcome', '__dspy_field_type': 'input', 'prefix': 'Expected Outcome:'})
    confidence_score = Field(annotation=int required=True json_schema_extra={'desc': "Model's confidence score", '__dspy_field_type': 'input', 'prefix': 'Confidence Score:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    is_correct = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether prediction matches gold standard', '__dspy_field_type': 'output', 'prefix': 'Is Correct:'})
    accuracy_reasoning = Field(annotation=str required=True json_schema_extra={'desc': 'Reasoning for accuracy assessment', '__dspy_field_type': 'output', 'prefix': 'Accuracy Reasoning:'})
))
evaluation_program.accuracy_module.red_flag_evaluator.predict = Predict(StringSignature(detected_flags, expected_flags -> reasoning, detection_score, missed_critical
    instructions='Evaluate red flag detection capabilities'
    detected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Detected red flags by model', '__dspy_field_type': 'input', 'prefix': 'Detected Flags:'})
    expected_flags = Field(annotation=str required=True json_schema_extra={'desc': 'Expected red flags from gold standard', '__dspy_field_type': 'input', 'prefix': 'Expected Flags:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    detection_score = Field(annotation=float required=True json_schema_extra={'desc': 'Red flag detection accuracy 0-1', '__dspy_field_type': 'output', 'prefix': 'Detection Score:'})
    missed_critical = Field(annotation=bool required=True json_schema_extra={'desc': 'Whether critical flags were missed', '__dspy_field_type': 'output', 'prefix': 'Missed Critical:'})
))> due to asyncio.run() cannot be called from a running event loop.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 62.61it/s]
Bootstrapped 0 full traces after 7 examples for up to 1 rounds, amounting to 8 attempts.
2025-08-23 04:00:33 [info     ] 🧪 Starting DSPy module evaluation limit=30
2025-08-23 04:00:33 [debug    ] 📊 V2 Database session created 
2025-08-23 04:00:33 [debug    ] 🔌 Database connection established
2025-08-23 04:00:33 [debug    ] 📤 Database connection checked out from pool
2025-08-23 04:00:33,912 INFO sqlalchemy.engine.Engine BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:2698 BEGIN (implicit)
2025-08-23 04:00:33,913 INFO sqlalchemy.engine.Engine SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
INFO     sqlalchemy.engine.Engine:base.py:1842 SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
2025-08-23 04:00:33,913 INFO sqlalchemy.engine.Engine [cached since 209s ago] (50,)
INFO     sqlalchemy.engine.Engine:base.py:1842 [cached since 209s ago] (50,)
2025-08-23 04:00:33,916 INFO sqlalchemy.engine.Engine ROLLBACK
INFO     sqlalchemy.engine.Engine:base.py:2701 ROLLBACK
2025-08-23 04:00:33 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:00:33 [info     ] 📋 Loaded evaluation examples   count=8
2025-08-23 04:00:33 [info     ] 🔄 Conversation reset          
2025-08-23 04:00:33 [info     ] 🔄 Conversation reset          
  0%|                                                                                                                                                                                                                                                                | 0/8 [00:00<?, ?it/s]2025-08-23 04:00:33 [info     ] 🔄 Conversation reset          
2025-08-23 04:00:33 [info     ] 🔄 Conversation reset          
2025-08-23 04:00:33 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
2025-08-23 04:00:33 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=1
2025-08-23 04:00:33 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=2
2025-08-23 04:00:33 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=2
Average Metric: 0.00 / 1 (0.0%):   0%|                                                                                                                                                                                                                               | 0/8 [00:00<?, ?it/s]2025-08-23 04:00:33 [info     ] 🔄 Conversation reset          
2025-08-23 04:00:34 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
Average Metric: 0.00 / 2 (0.0%):  12%|██████████████████████████▉                                                                                                                                                                                            | 1/8 [00:00<00:00, 11.49it/s]2025-08-23 04:00:34 [info     ] 🔄 Conversation reset          
Average Metric: 0.00 / 3 (0.0%):  25%|█████████████████████████████████████████████████████▊                                                                                                                                                                 | 2/8 [00:00<00:00, 18.48it/s]2025-08-23 04:00:34 [info     ] 🔄 Conversation reset          
Average Metric: 0.00 / 3 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [00:00<00:00, 27.59it/s]2025-08-23 04:00:34 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
2025-08-23 04:00:34 [info     ] 🔄 Medical turn processed       confidence=90 outcome=routine thinking_captured=True turn=1
Average Metric: 0.00 / 4 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [00:00<00:00, 27.59it/s]2025-08-23 04:00:34 [info     ] 🔄 Conversation reset          
2025-08-23 04:00:34 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='Buy herbal Viagra at 90% discount!!!' urgency_level=low
2025-08-23 04:00:34 [info     ] 🔄 Medical turn processed       confidence=100 outcome=inconclusive thinking_captured=True turn=1
Average Metric: 0.00 / 5 (0.0%):  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 4/8 [00:00<00:00, 27.59it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 5 (0.0%):  62%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                | 5/8 [00:11<00:00, 27.59it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:01:06 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:01:13 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 6 (0.0%):  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                     | 6/8 [01:04<00:25, 12.58s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:01:43 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='Yes, it goes away within minutes of sitting down' urgency_level=low
2025-08-23 04:01:43 [info     ] 🔄 Medical turn processed       confidence=85 outcome=inconclusive thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:01:54 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 7 (0.0%):  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 7/8 [01:26<00:14, 14.71s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 8 (0.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:28<00:00, 11.03s/it]
2025/08/23 04:02:02 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 8 (0.0%)
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:02:02 [info     ] ✅ DSPy evaluation completed    accuracy=EvaluationResult(score=0.0, results=<list of 8 results>) examples=8
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_dspy_optimizer_strategies[mipro] 2025-08-23 04:02:02 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:02:02 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 04:02:02 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 04:02:02 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 04:02:02 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 04:02:02 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:02:02 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:02:02 [info     ] ⚙️ Starting DSPy optimization  iterations=1 optimizer=mipro
2025-08-23 04:02:02 [debug    ] 📊 V2 Database session created 

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
ERROR    sqlalchemy.pool.impl.AsyncAdaptedQueuePool:base.py:378 Exception terminating connection <AdaptedConnection <asyncpg.connection.Connection object at 0x73fc8005ff10>>
Traceback (most recent call last):
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/pool/base.py", line 374, in _close_connection
    self._dialect.do_terminate(connection)
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 1130, in do_terminate
    dbapi_connection.terminate()
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 907, in terminate
    self.await_(asyncio.shield(self._connection.close(timeout=2)))
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 132, in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 196, in greenlet_spawn
    value = await result
            ^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1504, in close
    await self._protocol.close(timeout)
  File "asyncpg/protocol/protocol.pyx", line 627, in close
  File "asyncpg/protocol/protocol.pyx", line 660, in asyncpg.protocol.protocol.BaseProtocol._request_cancel
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1673, in _cancel_current_command
    self._cancellations.add(self._loop.create_task(self._cancel(waiter)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 435, in create_task
    self._check_closed()
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 520, in _check_closed
    raise RuntimeError('Event loop is closed')
RuntimeError: Event loop is closed
2025-08-23 04:02:02 [error    ] ❌ V2 Database session error    error="Task <Task pending name='Task-116' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
2025-08-23 04:02:02 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:02:02 [warning  ] ⚠️ Database load failed, using seed data error="Task <Task pending name='Task-116' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
2025-08-23 04:02:02 [info     ] 📋 Loaded evaluation examples   count=8
2025-08-23 04:02:02 [error    ] Optimizer compilation failed for mipro: Minibatch size cannot exceed the size of the valset. Valset size: 6.
2025-08-23 04:02:02 [info     ] 🧪 Starting DSPy module evaluation limit=30
2025-08-23 04:02:02 [debug    ] 📊 V2 Database session created 
2025-08-23 04:02:02 [debug    ] 🔌 Database connection established
2025-08-23 04:02:02 [debug    ] 📤 Database connection checked out from pool
2025-08-23 04:02:02,239 INFO sqlalchemy.engine.Engine BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:2698 BEGIN (implicit)
2025-08-23 04:02:02,239 INFO sqlalchemy.engine.Engine SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
INFO     sqlalchemy.engine.Engine:base.py:1842 SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
2025-08-23 04:02:02,239 INFO sqlalchemy.engine.Engine [cached since 297.3s ago] (50,)
INFO     sqlalchemy.engine.Engine:base.py:1842 [cached since 297.3s ago] (50,)
2025-08-23 04:02:02,242 INFO sqlalchemy.engine.Engine ROLLBACK
INFO     sqlalchemy.engine.Engine:base.py:2701 ROLLBACK
2025-08-23 04:02:02 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:02:02 [info     ] 📋 Loaded evaluation examples   count=8
  0%|                                                                                                                                                                                                                                                                | 0/8 [00:00<?, ?it/s]2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=3
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 1 (0.0%):   0%|                                                                                                                                                                                                                               | 0/8 [00:00<?, ?it/s]2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
Average Metric: 0.00 / 2 (0.0%):  12%|██████████████████████████▉                                                                                                                                                                                            | 1/8 [00:00<00:00, 10.42it/s]2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
Average Metric: 0.00 / 3 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [00:00<00:00, 24.65it/s]2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=90 outcome=routine thinking_captured=True turn=1
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=2
Average Metric: 0.00 / 4 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [00:00<00:00, 24.65it/s]2025-08-23 04:02:02 [info     ] 🔄 Conversation reset          
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:02:02 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='Buy herbal Viagra at 90% discount!!!' urgency_level=low
2025-08-23 04:02:02 [info     ] 🔄 Medical turn processed       confidence=100 outcome=inconclusive thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 5 (0.0%):  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 4/8 [00:00<00:00, 24.65it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:02:11 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 5 (0.0%):  62%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                | 5/8 [00:12<00:00, 24.65it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:02:19 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 6 (0.0%):  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                     | 6/8 [00:30<00:12,  6.02s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:02:48 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:02:56 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 7 (0.0%):  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 7/8 [00:57<00:10, 10.57s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:03:13 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 8 (0.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:14<00:00, 12.01s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 8 (0.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:14<00:00,  9.27s/it]
2025/08/23 04:03:16 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 8 (0.0%)
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:03:16 [info     ] ✅ DSPy evaluation completed    accuracy=EvaluationResult(score=0.0, results=<list of 8 results>) examples=8
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_dspy_optimizer_strategies[copro] 2025-08-23 04:03:16 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:03:16 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 04:03:16 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 04:03:16 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 04:03:16 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b
2025-08-23 04:03:16 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:03:16 [info     ] 📊 DSPy Evaluation Optimizer initialized model=gemma3n:e4b
2025-08-23 04:03:16 [info     ] ⚙️ Starting DSPy optimization  iterations=1 optimizer=copro
2025-08-23 04:03:16 [debug    ] 📊 V2 Database session created 

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
ERROR    sqlalchemy.pool.impl.AsyncAdaptedQueuePool:base.py:378 Exception terminating connection <AdaptedConnection <asyncpg.connection.Connection object at 0x73fc802e88b0>>
Traceback (most recent call last):
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/pool/base.py", line 374, in _close_connection
    self._dialect.do_terminate(connection)
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 1130, in do_terminate
    dbapi_connection.terminate()
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 907, in terminate
    self.await_(asyncio.shield(self._connection.close(timeout=2)))
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 132, in await_only
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/sqlalchemy/util/_concurrency_py3k.py", line 196, in greenlet_spawn
    value = await result
            ^^^^^^^^^^^^
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1504, in close
    await self._protocol.close(timeout)
  File "asyncpg/protocol/protocol.pyx", line 627, in close
  File "asyncpg/protocol/protocol.pyx", line 660, in asyncpg.protocol.protocol.BaseProtocol._request_cancel
  File "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv/lib/python3.11/site-packages/asyncpg/connection.py", line 1673, in _cancel_current_command
    self._cancellations.add(self._loop.create_task(self._cancel(waiter)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 435, in create_task
    self._check_closed()
  File "/home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py", line 520, in _check_closed
    raise RuntimeError('Event loop is closed')
RuntimeError: Event loop is closed
2025-08-23 04:03:16 [error    ] ❌ V2 Database session error    error="Task <Task pending name='Task-158' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
2025-08-23 04:03:16 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:03:16 [warning  ] ⚠️ Database load failed, using seed data error="Task <Task pending name='Task-158' coro=<test_dspy_optimizer_strategies() running at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/src/tests/poc/test_medical_agent_optimization.py:134> cb=[_run_until_complete_cb() at /home/riju279/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/asyncio/base_events.py:181]> got Future <Future pending cb=[BaseProtocol._on_waiter_completed()]> attached to a different loop"
2025-08-23 04:03:16 [info     ] 📋 Loaded evaluation examples   count=8
WARNING  root:base_module.py:134 Failed to deep copy attribute 'medical_agent' of EvaluationProgram, falling back to shallow copy or reference copy.
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025/08/23 04:03:22 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.
2025-08-23 04:03:25 [error    ] Optimizer compilation failed for copro: litellm.UnsupportedParamsError: ollama does not support parameters: ['n'], for model=gemma3n:e4b. To drop these, set `litellm.drop_params=True` or for proxy:

`litellm_settings:
 drop_params: true`
. 
 If you want to use these params dynamically send allowed_openai_params=['n'] in your request.
2025-08-23 04:03:25 [info     ] 🧪 Starting DSPy module evaluation limit=30
2025-08-23 04:03:25 [debug    ] 📊 V2 Database session created 
2025-08-23 04:03:25 [debug    ] 🔌 Database connection established
2025-08-23 04:03:25 [debug    ] 📤 Database connection checked out from pool
2025-08-23 04:03:25,735 INFO sqlalchemy.engine.Engine BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:2698 BEGIN (implicit)
2025-08-23 04:03:25,736 INFO sqlalchemy.engine.Engine SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
INFO     sqlalchemy.engine.Engine:base.py:1842 SELECT gold_standard_dialogues_v2.standard_id, gold_standard_dialogues_v2.title, gold_standard_dialogues_v2.description, gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.expected_outcome, gold_standard_dialogues_v2.patient_age, gold_standard_dialogues_v2.patient_gender, gold_standard_dialogues_v2.expected_red_flags, gold_standard_dialogues_v2.should_escalate, gold_standard_dialogues_v2.conversation_dialogue, gold_standard_dialogues_v2.relevant_protocols, gold_standard_dialogues_v2.minimum_confidence_threshold, gold_standard_dialogues_v2.expected_turn_count, gold_standard_dialogues_v2.max_acceptable_turns, gold_standard_dialogues_v2.created_by, gold_standard_dialogues_v2.reviewed_by, gold_standard_dialogues_v2.clinical_notes, gold_standard_dialogues_v2.version, gold_standard_dialogues_v2.is_active, gold_standard_dialogues_v2.created_at, gold_standard_dialogues_v2.updated_at 
FROM gold_standard_dialogues_v2 
WHERE gold_standard_dialogues_v2.is_active ORDER BY gold_standard_dialogues_v2.primary_symptom, gold_standard_dialogues_v2.patient_age 
 LIMIT $1::INTEGER
2025-08-23 04:03:25,736 INFO sqlalchemy.engine.Engine [cached since 380.8s ago] (50,)
INFO     sqlalchemy.engine.Engine:base.py:1842 [cached since 380.8s ago] (50,)
2025-08-23 04:03:25,739 INFO sqlalchemy.engine.Engine ROLLBACK
INFO     sqlalchemy.engine.Engine:base.py:2701 ROLLBACK
2025-08-23 04:03:25 [debug    ] 🔒 V2 Database session closed  
2025-08-23 04:03:25 [info     ] 📋 Loaded evaluation examples   count=8
2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
  0%|                                                                                                                                                                                                                                                                | 0/8 [00:00<?, ?it/s]2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=3
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=3
Average Metric: 0.00 / 1 (0.0%):   0%|                                                                                                                                                                                                                               | 0/8 [00:00<?, ?it/s]2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
Average Metric: 0.00 / 2 (0.0%):  12%|██████████████████████████▉                                                                                                                                                                                            | 1/8 [00:00<00:00, 14.68it/s]2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
Average Metric: 0.00 / 3 (0.0%):  25%|█████████████████████████████████████████████████████▊                                                                                                                                                                 | 2/8 [00:00<00:00, 18.73it/s]2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
Average Metric: 0.00 / 4 (0.0%):  38%|████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 3/8 [00:00<00:00, 27.60it/s]2025-08-23 04:03:25 [info     ] 🔄 Conversation reset          
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=90 outcome=routine thinking_captured=True turn=1
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:03:25 [info     ] ❓ Questions generated          emergency_indicators=0 questions_count=2 symptom_keywords='Buy herbal Viagra at 90% discount!!!' urgency_level=low
2025-08-23 04:03:25 [info     ] 🔄 Medical turn processed       confidence=100 outcome=inconclusive thinking_captured=True turn=2
Average Metric: 0.00 / 6 (0.0%):  62%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                | 5/8 [00:00<00:00, 27.60it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:03:34 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=2
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:03:44 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=3
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 6 (0.0%):  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                     | 6/8 [00:19<00:00, 27.60it/s]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:04:03 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=4
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:04:09 [info     ] 🔄 Medical turn processed       confidence=75 outcome=routine thinking_captured=True turn=5
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 7 (0.0%):  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 7/8 [00:51<00:08,  8.51s/it]INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:04:29 [info     ] 🔄 Medical turn processed       confidence=85 outcome=routine thinking_captured=True turn=5
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
Average Metric: 0.00 / 8 (0.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:06<00:00,  8.33s/it]
2025/08/23 04:04:32 INFO dspy.evaluate.evaluate: Average Metric: 0.0 / 8 (0.0%)
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [error    ] ❌ Evaluation error             error='asyncio.run() cannot be called from a running event loop'
2025-08-23 04:04:32 [info     ] ✅ DSPy evaluation completed    accuracy=EvaluationResult(score=0.0, results=<list of 8 results>) examples=8
PASSED
src/tests/poc/test_medical_agent_optimization.py::test_medical_outcome_mapping PASSED
src/tests/poc/test_medical_agent_optimization.py::test_red_flag_detection_accuracy 2025-08-23 04:04:32 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:04:32 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:04:32 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:04:32 [info     ] ✅ Question Generator DSPy configured via centralized provider: gemma3n:e4b
2025-08-23 04:04:32 [info     ] ✅ DSPy configured with model: gemma3n:e4b
2025-08-23 04:04:32 [info     ] ✅ Few-shot optimizer configured with medical examples
2025-08-23 04:04:32 [info     ] 📊 Emergency-pattern DataFrame prepared patterns=102
2025-08-23 04:04:32 [info     ] ❓ Medical Question Generator initialised model=gemma3n:e4b
2025-08-23 04:04:32 [info     ] 🩺 Medical Triage Agent initialized model=gemma3n:e4b

-------------------------------------------------------------------------------------------------------------------------------------- live log call --------------------------------------------------------------------------------------------------------------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
2025-08-23 04:05:53 [info     ] 🔄 Medical turn processed       confidence=95 outcome=emergency thinking_captured=True turn=1
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/show "HTTP/1.1 200 OK"
INFO     httpx:_client.py:1025 HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
INFO     httpx
