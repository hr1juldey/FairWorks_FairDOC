# Progress

This document tracks what works, what's left to build, the current status, known issues, and the evolution of project decisions for the Fairdoc AI Triage System.

## What Works:
- **Basic Conversation Flow:** The system can process a conversation turn, as seen in `test_conversation_initialization` passing.
- **Emergency Detection:** The medical agent can detect emergency situations with high confidence in some scenarios (`test_medical_agent_basic_triage`, `test_emergency_detection_accuracy`).
- **NICE Protocol Integration:** The system can match symptoms to NICE protocols (`test_nice_protocol_integration`).
- **DSPy Configuration:** DSPy is configured and used for various components, although with some async-related issues.
- **Database Seeding:** Gold standard dialogues are seeded for evaluation.

## What's Left to Build:
- **Robust Asynchronous Handling:** The `asyncio.run()` errors need to be resolved to ensure stable asynchronous operations.
- **Context Continuity:** `test_context_continuity` is failing, indicating that context is not being maintained across conversation turns. This is a critical issue to fix.
- **State Management:** `test_redis_state_management` is failing, suggesting issues with Redis-based state management.
- **Stakeholder Routing:** `test_stakeholder_routing` is failing, meaning the system is not correctly routing messages to different stakeholders (e.g., patient, doctor).
- **Emergency Escalation Workflow:** `test_emergency_escalation_workflow` is failing, which is a critical safety issue.
- **DSPy Optimizer Strategies:** Several DSPy optimization strategies are failing (`mipro`, `copro`, `labeled_fewshot`, `knn`), which limits the ability to improve the AI's performance.
- **Performance:** `test_performance_metrics` is failing due to slow response times.

## Current Status:
- The project is in a proof-of-concept (POC) stage, with many core functionalities implemented but several critical tests failing.
- The focus is on fixing the failing tests to stabilize the system.

## Known Issues:
- **`asyncio.run() cannot be called from a running event loop`:** This is a recurring error in the logs, indicating a fundamental issue with how async functions are being called within the test suite or application.
- **`UnboundLocalError: cannot access local variable 'optimized_program'`:** This error in `test_nice_optimization_strategies` suggests a problem with the optimization logic.
- **`KeyError: 'examples_evaluated'`:** This error in `test_model_evaluation_against_gold_standards` points to an issue with the evaluation results format.
- **Slow LLM Response Time:** The performance test failure indicates that the LLM is taking too long to respond.
- **Red Flag Detection Failure:** `test_red_flag_detection` is failing, which is a critical safety concern.

## Evolution of Project Decisions:
- The project has evolved from a single `app` to a more structured `app2`, indicating a significant refactoring or redesign.
- The use of DSPy suggests a shift towards more systematic and optimizable LLM development.
- The extensive test suite, although with many failures, shows a commitment to quality and reliability.
