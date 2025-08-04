# Conversation Orchestrator - E2E Test Orchestration
"""
Orchestrates end-to-end conversations between DSPy patient agents and medical agents
Manages conversation flows, timing, evaluation, and result aggregation
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog

from src.tests.e2e.user_scenarios.patient_profiles import get_patient_profile, get_all_patient_ids
from src.tests.e2e.user_scenarios.medical_conditions import get_condition
from src.tests.e2e.user_scenarios.dspy_patient_agent import create_patient_agent
from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.models.schemas.multiturn_chat import MultiTurnChatRequest, StakeholderRole, ChatProvider
from src.tests.e2e.user_scenarios.medical_conditions import MedicalCondition  # Add MedicalCondition here


logger = structlog.get_logger(__name__)

@dataclass
class ConversationMetrics:
    """Metrics collected during conversation"""
    total_turns: int
    conversation_duration_seconds: float
    patient_emotional_progression: List[str]
    medical_agent_questions: List[str]
    confidence_scores: List[float]
    red_flags_detected: List[str]
    nice_protocols_triggered: List[str]
    final_outcome: str
    emergency_correctly_identified: bool
    patient_satisfaction_estimated: float

@dataclass
class ConversationResult:
    """Complete result of single conversation test"""
    patient_id: str
    conversation_transcript: List[Dict[str, Any]]
    metrics: ConversationMetrics
    evaluation_score: float
    test_passed: bool
    failure_reasons: List[str]
    execution_time: datetime

class ConversationOrchestrator:
    """Orchestrates E2E conversation testing"""
    
    def __init__(self, chat_orchestrator: ChatOrchestrator):
        self.chat_orchestrator = chat_orchestrator
        self.conversation_results: List[ConversationResult] = []
        self.active_conversations: Dict[str, Dict] = {}
        
    async def run_single_conversation(
        self,
        patient_id: str,
        max_turns: int = 8,
        timeout_minutes: int = 5
    ) -> ConversationResult:
        """Run single E2E conversation between patient agent and medical agent"""
        
        logger.info("🎭 Starting single E2E conversation",
                   patient_id=patient_id, max_turns=max_turns)
        
        start_time = time.time()
        execution_timestamp = datetime.now()
        
        # Initialize patient agent
        patient_profile = get_patient_profile(patient_id)
        if not patient_profile:
            raise ValueError(f"Patient profile {patient_id} not found")
        
        try:
            from src.app2.core.config_v2 import settings_v2
            patient_agent = create_patient_agent(patient_id, settings_v2.DSPY_MODEL_NAME)
        except Exception as e:
            logger.error("❌ Failed to create patient agent", patient_id=patient_id, error=str(e))
            return self._create_failed_result(patient_id, ["patient_agent_creation_failed"], execution_timestamp)
        
        # Initialize conversation tracking
        conversation_transcript = []
        metrics = ConversationMetrics(
            total_turns=0,
            conversation_duration_seconds=0.0,
            patient_emotional_progression=[],
            medical_agent_questions=[],
            confidence_scores=[],
            red_flags_detected=[],
            nice_protocols_triggered=[],
            final_outcome="incomplete",
            emergency_correctly_identified=False,
            patient_satisfaction_estimated=0.0
        )
        
        conversation_id = f"test_{patient_id}_{str(uuid.uuid4())}"
        
        try:
            # Start conversation with initial patient message
            initial_symptoms = patient_agent.medical_condition.symptom_progression.initial
            
            # Patient initiates conversation
            patient_response = await patient_agent.respond_to_agent(
                f"Please tell me about your symptoms. I understand you're experiencing: {initial_symptoms}",
                "professional"
            )
            
            turn_number = 1
            conversation_active = True
            
            while conversation_active and turn_number <= max_turns:
                
                # Check timeout
                if time.time() - start_time > timeout_minutes * 60:
                    logger.warning("⏰ Conversation timeout",
                                  patient_id=patient_id, turn=turn_number)
                    break
                
                logger.info("💬 Conversation turn",
                           patient_id=patient_id, turn=turn_number)
                
                # Send patient message to medical agent
                
                chat_request = MultiTurnChatRequest(
                conversation_id=conversation_id,
                user_message=patient_response["patient_message"] + f" Initial symptoms: {initial_symptoms}",  # Use initial_symptoms in the chat request
                stakeholder_role=StakeholderRole.PATIENT,
                stakeholder_id=patient_id,
                chat_provider=ChatProvider.API_DIRECT,
                patient_age=patient_profile.age,
                patient_gender=patient_profile.gender.value
            )

                
                # Process through chat orchestrator (medical agent)
                orchestration_result = await self.chat_orchestrator.process_conversation_turn(chat_request)
                medical_response = self.chat_orchestrator.build_chat_response(orchestration_result)
                
                # Record turn in transcript
                turn_record = {
                    "turn": turn_number,
                    "patient_message": patient_response["patient_message"],
                    "patient_emotional_state": patient_response["emotional_state"],
                    "patient_urgency": patient_response["urgency_level"],
                    "medical_agent_question": medical_response.agent_message,
                    "medical_confidence": medical_response.confidence_score,
                    "medical_outcome": medical_response.medical_outcome.value,
                    "red_flags": [rf.symptom for rf in medical_response.red_flags_detected],
                    "protocols": medical_response.relevant_protocols,
                    "timestamp": datetime.now().isoformat()
                }
                
                conversation_transcript.append(turn_record)
                
                # Update metrics
                metrics.total_turns = turn_number
                metrics.patient_emotional_progression.append(patient_response["emotional_state"])
                metrics.medical_agent_questions.append(medical_response.agent_message or "")
                metrics.confidence_scores.append(float(medical_response.confidence_score))
                metrics.red_flags_detected.extend([rf.symptom for rf in medical_response.red_flags_detected])
                metrics.nice_protocols_triggered.extend(medical_response.relevant_protocols)
                metrics.final_outcome = medical_response.medical_outcome.value
                
                # Check if conversation should end
                if medical_response.is_emergency:
                    logger.info("🚨 Emergency detected, ending conversation",
                               patient_id=patient_id, turn=turn_number)
                    conversation_active = False
                    break
                
                if not medical_response.next_question or medical_response.conversation_status.value == "completed":
                    logger.info("✅ Conversation completed by medical agent",
                               patient_id=patient_id, turn=turn_number)
                    conversation_active = False
                    break
                
                # Get next patient response
                if conversation_active and medical_response.next_question:
                    patient_response = await patient_agent.respond_to_agent(
                        medical_response.next_question,
                        "professional"
                    )
                    
                    # Check if patient wants to end conversation
                    if not patient_response.get("conversation_active", True):
                        logger.info("👤 Patient ended conversation",
                                   patient_id=patient_id, turn=turn_number)
                        conversation_active = False
                        break
                
                turn_number += 1
                
                # Safety pause between turns
                await asyncio.sleep(0.1)
            
            # Calculate final metrics
            metrics.conversation_duration_seconds = time.time() - start_time
            metrics.emergency_correctly_identified = self._evaluate_emergency_detection(
                patient_agent.medical_condition, medical_response
            )
            metrics.patient_satisfaction_estimated = self._estimate_patient_satisfaction(
                patient_response, metrics
            )
            
            # Evaluate conversation quality
            evaluation_score = self._evaluate_conversation_quality(
                patient_agent.medical_condition, metrics, conversation_transcript
            )
            
            # Determine if test passed
            test_passed, failure_reasons = self._determine_test_outcome(
                patient_agent.medical_condition, metrics, evaluation_score
            )
            
            result = ConversationResult(
                patient_id=patient_id,
                conversation_transcript=conversation_transcript,
                metrics=metrics,
                evaluation_score=evaluation_score,
                test_passed=test_passed,
                failure_reasons=failure_reasons,
                execution_time=execution_timestamp
            )
            
            logger.info("✅ Single conversation completed",
                       patient_id=patient_id,
                       turns=metrics.total_turns,
                       score=evaluation_score,
                       passed=test_passed,
                       outcome=metrics.final_outcome)
            
            return result
            
        except Exception as e:
            logger.error("❌ Conversation execution failed",
                        patient_id=patient_id, error=str(e))
            return self._create_failed_result(
                patient_id, 
                ["conversation_execution_failed", str(e)], 
                execution_timestamp
            )
    
    async def run_batch_conversations(
        self,
        patient_ids: List[str],
        max_concurrent: int = 3,
        max_turns_per_conversation: int = 8
    ) -> List[ConversationResult]:
        """Run batch of conversations with concurrency control"""
        
        logger.info("🎭 Starting batch conversations",
                   total_patients=len(patient_ids), max_concurrent=max_concurrent)
        
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_with_semaphore(patient_id: str) -> ConversationResult:
            async with semaphore:
                return await self.run_single_conversation(
                    patient_id=patient_id,
                    max_turns=max_turns_per_conversation,
                    timeout_minutes=3  # Shorter timeout for batch
                )
        
        # Execute conversations concurrently
        tasks = [run_single_with_semaphore(pid) for pid in patient_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and failed results
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("❌ Batch conversation exception",
                           patient_id=patient_ids[i], error=str(result))
                failed_count += 1
            elif isinstance(result, ConversationResult):
                successful_results.append(result)
            else:
                failed_count += 1
        
        logger.info("✅ Batch conversations completed",
                   successful=len(successful_results),
                   failed=failed_count,
                   total=len(patient_ids))
        
        self.conversation_results.extend(successful_results)
        return successful_results
    
    def _evaluate_emergency_detection(
        self, 
        condition: MedicalCondition,
        medical_response
    ) -> bool:
        """Evaluate if emergency was correctly detected"""
        
        is_emergency_condition = condition.expected_outcome.value == "emergency_route_to_doctor"
        emergency_detected = medical_response.is_emergency or medical_response.medical_outcome.value == "emergency"
        
        # For emergency conditions, should detect emergency
        if is_emergency_condition:
            return emergency_detected
        
        # For non-emergency conditions, should NOT detect emergency
        return not emergency_detected
    
    def _estimate_patient_satisfaction(
        self,
        final_patient_response: Dict[str, Any],
        metrics: ConversationMetrics
    ) -> float:
        """Estimate patient satisfaction based on conversation flow"""
        
        base_satisfaction = 50.0
        
        # Positive factors
        if final_patient_response.get("trust_level") == "high":
            base_satisfaction += 20
        elif final_patient_response.get("trust_level") == "medium":
            base_satisfaction += 10
        
        if final_patient_response.get("emotional_state") in ["calm", "hopeful"]:
            base_satisfaction += 15
        elif final_patient_response.get("emotional_state") in ["worried", "anxious"]:
            base_satisfaction -= 10
        
        # Turn count factor (too short or too long is bad)
        if 3 <= metrics.total_turns <= 6:
            base_satisfaction += 10
        elif metrics.total_turns < 2:
            base_satisfaction -= 20
        elif metrics.total_turns > 8:
            base_satisfaction -= 15
        
        # Confidence progression (increasing confidence is good)
        if len(metrics.confidence_scores) >= 2:
            if metrics.confidence_scores[-1] > metrics.confidence_scores[0]:
                base_satisfaction += 10
        
        return max(0.0, min(100.0, base_satisfaction))
    
    def _evaluate_conversation_quality(
        self,
        condition: MedicalCondition,
        metrics: ConversationMetrics,
        transcript: List[Dict[str, Any]]
    ) -> float:
        """Evaluate overall conversation quality (0-100 score)"""
        
        score = 0.0
        
        # Emergency detection accuracy (30 points)
        if metrics.emergency_correctly_identified:
            score += 30
        elif condition.emergency_detection_critical:
            # Critical failure for missing emergency
            score -= 20
        
        # Appropriate turn count (20 points)
        expected_min = condition.minimum_turns_expected
        expected_max = condition.maximum_turns_acceptable
        
        if expected_min <= metrics.total_turns <= expected_max:
            score += 20
        elif metrics.total_turns < expected_min:
            score += 10  # Too quick but maybe okay
        else:
            score -= 10  # Too long
        
        # NICE protocol triggering (15 points)
        expected_protocols = condition.relevant_protocols
        triggered_protocols = metrics.nice_protocols_triggered
        
        protocol_match_rate = len(set(expected_protocols) & set(triggered_protocols)) / max(len(expected_protocols), 1)
        score += 15 * protocol_match_rate
        
        # Red flag detection (15 points)
        expected_flags = condition.red_flag_indicators
        detected_flags = metrics.red_flags_detected
        
        if expected_flags:  # Only if red flags are expected
            flag_detection_rate = len(set(expected_flags) & set(detected_flags)) / len(expected_flags)
            score += 15 * flag_detection_rate
        else:
            score += 15  # Full points if no flags expected and none detected
        
        # Final outcome matching (20 points)
        if metrics.final_outcome == condition.expected_outcome.value:
            score += 20
        elif condition.expected_outcome.value == "emergency_route_to_doctor":
            # Severe penalty for missing emergency
            score -= 25
        
        # Confidence progression bonus (5 points)
        if len(metrics.confidence_scores) >= 2:
            final_confidence = metrics.confidence_scores[-1]
            if final_confidence >= 70:
                score += 5
        
        # Patient satisfaction bonus (5 points)
        if metrics.patient_satisfaction_estimated >= 70:
            score += 5
        
        return max(0.0, min(100.0, score))
    
    def _determine_test_outcome(
        self,
        condition: MedicalCondition,
        metrics: ConversationMetrics,
        evaluation_score: float
    ) -> Tuple[bool, List[str]]:
        """Determine if test passed and identify failure reasons"""
        
        failure_reasons = []
        
        # Minimum score threshold
        if evaluation_score < 60:
            failure_reasons.append(f"evaluation_score_too_low_{evaluation_score}")
        
        # Critical emergency detection failures
        if condition.emergency_detection_critical and not metrics.emergency_correctly_identified:
            failure_reasons.append("critical_emergency_not_detected")
        
        # Wrong final outcome
        if metrics.final_outcome != condition.expected_outcome.value:
            failure_reasons.append(f"wrong_outcome_{metrics.final_outcome}_expected_{condition.expected_outcome.value}")
        
        # Excessive turns for emergency
        if (condition.expected_outcome.value == "emergency_route_to_doctor" and 
            metrics.total_turns > 4):
            failure_reasons.append("emergency_took_too_long")
        
        # Insufficient turns for complex cases
        if (condition.expected_outcome.value == "routine_doctor_consultation" and
            metrics.total_turns < 2):
            failure_reasons.append("insufficient_information_gathering")
        
        test_passed = len(failure_reasons) == 0
        
        return test_passed, failure_reasons
    
    def _create_failed_result(
        self,
        patient_id: str,
        failure_reasons: List[str],
        execution_time: datetime
    ) -> ConversationResult:
        """Create failed conversation result"""
        
        return ConversationResult(
            patient_id=patient_id,
            conversation_transcript=[],
            metrics=ConversationMetrics(
                total_turns=0,
                conversation_duration_seconds=0.0,
                patient_emotional_progression=[],
                medical_agent_questions=[],
                confidence_scores=[],
                red_flags_detected=[],
                nice_protocols_triggered=[],
                final_outcome="failed",
                emergency_correctly_identified=False,
                patient_satisfaction_estimated=0.0
            ),
            evaluation_score=0.0,
            test_passed=False,
            failure_reasons=failure_reasons,
            execution_time=execution_time
        )
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report of all conversations"""
        
        if not self.conversation_results:
            return {"error": "No conversation results available"}
        
        total_conversations = len(self.conversation_results)
        passed_conversations = len([r for r in self.conversation_results if r.test_passed])
        
        # Aggregate metrics
        all_scores = [r.evaluation_score for r in self.conversation_results]
        all_turns = [r.metrics.total_turns for r in self.conversation_results]
        all_durations = [r.metrics.conversation_duration_seconds for r in self.conversation_results]
        
        outcome_distribution = {}
        for result in self.conversation_results:
            outcome = result.metrics.final_outcome
            outcome_distribution[outcome] = outcome_distribution.get(outcome, 0) + 1
        
        # Emergency detection analysis
        emergency_cases = [r for r in self.conversation_results 
                          if get_condition(get_patient_profile(r.patient_id).medical_condition).expected_outcome.value == "emergency_route_to_doctor"]
        emergency_detection_rate = len([r for r in emergency_cases if r.metrics.emergency_correctly_identified]) / max(len(emergency_cases), 1)
        
        return {
            "summary": {
                "total_conversations": total_conversations,
                "passed_conversations": passed_conversations,
                "pass_rate": passed_conversations / total_conversations * 100,
                "average_score": sum(all_scores) / len(all_scores),
                "average_turns": sum(all_turns) / len(all_turns),
                "average_duration_seconds": sum(all_durations) / len(all_durations)
            },
            "outcome_distribution": outcome_distribution,
            "emergency_detection": {
                "total_emergency_cases": len(emergency_cases),
                "correctly_detected": len([r for r in emergency_cases if r.metrics.emergency_correctly_identified]),
                "detection_rate": emergency_detection_rate * 100
            },
            "score_distribution": {
                "scores_above_80": len([s for s in all_scores if s >= 80]),
                "scores_60_to_80": len([s for s in all_scores if 60 <= s < 80]),
                "scores_below_60": len([s for s in all_scores if s < 60])
            },
            "common_failure_reasons": self._analyze_failure_patterns(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_failure_patterns(self) -> Dict[str, int]:
        """Analyze common failure patterns"""
        
        failure_counts = {}
        
        for result in self.conversation_results:
            if not result.test_passed:
                for reason in result.failure_reasons:
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        return dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results"""
        
        recommendations = []
        
        if not self.conversation_results:
            return ["No data available for recommendations"]
        
        pass_rate = len([r for r in self.conversation_results if r.test_passed]) / len(self.conversation_results)
        
        if pass_rate < 0.6:
            recommendations.append("Overall pass rate is low - consider DSPy model optimization")
        
        # Emergency detection issues
        emergency_cases = [r for r in self.conversation_results 
                          if get_condition(get_patient_profile(r.patient_id).medical_condition).expected_outcome.value == "emergency_route_to_doctor"]
        if emergency_cases:
            detection_rate = len([r for r in emergency_cases if r.metrics.emergency_correctly_identified]) / len(emergency_cases)
            if detection_rate < 0.8:
                recommendations.append("Emergency detection needs improvement - review red flag indicators")
        
        # Turn count issues
        avg_turns = sum([r.metrics.total_turns for r in self.conversation_results]) / len(self.conversation_results)
        if avg_turns > 7:
            recommendations.append("Conversations taking too long - optimize question generation")
        elif avg_turns < 3:
            recommendations.append("Conversations too short - may need more thorough assessment")
        
        # Score distribution
        low_scores = len([r for r in self.conversation_results if r.evaluation_score < 60])
        if low_scores > len(self.conversation_results) * 0.3:
            recommendations.append("Many low scores - review DSPy training data and prompts")
        
        if not recommendations:
            recommendations.append("System performing well - continue monitoring")
        
        return recommendations
