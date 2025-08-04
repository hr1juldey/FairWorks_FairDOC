# DSPy Module Evaluation Framework
"""
Comprehensive evaluation framework for DSPy medical modules
Tests question generation, medical reasoning, and emergency detection
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import structlog
import statistics

from src.tests.e2e.user_scenarios.patient_profiles import get_all_patient_ids, get_patient_profile
from src.tests.e2e.user_scenarios.medical_conditions import get_condition, get_all_condition_ids
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.services.dspy.evaluation_optimizer import EvaluationOptimizer
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

class DSPyModuleEvaluator:
    """Evaluates individual DSPy modules for medical triage"""
    
    def __init__(self):
        self.medical_agent = MedicalTriageAgent(model_name=settings_v2.DSPY_MODEL_NAME)
        self.question_generator = MedicalQuestionGenerator(model_name=settings_v2.DSPY_MODEL_NAME)
        self.nice_lookup = NICELookupService()
        self.evaluation_optimizer = EvaluationOptimizer(model_name=settings_v2.DSPY_MODEL_NAME)
        
        self.test_results = {
            "question_generator": [],
            "medical_agent": [],
            "emergency_detection": [],
            "nice_lookup": []
        }
    
    async def evaluate_question_generator(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test question generator with various symptom scenarios"""
        logger.info("🧪 Testing DSPy Question Generator")
        
        results = []
        
        for scenario in test_scenarios:
            try:
                # Generate questions using DSPy
                question_result = self.question_generator.suggest_questions(
                    symptom_text=scenario["symptoms"],
                    conversation_context=scenario.get("context", ""),
                    nice_protocols=scenario.get("protocols", ""),
                    max_questions=3
                )
                
                # Evaluate question quality
                evaluation = self._evaluate_question_quality(
                    questions=question_result["questions"],
                    expected_focus=scenario.get("expected_focus", []),
                    urgency_level=question_result.get("urgency_level", "low")
                )
                
                result = {
                    "scenario_id": scenario["id"],
                    "symptoms": scenario["symptoms"],
                    "generated_questions": question_result["questions"],
                    "urgency_detected": question_result.get("urgency_level", "low"),
                    "medical_reasoning": question_result.get("medical_reasoning", ""),
                    "emergency_indicators": question_result.get("emergency_indicators", []),
                    "quality_score": evaluation["quality_score"],
                    "relevance_score": evaluation["relevance_score"],
                    "clinical_appropriateness": evaluation["clinical_score"]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error("❌ Question generator test failed", 
                           scenario_id=scenario["id"], error=str(e))
                results.append({
                    "scenario_id": scenario["id"],
                    "error": str(e),
                    "quality_score": 0,
                    "relevance_score": 0,
                    "clinical_appropriateness": 0
                })
        
        # Calculate aggregate metrics
        quality_scores = [r.get("quality_score", 0) for r in results]
        relevance_scores = [r.get("relevance_score", 0) for r in results]
        clinical_scores = [r.get("clinical_appropriateness", 0) for r in results]
        
        summary = {
            "total_scenarios": len(test_scenarios),
            "successful_tests": len([r for r in results if "error" not in r]),
            "average_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
            "average_relevance_score": statistics.mean(relevance_scores) if relevance_scores else 0,
            "average_clinical_score": statistics.mean(clinical_scores) if clinical_scores else 0,
            "overall_performance": statistics.mean([
                statistics.mean(quality_scores) if quality_scores else 0,
                statistics.mean(relevance_scores) if relevance_scores else 0,
                statistics.mean(clinical_scores) if clinical_scores else 0
            ])
        }
        
        self.test_results["question_generator"] = {
            "individual_results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Question generator evaluation completed", 
                   performance=summary["overall_performance"])
        
        return self.test_results["question_generator"]
    
    async def evaluate_medical_agent(self, patient_symptom_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Test medical agent reasoning with patient profiles"""
        logger.info("🩺 Testing DSPy Medical Agent")
        
        results = []
        
        for patient_id, condition_id in patient_symptom_pairs:
            try:
                # Get patient profile and condition
                profile = get_patient_profile(patient_id)
                condition = get_condition(condition_id)
                
                if not profile or not condition:
                    continue
                
                # Create initial symptom presentation
                initial_symptoms = condition.symptom_progression.initial
                
                # Get NICE protocols
                nice_result = self.nice_lookup.find_relevant_protocols(initial_symptoms)
                
                # Process through medical agent
                agent_result = await self.medical_agent.process_turn(
                    symptoms=initial_symptoms,
                    nice_context=nice_result["protocol_text"],
                    history=None
                )
                
                # Evaluate reasoning quality
                evaluation = self._evaluate_medical_reasoning(
                    agent_result=agent_result,
                    expected_outcome=condition.expected_outcome.value,
                    expected_red_flags=condition.red_flag_indicators,
                    patient_profile=profile
                )
                
                result = {
                    "patient_id": patient_id,
                    "condition_id": condition_id,
                    "initial_symptoms": initial_symptoms,
                    "agent_outcome": agent_result.get("outcome", "inconclusive"),
                    "agent_confidence": agent_result.get("confidence", 0),
                    "red_flags_detected": agent_result.get("red_flags", []),
                    "next_question": agent_result.get("next_question"),
                    "reasoning_quality": evaluation["reasoning_score"],
                    "outcome_accuracy": evaluation["outcome_score"],
                    "emergency_detection": evaluation["emergency_score"],
                    "overall_score": evaluation["overall_score"]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error("❌ Medical agent test failed",
                           patient_id=patient_id, error=str(e))
                results.append({
                    "patient_id": patient_id,
                    "condition_id": condition_id,
                    "error": str(e),
                    "overall_score": 0
                })
        
        # Calculate aggregate metrics
        overall_scores = [r.get("overall_score", 0) for r in results]
        outcome_accuracies = [r.get("outcome_accuracy", 0) for r in results]
        emergency_scores = [r.get("emergency_detection", 0) for r in results]
        
        summary = {
            "total_tests": len(patient_symptom_pairs),
            "successful_tests": len([r for r in results if "error" not in r]),
            "average_overall_score": statistics.mean(overall_scores) if overall_scores else 0,
            "average_outcome_accuracy": statistics.mean(outcome_accuracies) if outcome_accuracies else 0,
            "average_emergency_detection": statistics.mean(emergency_scores) if emergency_scores else 0,
            "pass_rate": len([s for s in overall_scores if s >= 60]) / len(overall_scores) * 100 if overall_scores else 0
        }
        
        self.test_results["medical_agent"] = {
            "individual_results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Medical agent evaluation completed",
                   performance=summary["average_overall_score"])
        
        return self.test_results["medical_agent"]
    
    async def evaluate_emergency_detection(self, emergency_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test emergency detection accuracy"""
        logger.info("🚨 Testing Emergency Detection")
        
        results = []
        
        for scenario in emergency_scenarios:
            try:
                # Test question generator emergency detection
                question_result = self.question_generator.suggest_questions(
                    symptom_text=scenario["symptoms"],
                    max_questions=2
                )
                
                # Test medical agent emergency routing
                nice_result = self.nice_lookup.find_relevant_protocols(scenario["symptoms"])
                agent_result = await self.medical_agent.process_turn(
                    symptoms=scenario["symptoms"],
                    nice_context=nice_result["protocol_text"]
                )
                
                # Evaluate emergency detection
                question_urgency = question_result.get("urgency_level", "low")
                agent_outcome = agent_result.get("outcome", "inconclusive")
                
                emergency_detected = (
                    question_urgency in ["high", "critical"] or
                    agent_outcome == "emergency"
                )
                
                result = {
                    "scenario_id": scenario["id"],
                    "symptoms": scenario["symptoms"],
                    "is_emergency": scenario["is_emergency"],
                    "question_urgency": question_urgency,
                    "agent_outcome": agent_outcome,
                    "emergency_detected": emergency_detected,
                    "true_positive": scenario["is_emergency"] and emergency_detected,
                    "false_positive": not scenario["is_emergency"] and emergency_detected,
                    "true_negative": not scenario["is_emergency"] and not emergency_detected,
                    "false_negative": scenario["is_emergency"] and not emergency_detected
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error("❌ Emergency detection test failed",
                           scenario_id=scenario["id"], error=str(e))
                results.append({
                    "scenario_id": scenario["id"],
                    "error": str(e),
                    "true_positive": False,
                    "false_positive": False,
                    "true_negative": False,
                    "false_negative": True if scenario["is_emergency"] else False
                })
        
        # Calculate confusion matrix metrics
        tp = len([r for r in results if r.get("true_positive", False)])
        fp = len([r for r in results if r.get("false_positive", False)])
        tn = len([r for r in results if r.get("true_negative", False)])
        fn = len([r for r in results if r.get("false_negative", False)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(results) if results else 0
        
        summary = {
            "total_scenarios": len(emergency_scenarios),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "emergency_detection_rate": recall  # Sensitivity for emergency cases
        }
        
        self.test_results["emergency_detection"] = {
            "individual_results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Emergency detection evaluation completed",
                   f1_score=f1_score, recall=recall)
        
        return self.test_results["emergency_detection"]
    
    async def evaluate_nice_lookup(self, symptom_protocol_pairs: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
        """Test NICE protocol lookup accuracy"""
        logger.info("📋 Testing NICE Lookup Service")
        
        results = []
        
        for symptoms, expected_protocols in symptom_protocol_pairs:
            try:
                # Lookup protocols
                nice_result = self.nice_lookup.find_relevant_protocols(symptoms)
                found_protocol = nice_result.get("protocol_code", "")
                
                # Check if found protocol matches expected
                protocol_match = any(expected in found_protocol for expected in expected_protocols)
                
                result = {
                    "symptoms": symptoms,
                    "expected_protocols": expected_protocols,
                    "found_protocol": found_protocol,
                    "protocol_match": protocol_match,
                    "protocol_text_length": len(nice_result.get("protocol_text", ""))
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error("❌ NICE lookup test failed",
                           symptoms=symptoms, error=str(e))
                results.append({
                    "symptoms": symptoms,
                    "expected_protocols": expected_protocols,
                    "error": str(e),
                    "protocol_match": False
                })
        
        # Calculate metrics
        matches = [r.get("protocol_match", False) for r in results]
        match_rate = sum(matches) / len(matches) * 100 if matches else 0
        
        summary = {
            "total_tests": len(symptom_protocol_pairs),
            "successful_lookups": len([r for r in results if "error" not in r]),
            "protocol_match_rate": match_rate,
            "average_protocol_text_length": statistics.mean([
                r.get("protocol_text_length", 0) for r in results if "error" not in r
            ]) if results else 0
        }
        
        self.test_results["nice_lookup"] = {
            "individual_results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ NICE lookup evaluation completed",
                   match_rate=match_rate)
        
        return self.test_results["nice_lookup"]
    
    def _evaluate_question_quality(self, questions: List[str], expected_focus: List[str], urgency_level: str) -> Dict[str, float]:
        """Evaluate quality of generated questions"""
        
        # Quality indicators
        quality_indicators = [
            "severity", "duration", "location", "pain", "scale", "when", "how", "what", "any"
        ]
        
        clinical_indicators = [
            "emergency", "breathing", "chest", "heart", "blood", "hospital", "doctor"
        ]
        
        quality_score = 0
        relevance_score = 0
        clinical_score = 0
        
        for question in questions:
            question_lower = question.lower()
            
            # Quality: structured medical questions
            quality_score += sum(10 for indicator in quality_indicators if indicator in question_lower) / len(questions)
            
            # Relevance: matches expected focus areas
            if expected_focus:
                relevance_score += sum(15 for focus in expected_focus if focus.lower() in question_lower) / len(questions)
            else:
                relevance_score = 80  # Default if no specific focus expected
            
            # Clinical appropriateness
            clinical_score += sum(12 for indicator in clinical_indicators if indicator in question_lower) / len(questions)
        
        # Urgency appropriateness bonus
        if urgency_level in ["high", "critical"]:
            clinical_score += 20
        
        return {
            "quality_score": min(100, quality_score),
            "relevance_score": min(100, relevance_score),
            "clinical_score": min(100, clinical_score)
        }
    
    def _evaluate_medical_reasoning(self, agent_result: Dict[str, Any], expected_outcome: str, 
                                  expected_red_flags: List[str], patient_profile: Any) -> Dict[str, float]:
        """Evaluate medical agent reasoning quality"""
        
        # Outcome accuracy
        outcome_mapping = {
            "emergency_route_to_doctor": "emergency",
            "routine_doctor_consultation": "routine", 
            "self_care_advice": "self_care",
            "need_more_questions": "inconclusive"
        }
        
        expected = outcome_mapping.get(expected_outcome, expected_outcome)
        actual = outcome_mapping.get(agent_result.get("outcome", ""), agent_result.get("outcome", ""))
        
        if expected == actual:
            outcome_score = 100
        elif expected == "emergency" and actual in ["routine", "emergency"]:
            outcome_score = 80  # At least escalated
        elif expected == "routine" and actual in ["routine", "emergency"]:
            outcome_score = 90  # Conservative approach
        else:
            outcome_score = 40
        
        # Red flag detection
        detected_flags = set(agent_result.get("red_flags", []))
        expected_flags = set(expected_red_flags)
        
        if expected_flags:
            red_flag_recall = len(expected_flags.intersection(detected_flags)) / len(expected_flags)
            emergency_score = red_flag_recall * 100
        else:
            emergency_score = 90 if not detected_flags else 70  # No false positives preferred
        
        # Reasoning quality (confidence and consistency)
        confidence = agent_result.get("confidence", 0)
        has_reasoning = bool(agent_result.get("reasoning", ""))
        reasoning_score = (confidence + (50 if has_reasoning else 0)) / 1.5
        
        overall_score = (outcome_score * 0.5 + emergency_score * 0.3 + reasoning_score * 0.2)
        
        return {
            "outcome_score": outcome_score,
            "emergency_score": emergency_score,
            "reasoning_score": reasoning_score,
            "overall_score": overall_score
        }
    
    def generate_module_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report for all DSPy modules"""
        
        # Overall summary
        module_scores = {}
        for module_name, results in self.test_results.items():
            if results and "summary" in results:
                summary = results["summary"]
                if "overall_performance" in summary:
                    module_scores[module_name] = summary["overall_performance"]
                elif "average_overall_score" in summary:
                    module_scores[module_name] = summary["average_overall_score"]
                elif "f1_score" in summary:
                    module_scores[module_name] = summary["f1_score"] * 100
                elif "protocol_match_rate" in summary:
                    module_scores[module_name] = summary["protocol_match_rate"]
                else:
                    module_scores[module_name] = 50  # Default neutral score
        
        overall_system_score = statistics.mean(module_scores.values()) if module_scores else 0
        
        return {
            "evaluation_summary": {
                "overall_system_score": overall_system_score,
                "module_scores": module_scores,
                "system_pass": overall_system_score >= 60.0,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations(module_scores),
            "test_configuration": {
                "model_name": settings_v2.DSPY_MODEL_NAME,
                "pass_threshold": 60.0
            }
        }
    
    def _generate_recommendations(self, module_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        for module, score in module_scores.items():
            if score < 60:
                recommendations.append(f"❌ {module.title()}: Score {score:.1f} - Requires optimization")
            elif score < 80:
                recommendations.append(f"⚠️ {module.title()}: Score {score:.1f} - Consider fine-tuning")
            else:
                recommendations.append(f"✅ {module.title()}: Score {score:.1f} - Performing well")
        
        if not recommendations:
            recommendations.append("ℹ️ No test results available for analysis")
        
        return recommendations
