# tests/training_poc/evaluation_suite.py
"""
Medical Evaluation Suite - POC Implementation

Comprehensive evaluation framework for medical triage AI systems.
Implements standard medical AI metrics and evaluation protocols.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import dspy

# Import existing modules
import sys
from pathlib import Path

from src.app2.services.dspy.medical_agent import MedicalTriageAgent

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for medical triage"""
    
    # Basic accuracy metrics
    overall_accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Medical-specific metrics
    emergency_precision: float
    emergency_recall: float
    emergency_f1: float
    emergency_sensitivity: float  # True positive rate for emergencies
    emergency_specificity: float  # True negative rate for emergencies
    
    # Safety metrics  
    missed_emergencies: int       # False negatives for emergency
    false_emergency_alerts: int   # False positives for emergency
    overtriage_rate: float        # Proportion over-triaged
    undertriage_rate: float       # Proportion under-triaged
    
    # Performance metrics
    avg_response_time_ms: float
    total_evaluations: int
    evaluation_time_seconds: float
    
    # Detailed breakdown
    confusion_matrix_data: Dict[str, Dict[str, int]]
    per_class_metrics: Dict[str, Dict[str, float]]

class MedicalEvaluationSuite:
    """Comprehensive evaluation suite for medical triage systems"""
    
    def __init__(self):
        self.outcome_classes = ["emergency", "routine", "self_care"]
        self.safety_weights = {
            "emergency_missed": -10.0,      # Very high penalty
            "emergency_false_positive": -1.0,  # Minor penalty  
            "routine_to_emergency": 0.5,    # Over-cautious but safe
            "self_care_to_routine": 0.2     # Minor over-caution
        }
    
    async def evaluate_agent(self, agent: MedicalTriageAgent, 
                           num_examples: int = 50) -> Dict[str, float]:
        """Evaluate agent performance on test examples"""
        
        logger.info(f"🔍 Evaluating agent on {num_examples} examples...")
        
        # Generate test cases
        test_examples = await self._generate_test_cases(num_examples)
        
        # Run evaluation
        start_time = time.time()
        predictions = []
        actuals = []
        response_times = []
        errors = 0
        
        for i, example in enumerate(test_examples):
            try:
                # Time the prediction
                pred_start = time.time()
                
                result = await agent.process_turn(
                    symptoms=example["user_message"],
                    nice_context="",
                    history=dspy.History(messages=[])
                )
                
                pred_end = time.time()
                response_times.append((pred_end - pred_start) * 1000)  # Convert to ms
                
                # Extract prediction
                predicted_outcome = result.get("outcome", "routine").lower()
                actual_outcome = example["expected_outcome"].lower()
                
                predictions.append(predicted_outcome)
                actuals.append(actual_outcome)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"   Evaluated {i + 1}/{num_examples} examples")
                
            except Exception as e:
                logger.warning(f"⚠️ Evaluation error on example {i}: {str(e)}")
                errors += 1
                # Add default values to maintain array consistency
                predictions.append("routine")
                actuals.append(example.get("expected_outcome", "routine").lower())
                response_times.append(2000)  # Default timeout response time
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions=predictions,
            actuals=actuals,
            response_times=response_times,
            total_time=total_time,
            errors=errors
        )
        
        logger.info(f"✅ Evaluation completed in {total_time:.1f}s")
        logger.info(f"   Overall accuracy: {metrics.overall_accuracy:.3f}")
        logger.info(f"   Emergency F1: {metrics.emergency_f1:.3f}")
        logger.info(f"   Missed emergencies: {metrics.missed_emergencies}")
        logger.info(f"   Avg response time: {metrics.avg_response_time_ms:.1f}ms")
        
        return self._metrics_to_dict(metrics)
    
    async def _generate_test_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate diverse test cases for evaluation"""
        
        # Import persona generator
        from test_patient_generator import PersonaGenerator
        
        generator = PersonaGenerator()
        personas = await generator.generate_diverse_batch(
            count=count,
            include_edge_cases=True
        )
        
        test_cases = []
        for persona in personas:
            case = persona.to_training_example()
            test_cases.append(case)
        
        return test_cases
    
    def _calculate_comprehensive_metrics(self, predictions: List[str], actuals: List[str],
                                       response_times: List[float], total_time: float,
                                       errors: int) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        
        # Ensure consistent class labels
        predictions = [self._normalize_outcome(p) for p in predictions]
        actuals = [self._normalize_outcome(a) for a in actuals]
        
        # Basic accuracy metrics
        correct = sum(1 for p, a in zip(predictions, actuals, strict=True) if p == a)
        overall_accuracy = correct / len(predictions) if predictions else 0.0
        
        # Create confusion matrix
        cm = confusion_matrix(actuals, predictions, labels=self.outcome_classes)
        cm_dict = self._confusion_matrix_to_dict(cm)
        
        # Calculate per-class metrics
        per_class = {}
        try:
            # Use sklearn's classification report
            report = classification_report(
                actuals, predictions, 
                labels=self.outcome_classes,
                output_dict=True,
                zero_division=0
            )
            
            for outcome in self.outcome_classes:
                if outcome in report:
                    per_class[outcome] = {
                        "precision": report[outcome]["precision"],
                        "recall": report[outcome]["recall"], 
                        "f1": report[outcome]["f1-score"],
                        "support": report[outcome]["support"]
                    }
                else:
                    per_class[outcome] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
                    
        except Exception as e:
            logger.warning(f"Could not calculate per-class metrics: {e}")
            for outcome in self.outcome_classes:
                per_class[outcome] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
        
        # Emergency-specific metrics
        emergency_metrics = self._calculate_emergency_metrics(predictions, actuals)
        
        # Safety metrics
        safety_metrics = self._calculate_safety_metrics(predictions, actuals)
        
        # Performance metrics
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # Overall F1 score
        overall_f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
        overall_precision = per_class.get("emergency", {}).get("precision", 0) + \
                           per_class.get("routine", {}).get("precision", 0) + \
                           per_class.get("self_care", {}).get("precision", 0)
        overall_precision /= 3
        
        overall_recall = per_class.get("emergency", {}).get("recall", 0) + \
                        per_class.get("routine", {}).get("recall", 0) + \
                        per_class.get("self_care", {}).get("recall", 0)
        overall_recall /= 3
        
        return EvaluationMetrics(
            overall_accuracy=overall_accuracy,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            
            emergency_precision=emergency_metrics["precision"],
            emergency_recall=emergency_metrics["recall"],
            emergency_f1=emergency_metrics["f1"],
            emergency_sensitivity=emergency_metrics["sensitivity"],
            emergency_specificity=emergency_metrics["specificity"],
            
            missed_emergencies=safety_metrics["missed_emergencies"],
            false_emergency_alerts=safety_metrics["false_emergencies"],
            overtriage_rate=safety_metrics["overtriage_rate"],
            undertriage_rate=safety_metrics["undertriage_rate"],
            
            avg_response_time_ms=avg_response_time,
            total_evaluations=len(predictions),
            evaluation_time_seconds=total_time,
            
            confusion_matrix_data=cm_dict,
            per_class_metrics=per_class
        )
    
    def _normalize_outcome(self, outcome: str) -> str:
        """Normalize outcome to standard classes"""
        outcome = outcome.lower().strip()
        
        # Map various outcome formats to standard classes
        if outcome in ["emergency", "emergency_route_to_doctor"]:
            return "emergency"
        elif outcome in ["routine", "routine_doctor", "routine_doctor_consultation"]:
            return "routine" 
        elif outcome in ["self_care", "self_care_advice"]:
            return "self_care"
        else:
            return "routine"  # Default fallback
    
    def _confusion_matrix_to_dict(self, cm: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Convert confusion matrix to dictionary format"""
        result = {}
        
        for i, actual_class in enumerate(self.outcome_classes):
            result[actual_class] = {}
            for j, predicted_class in enumerate(self.outcome_classes):
                result[actual_class][predicted_class] = int(cm[i, j]) if i < cm.shape[0] and j < cm.shape[1] else 0
        
        return result
    
    def _calculate_emergency_metrics(self, predictions: List[str], actuals: List[str]) -> Dict[str, float]:
        """Calculate emergency-specific metrics"""
        
        # Convert to binary classification (emergency vs non-emergency)
        pred_emergency = [1 if p == "emergency" else 0 for p in predictions]
        actual_emergency = [1 if a == "emergency" else 0 for a in actuals]
        
        # Calculate confusion matrix values
        tp = sum(1 for p, a in zip(pred_emergency, actual_emergency, strict=True) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(pred_emergency, actual_emergency, strict=True) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(pred_emergency, actual_emergency, strict=True) if p == 0 and a == 1)
        tn = sum(1 for p, a in zip(pred_emergency, actual_emergency, strict=True) if p == 0 and a == 0)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        sensitivity = recall  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
    
    def _calculate_safety_metrics(self, predictions: List[str], actuals: List[str]) -> Dict[str, Any]:
        """Calculate safety-related metrics"""
        
        missed_emergencies = 0
        false_emergencies = 0
        overtriage_cases = 0
        undertriage_cases = 0
        
        for pred, actual in zip(predictions, actuals, strict=True):
            # Missed emergencies (most critical)
            if actual == "emergency" and pred != "emergency":
                missed_emergencies += 1
            
            # False emergency alerts
            if pred == "emergency" and actual != "emergency":
                false_emergencies += 1
            
            # Overtriage (recommending higher level of care than needed)
            if ((actual == "self_care" and pred in ["routine", "emergency"]) or
                (actual == "routine" and pred == "emergency")):
                overtriage_cases += 1
            
            # Undertriage (recommending lower level of care than needed)  
            if ((actual == "routine" and pred == "self_care") or
                (actual == "emergency" and pred in ["routine", "self_care"])):
                undertriage_cases += 1
        
        total_cases = len(predictions)
        overtriage_rate = overtriage_cases / total_cases if total_cases > 0 else 0.0
        undertriage_rate = undertriage_cases / total_cases if total_cases > 0 else 0.0
        
        return {
            "missed_emergencies": missed_emergencies,
            "false_emergencies": false_emergencies,
            "overtriage_cases": overtriage_cases,
            "undertriage_cases": undertriage_cases,
            "overtriage_rate": overtriage_rate,
            "undertriage_rate": undertriage_rate
        }
    
    def _metrics_to_dict(self, metrics: EvaluationMetrics) -> Dict[str, float]:
        """Convert metrics object to dictionary"""
        return {
            "accuracy": metrics.overall_accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            
            "emergency_precision": metrics.emergency_precision,
            "emergency_recall": metrics.emergency_recall, 
            "emergency_f1": metrics.emergency_f1,
            "emergency_sensitivity": metrics.emergency_sensitivity,
            "emergency_specificity": metrics.emergency_specificity,
            
            "missed_emergencies": float(metrics.missed_emergencies),
            "false_emergency_alerts": float(metrics.false_emergency_alerts),
            "overtriage_rate": metrics.overtriage_rate,
            "undertriage_rate": metrics.undertriage_rate,
            
            "avg_response_time_ms": metrics.avg_response_time_ms,
            "total_evaluations": float(metrics.total_evaluations),
            "evaluation_time_seconds": metrics.evaluation_time_seconds
        }
    
    async def comparative_evaluation(self, agents: Dict[str, MedicalTriageAgent],
                                   num_examples: int = 50) -> Dict[str, Dict[str, float]]:
        """Compare multiple agents side by side"""
        
        logger.info(f"🔍 Running comparative evaluation on {len(agents)} agents")
        
        results = {}
        for agent_name, agent in agents.items():
            logger.info(f"   Evaluating {agent_name}...")
            metrics = await self.evaluate_agent(agent, num_examples)
            results[agent_name] = metrics
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(results)
        results["_comparison_summary"] = comparison_summary
        
        return results
    
    def _generate_comparison_summary(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate summary comparing different agents"""
        
        if len(results) < 2:
            return {"error": "Need at least 2 agents for comparison"}
        
        metrics_to_compare = [
            "accuracy", "emergency_f1", "missed_emergencies", 
            "avg_response_time_ms", "overtriage_rate"
        ]
        
        summary = {
            "best_overall_accuracy": None,
            "best_emergency_f1": None,
            "safest_agent": None,  # Lowest missed emergencies
            "fastest_agent": None,
            "metric_rankings": {}
        }
        
        # Find best performers for each metric
        for metric in metrics_to_compare:
            agent_scores = {
                agent: results[agent].get(metric, 0) 
                for agent in results if not agent.startswith("_")
            }
            
            if metric == "missed_emergencies":
                # Lower is better
                best_agent = min(agent_scores, key=agent_scores.get)
                summary["safest_agent"] = {
                    "agent": best_agent,
                    "missed_emergencies": agent_scores[best_agent]
                }
            elif metric == "avg_response_time_ms":
                # Lower is better  
                best_agent = min(agent_scores, key=agent_scores.get)
                summary["fastest_agent"] = {
                    "agent": best_agent,
                    "response_time_ms": agent_scores[best_agent]
                }
            else:
                # Higher is better
                best_agent = max(agent_scores, key=agent_scores.get)
                if metric == "accuracy":
                    summary["best_overall_accuracy"] = {
                        "agent": best_agent,
                        "accuracy": agent_scores[best_agent]
                    }
                elif metric == "emergency_f1":
                    summary["best_emergency_f1"] = {
                        "agent": best_agent,
                        "emergency_f1": agent_scores[best_agent]
                    }
            
            # Store rankings for all agents
            ranked_agents = sorted(agent_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=(metric not in ["missed_emergencies", "avg_response_time_ms"]))
            summary["metric_rankings"][metric] = ranked_agents
        
        return summary

# Test function
async def test_evaluation_suite():
    """Test the evaluation suite"""
    print("🧪 Testing Evaluation Suite...")
    
    # Import required modules
    from src.app2.core.dspy_config_v2 import get_llm_provider, ensure_dspy_configured
    ensure_dspy_configured("gemma3n:e4b")
    
    # Create test agent
    agent = MedicalTriageAgent(model_name="gemma3n:e4b")
    
    # Create evaluation suite
    suite = MedicalEvaluationSuite()
    
    # Run evaluation
    print("🔍 Running evaluation...")
    metrics = await suite.evaluate_agent(agent, num_examples=10)
    
    print("✅ Evaluation completed:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_evaluation_suite())
