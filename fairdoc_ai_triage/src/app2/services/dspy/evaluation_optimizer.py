"""
DSPy Evaluation Optimizer for Fairdoc AI V2
Comprehensive evaluation pipeline using gold standard conversations
for medical triage model optimization and performance measurement
"""

import asyncio
from typing import List, Dict, Any, Optional
import structlog
import dspy
from dspy.evaluate import SemanticF1

from src.app2.models.database.gold_standards import GoldStandardDialogue
from src.app2.models.database.gold_standards_seed import (
    GOLD_STANDARDS_SEED_DATA, 
    get_gold_standards_by_outcome,
    validate_gold_standards
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.core.database_v2 import get_async_session
from src.app2.models.schemas.medical_triage import MedicalOutcome

logger = structlog.get_logger(__name__)

class EvaluationOptimizer:
    """
    DSPy evaluation and optimization pipeline for medical triage
    Uses gold standard conversations for training and performance measurement
    """
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.agent = MedicalTriageAgent(model_name=model_name)
        self.metric = SemanticF1()
        self.model_name = model_name
        logger.info("📊 Evaluation Optimizer initialized", model=model_name)
    
    async def evaluate_model(self, limit: int = 50) -> Dict[str, Any]:
        """
        Run complete model evaluation against gold standards
        Returns comprehensive performance metrics
        """
        logger.info("🧪 Starting model evaluation", limit=limit)
        
        # Validate gold standards quality first
        validation_result = validate_gold_standards()
        if not validation_result.get("coverage_balanced", False):
            logger.warning("⚠️ Gold standards may not be balanced across outcomes")
        
        # Load gold standards (from seed data and database)
        gold_standards = await self._load_evaluation_examples(limit)
        if not gold_standards:
            logger.error("❌ No gold standards available for evaluation")
            return {"error": "No evaluation data available", "examples": 0}
        
        # Run evaluation across all examples
        results = []
        for gs in gold_standards:
            try:
                prediction = await self._evaluate_single_conversation(gs)
                validation_score = self._validate_prediction_against_gold_standard(gs, prediction)
                
                results.append({
                    "gold_standard_id": gs.get("title", "unknown"),
                    "prediction": prediction,
                    "validation_score": validation_score,
                    "expected_outcome": gs["expected_outcome"].value,
                    "actual_outcome": prediction.get("medical_outcome", "unknown")
                })
                
            except Exception as e:
                logger.error("❌ Evaluation error for example", 
                           title=gs.get("title", "unknown"), error=str(e))
                continue
        
        # Calculate aggregate metrics
        metrics = self._calculate_evaluation_metrics(results)
        
        logger.info("✅ Model evaluation completed", 
                   examples=len(results), 
                   accuracy=metrics.get("accuracy", 0))
        
        return {
            "model_name": self.model_name,
            "evaluation_timestamp": asyncio.get_event_loop().time(),
            "examples_evaluated": len(results),
            "gold_standards_used": len(gold_standards),
            "metrics": metrics,
            "detailed_results": results[:10]  # First 10 for debugging
        }
    
    async def optimize_model(self, iterations: int = 3) -> Dict[str, Any]:
        """
        Run DSPy optimization using gold standards
        Future implementation for model fine-tuning
        """
        logger.info("⚙️ Starting model optimization", iterations=iterations)
        
        # For now, just run evaluation
        # TODO: Implement actual DSPy optimization with gold standards
        evaluation_result = await self.evaluate_model()
        
        return {
            "optimization_status": "completed_evaluation_only",
            "iterations_planned": iterations,
            "baseline_metrics": evaluation_result.get("metrics", {}),
            "message": "Full optimization implementation pending"
        }
    
    async def _load_evaluation_examples(self, limit: int) -> List[Dict[str, Any]]:
        """Load gold standard examples from seed data and database"""
        examples = []
        
        # Load from seed data first
        seed_examples = GOLD_STANDARDS_SEED_DATA[:limit]
        examples.extend(seed_examples)
        
        # Load additional from database if available
        try:
            async with get_async_session() as session:
                db_examples = await GoldStandardDialogue.get_evaluation_set(
                    session, 
                    limit=max(0, limit - len(examples))
                )
                examples.extend([ex.to_training_example() for ex in db_examples])
        except Exception as e:
            logger.warning("⚠️ Could not load from database, using seed data only", 
                         error=str(e))
        
        logger.info("📋 Loaded evaluation examples", 
                   seed_count=len(seed_examples),
                   total_count=len(examples))
        
        return examples
    
    async def _evaluate_single_conversation(self, gold_standard: Dict[str, Any]) -> Dict[str, Any]:
        """Replay a single gold standard conversation through the agent"""
        self.agent.reset_conversation()
        
        dialogue = gold_standard["conversation_dialogue"]
        protocols = " | ".join(gold_standard.get("relevant_protocols", []))
        
        final_result = None
        
        for turn in dialogue:
            user_message = turn["user_message"]
            
            result = await self.agent.process_turn(
                symptoms=user_message,
                nice_context=protocols
            )
            
            final_result = result
            
            # Stop if agent declares completion or reaches emergency
            if result.get("is_complete") or result.get("outcome") == "emergency":
                break
        
        return {
            "medical_outcome": final_result.get("outcome", "inconclusive"),
            "confidence_score": final_result.get("confidence", 0),
            "turn_count": len(dialogue),
            "red_flags_detected": final_result.get("red_flags", []),
            "reasoning": final_result.get("reasoning", "")
        }
    
    def _validate_prediction_against_gold_standard(
        self, 
        gold_standard: Dict[str, Any], 
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model prediction against gold standard expectations"""
        expected_outcome = gold_standard["expected_outcome"].value
        actual_outcome = prediction.get("medical_outcome", "unknown")
        
        expected_flags = set(gold_standard.get("expected_red_flags", []))
        actual_flags = set(prediction.get("red_flags_detected", []))
        
        return {
            "outcome_correct": expected_outcome == actual_outcome,
            "confidence_adequate": prediction.get("confidence_score", 0) >= 
                                 gold_standard.get("minimum_confidence_threshold", 70),
            "red_flags_detected": len(expected_flags.intersection(actual_flags)) >= 
                                len(expected_flags) * 0.8,
            "turn_count_acceptable": prediction.get("turn_count", 0) <= 
                                   gold_standard.get("max_acceptable_turns", 10)
        }
    
    def _calculate_evaluation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate evaluation metrics from results"""
        if not results:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Overall accuracy
        correct_outcomes = sum(1 for r in results 
                             if r["validation_score"]["outcome_correct"])
        accuracy = correct_outcomes / len(results)
        
        # Emergency detection metrics
        emergency_results = [r for r in results 
                           if r["expected_outcome"] == "emergency_route_to_doctor"]
        emergency_detected = sum(1 for r in emergency_results 
                               if r["actual_outcome"] == "emergency")
        emergency_recall = (emergency_detected / len(emergency_results) 
                          if emergency_results else 0.0)
        
        # Confidence adequacy
        adequate_confidence = sum(1 for r in results 
                                if r["validation_score"]["confidence_adequate"])
        confidence_rate = adequate_confidence / len(results)
        
        return {
            "accuracy": round(accuracy, 3),
            "emergency_recall": round(emergency_recall, 3),
            "confidence_adequacy_rate": round(confidence_rate, 3),
            "total_examples": len(results)
        }

# Singleton instance for dependency injection
evaluation_optimizer = EvaluationOptimizer()
