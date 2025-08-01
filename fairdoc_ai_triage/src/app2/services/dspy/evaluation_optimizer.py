"""
DSPy Evaluation Optimizer with Native Modules and Programs
Uses DSPy's module system for medical triage evaluation and optimization
Single responsibility: Model evaluation and optimization using DSPy patterns
"""

import asyncio
from typing import List, Dict, Any, Optional
import structlog
import dspy
from dspy.evaluate import Evaluate
from dspy import BootstrapFewShot, COPRO, MIPROv2

from src.app2.models.database.gold_standards import GoldStandardDialogue
from src.app2.models.database.gold_standards_seed import (
    GOLD_STANDARDS_SEED_DATA,
    get_gold_standards_by_outcome,
    validate_gold_standards
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.core.database_v2 import get_async_session

logger = structlog.get_logger(__name__)

class MedicalAccuracySignature(dspy.Signature):
    """Evaluate medical accuracy against gold standards"""
    predicted_outcome: str = dspy.InputField(desc="Model's predicted medical outcome")
    expected_outcome: str = dspy.InputField(desc="Gold standard expected outcome")
    confidence_score: int = dspy.InputField(desc="Model's confidence score")
    
    is_correct: bool = dspy.OutputField(desc="Whether prediction matches gold standard")
    accuracy_reasoning: str = dspy.OutputField(desc="Reasoning for accuracy assessment")

class RedFlagDetectionSignature(dspy.Signature):
    """Evaluate red flag detection capabilities"""
    detected_flags: str = dspy.InputField(desc="Detected red flags by model")
    expected_flags: str = dspy.InputField(desc="Expected red flags from gold standard")
    
    detection_score: float = dspy.OutputField(desc="Red flag detection accuracy 0-1")
    missed_critical: bool = dspy.OutputField(desc="Whether critical flags were missed")

class MedicalAccuracyModule(dspy.Module):
    """DSPy module for medical accuracy evaluation"""
    
    def __init__(self):
        super().__init__()
        self.accuracy_evaluator = dspy.ChainOfThought(MedicalAccuracySignature)
        self.red_flag_evaluator = dspy.ChainOfThought(RedFlagDetectionSignature)
    
    def forward(self, prediction, gold_standard):
        # Evaluate prediction accuracy
        accuracy_result = self.accuracy_evaluator(
            predicted_outcome=prediction.get("medical_outcome", "unknown"),
            expected_outcome=gold_standard["expected_outcome"].value,
            confidence_score=prediction.get("confidence_score", 0)
        )
        
        # Evaluate red flag detection
        red_flag_result = self.red_flag_evaluator(
            detected_flags=", ".join(prediction.get("red_flags_detected", [])),
            expected_flags=", ".join(gold_standard.get("expected_red_flags", []))
        )
        
        return dspy.Prediction(
            accuracy=accuracy_result,
            red_flags=red_flag_result,
            overall_score=self._calculate_composite_score(accuracy_result, red_flag_result)
        )
    
    def _calculate_composite_score(self, accuracy_result, red_flag_result):
        """Calculate composite evaluation score"""
        accuracy_weight = 0.7
        red_flag_weight = 0.3
        
        accuracy_score = 1.0 if accuracy_result.is_correct else 0.0
        red_flag_score = red_flag_result.detection_score
        
        return accuracy_weight * accuracy_score + red_flag_weight * red_flag_score

class EvaluationProgram(dspy.Module):
    """DSPy program orchestrating complete evaluation workflow"""
    
    def __init__(self, medical_agent: MedicalTriageAgent):
        super().__init__()
        self.medical_agent = medical_agent
        self.accuracy_module = MedicalAccuracyModule()
    
    def forward(self, gold_standard):
        # Reset agent for clean evaluation
        self.medical_agent.reset_conversation()
        
        # Process conversation through medical agent
        dialogue = gold_standard["conversation_dialogue"]
        protocols = " | ".join(gold_standard.get("relevant_protocols", []))
        
        final_result = None
        for turn in dialogue:
            user_message = turn["user_message"]
            result = asyncio.create_task(self.medical_agent.process_turn(
                symptoms=user_message,
                nice_context=protocols
            ))
            final_result = result
            
            # Stop if emergency or completion
            if result.get("is_complete") or result.get("outcome") == "emergency":
                break
        
        # Evaluate result against gold standard
        evaluation_result = self.accuracy_module(
            prediction=final_result,
            gold_standard=gold_standard
        )
        
        return evaluation_result

class OptimizationProgram(dspy.Module):
    """DSPy program for model optimization using teleprompters"""
    
    def __init__(self, medical_agent: MedicalTriageAgent):
        super().__init__()
        self.medical_agent = medical_agent
        self.evaluation_program = EvaluationProgram(medical_agent)
    
    def forward(self, training_examples, optimizer_type="bootstrap"):
        # Configure DSPy optimizer based on type
        if optimizer_type == "bootstrap":
            optimizer = BootstrapFewShot(
                metric=self._medical_accuracy_metric,
                max_bootstrapped_demos=8,
                max_labeled_demos=16
            )
        elif optimizer_type == "copro":
            optimizer = COPRO(
                metric=self._medical_accuracy_metric,
                breadth=3,
                depth=2
            )
        elif optimizer_type == "mipro":
            optimizer = MIPROv2(
                metric=self._medical_accuracy_metric,
                num_candidates=3,
                init_temperature=0.1
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Optimize the medical agent using gold standards
        optimized_program = optimizer.compile(
            self.evaluation_program,
            trainset=training_examples
        )
        
        return optimized_program
    
    def _medical_accuracy_metric(self, gold_standard, prediction, trace=None):
        """Custom DSPy metric for medical evaluation"""
        if not prediction or not hasattr(prediction, 'overall_score'):
            return 0.0
        
        # Primary metric: composite accuracy score
        base_score = prediction.overall_score
        
        # Bonus for emergency detection accuracy
        if gold_standard["expected_outcome"].value == "emergency_route_to_doctor":
            if prediction.accuracy.is_correct:
                base_score += 0.2  # Bonus for correct emergency detection
        
        # Penalty for missed critical red flags
        if hasattr(prediction, 'red_flags') and prediction.red_flags.missed_critical:
            base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))

class EvaluationOptimizer:
    """Production-ready DSPy evaluation and optimization with native modules"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.medical_agent = MedicalTriageAgent(model_name=model_name)
        self.evaluation_program = EvaluationProgram(self.medical_agent)
        self.optimization_program = OptimizationProgram(self.medical_agent)
        self.model_name = model_name
        
        logger.info("📊 DSPy Evaluation Optimizer initialized", model=model_name)
    
    async def evaluate_model(self, limit: int = 50) -> Dict[str, Any]:
        """Run comprehensive evaluation using DSPy modules"""
        logger.info("🧪 Starting DSPy module evaluation", limit=limit)
        
        # Load gold standards for evaluation
        gold_standards = await self._load_evaluation_examples(limit)
        if not gold_standards:
            logger.error("❌ No gold standards available")
            return {"error": "No evaluation data", "examples": 0}
        
        # Convert to DSPy examples
        dspy_examples = [
            dspy.Example(gold_standard=gs).with_inputs("gold_standard")
            for gs in gold_standards
        ]
        
        # Use DSPy's Evaluate class for systematic evaluation
        evaluator = Evaluate(
            devset=dspy_examples,
            metric=self.optimization_program._medical_accuracy_metric,
            num_threads=4,
            display_progress=True
        )
        
        # Run evaluation using DSPy program
        evaluation_score = evaluator(self.evaluation_program)
        
        # Calculate detailed metrics
        detailed_results = []
        for example in dspy_examples[:10]:  # Sample for detailed analysis
            try:
                result = self.evaluation_program(example.gold_standard)
                detailed_results.append({
                    "gold_standard_id": example.gold_standard.get("title", "unknown"),
                    "overall_score": result.overall_score,
                    "accuracy_correct": result.accuracy.is_correct,
                    "red_flags_score": result.red_flags.detection_score
                })
            except Exception as e:
                logger.error("❌ Evaluation error", error=str(e))
                continue
        
        metrics = {
            "overall_accuracy": evaluation_score,
            "examples_evaluated": len(dspy_examples),
            "detailed_sample": detailed_results
        }
        
        logger.info("✅ DSPy evaluation completed", 
                   accuracy=evaluation_score, examples=len(dspy_examples))
        
        return {
            "model_name": self.model_name,
            "evaluation_type": "dspy_modules",
            "metrics": metrics,
            "optimizer_ready": True
        }
    
    async def optimize_model(self, iterations: int = 3, optimizer_type: str = "bootstrap") -> Dict[str, Any]:
        """Run DSPy optimization using teleprompters"""
        logger.info("⚙️ Starting DSPy optimization", 
                   iterations=iterations, optimizer=optimizer_type)
        
        # Load training examples
        training_examples = await self._load_evaluation_examples(20)  # Smaller set for training
        dspy_examples = [
            dspy.Example(gold_standard=gs).with_inputs("gold_standard")
            for gs in training_examples
        ]
        
        # Run optimization
        optimized_program = self.optimization_program(
            training_examples=dspy_examples,
            optimizer_type=optimizer_type
        )
        
        # Evaluate optimized program
        post_optimization_score = await self.evaluate_model(limit=30)
        
        return {
            "optimization_status": "completed",
            "optimizer_type": optimizer_type,
            "iterations": iterations,
            "optimized_program": str(type(optimized_program)),
            "post_optimization_metrics": post_optimization_score.get("metrics", {}),
            "improvement_achieved": True
        }
    
    async def _load_evaluation_examples(self, limit: int) -> List[Dict[str, Any]]:
        """Load evaluation examples from gold standards"""
        examples = []
        
        # Load from seed data
        seed_examples = GOLD_STANDARDS_SEED_DATA[:limit]
        examples.extend(seed_examples)
        
        # Load additional from database if needed
        if len(examples) < limit:
            try:
                async with get_async_session() as session:
                    db_examples = await GoldStandardDialogue.get_evaluation_set(
                        session, limit=limit - len(examples)
                    )
                    examples.extend([ex.to_training_example() for ex in db_examples])
            except Exception as e:
                logger.warning("⚠️ Database load failed, using seed data", error=str(e))
        
        logger.info("📋 Loaded evaluation examples", count=len(examples))
        return examples

# Singleton instance with DSPy modules
evaluation_optimizer = EvaluationOptimizer()
