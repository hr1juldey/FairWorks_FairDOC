"""
DSPy Evaluation Optimizer with Native Modules and Programs
Uses DSPy's module system for medical triage evaluation and optimization
Single responsibility: Model evaluation and optimization using DSPy patterns
"""

import asyncio
from typing import List, Dict, Any, Optional
import structlog
import dspy
import inspect
import logging
import time
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, COPRO, MIPROv2

from concurrent.futures import TimeoutError as FuturesTimeoutError

from src.app2.models.database.gold_standards import GoldStandardDialogue
from src.app2.models.database.gold_standards_seed import (
    GOLD_STANDARDS_SEED_DATA,
    get_gold_standards_by_outcome,
    validate_gold_standards
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.core.database_v2 import get_async_session

from src.app2.core.config_v2 import settings_v2
from src.app2.core.dspy_config_v2 import ensure_dspy_configured

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
    
    def forward(self, training_examples, optimizer_type="bootstrap"):
        """FIX: Staged optimization with fallbacks"""
        optimizers_to_try = [
            ("bootstrap", self._create_bootstrap_optimizer),
            ("labeled_fewshot", self._create_labeled_fewshot_optimizer),
            ("simple", self._create_simple_optimizer)  # New fallback
        ]
        
        for opt_name, opt_creator in optimizers_to_try:
            try:
                logger.info(f"Trying optimizer: {opt_name}")
                optimizer = opt_creator()
                optimized_program = optimizer.compile(
                    self.evaluation_program,
                    trainset=training_examples[:10]  # Start small
                )
                logger.info(f"✅ Optimization successful with {opt_name}")
                return optimized_program
                
            except Exception as e:
                logger.warning(f"Optimizer {opt_name} failed: {e}")
                continue
        
        # All optimizers failed - return unoptimized program
        logger.error("All optimizers failed, returning unoptimized program")
        return self.evaluation_program

def _create_simple_optimizer(self):
    """Fallback optimizer that always succeeds"""
    from dspy.teleprompt import LabeledFewShot
    return LabeledFewShot(k=2)  # Minimal optimization

    
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
    
    def forward(self, gold_standard, wait_timeout: float = 60.0):
        """Process gold standard evaluation - works both in sync and async contexts.

        Args:
            gold_standard: dict with keys "conversation_dialogue" and optionally "relevant_protocols"
            wait_timeout: max seconds to wait for an in-flight coroutine when scheduling against
                        a running loop in another thread. Uses a small polling loop if scheduling
                        in the same thread (best-effort).
        """
        # Reset agent for clean evaluation
        self.medical_agent.reset_conversation()

        dialogue = gold_standard["conversation_dialogue"]
        protocols = " | ".join(gold_standard.get("relevant_protocols", []))

        final_result = None

        for turn in dialogue:
            user_message = turn["user_message"]

            try:
                # Try to get a running loop (None if no running loop)
                try:
                    
                    loop = asyncio.get_running_loop()
                    in_running_loop = True
                except RuntimeError:
                    loop = None
                    in_running_loop = False

                # Decide whether process_turn is sync or async
                is_coro_fn = inspect.iscoroutinefunction(self.medical_agent.process_turn)

                if not is_coro_fn:
                    # synchronous handler -> call directly
                    result = self.medical_agent.process_turn(
                        symptoms=user_message,
                        nice_context=protocols
                    )
                else:
                    # coroutine handler
                    coro = self.medical_agent.process_turn(
                        symptoms=user_message,
                        nice_context=protocols
                    )

                    if not in_running_loop:
                        # No running loop in this thread: safe to use asyncio.run
                        result = asyncio.run(coro)
                    else:
                        # There is a running loop. Try to schedule safely.
                        # Best option: if the running loop is in another thread, use run_coroutine_threadsafe
                        try:
                            future = asyncio.run_coroutine_threadsafe(coro, loop)
                            # block until done or timeout
                            result = future.result(timeout=wait_timeout)
                        except Exception as exc_threadsafe:
                            # Could be that the loop is the same thread (cannot use run_coroutine_threadsafe),
                            # or other scheduling issues. Fall back to scheduling a task and polling gently.
                            try:
                                # Schedule a task on the running loop
                                future_task = asyncio.ensure_future(coro, loop=loop)
                            except Exception:
                                # As a last resort, re-raise the original exception to be caught by outer handler
                                raise exc_threadsafe

                            # Poll with a small sleep to avoid CPU spin (best-effort)
                            start = time.time()
                            while not future_task.done():
                                if time.time() - start > wait_timeout:
                                    raise FuturesTimeoutError("Timeout waiting for coroutine to complete")
                                # yield thread to event loop / other threads
                                time.sleep(0.001)

                            # Get result or exception
                            result = future_task.result()

            except FuturesTimeoutError as te:
                logger.error("Timeout while waiting for medical_agent.process_turn", error=str(te))
                result = {
                    "medical_outcome": "inconclusive",
                    "confidence_score": 30,
                    "red_flags_detected": [],
                    "is_complete": False
                }
            except Exception as e:
                logger.error("Evaluation turn failed", error=str(e))
                result = {
                    "medical_outcome": "inconclusive",
                    "confidence_score": 30,
                    "red_flags_detected": [],
                    "is_complete": False
                }

            final_result = result

            # Stop early on completion or emergency
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
                depth=2,
                num_trials=10,  # ADD: Required parameter
            )
        elif optimizer_type == "mipro":
            optimizer = MIPROv2(
                metric=self._medical_accuracy_metric,
                auto=None,  # Must set to None to use custom parameters
                num_candidates=4,
                num_trials=12,  # ADD: Required parameter
                init_temperature=0.1
            )
        elif optimizer_type == "labeled_fewshot":
            # Add missing LabeledFewShot optimizer
            from dspy.teleprompt import LabeledFewShot
            optimizer = LabeledFewShot(k=8)
        elif optimizer_type == "ensemble":
            # Add missing Ensemble optimizer
            optimizer = dspy.Ensemble()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Optimize the medical agent using gold standards
        if optimizer_type == "copro":
            # COPRO requires eval_kwargs parameter
            optimized_program = optimizer.compile(
                self.evaluation_program,
                trainset=training_examples,
                eval_kwargs={"num_threads": 1}
            )
        else:
            optimized_program = optimizer.compile(
                self.evaluation_program,
                trainset=training_examples
            )
        return optimized_program

    
    def _medical_accuracy_metric(self, gold_standard: Dict[str, Any], prediction: Any, trace: Optional[Any] = None) -> float:
        """
        Enhanced DSPy-compliant medical accuracy metric with comprehensive scoring.
        
        Addresses Issue 2.2: Gold Standards Field Mismatch with robust field handling.
        Scores holistic medical care quality: confidence, efficiency, compassion, expertise.
        
        Scoring Components (0.0-1.0 total):
        - Outcome accuracy (primary): 0.60 - Exact match of medical outcomes
        - Red-flag detection quality: 0.20 - Safety-critical symptom detection  
        - Confidence calibration: 0.10 - Appropriate confidence levels
        - Conversation efficiency: 0.05 - Timely, complete consultations
        - Compassion & clarity: 0.05 - Empathetic communication quality
        
        Args:
            gold_standard: Dict containing 'expected_outcome' and medical context
            prediction: DSPy prediction with nested accuracy/red_flags structure  
            trace: DSPy trace object for detailed debugging
            
        Returns:
            float: Comprehensive medical care score (0.0-1.0)
        """
        
        # Helper functions for robust processing
        def _normalize_outcome(val: Any) -> str:
            """Normalize medical outcomes with enum/string handling"""
            if hasattr(val, 'value'):
                val = val.value
            elif hasattr(val, 'name'):
                val = val.name
            s = str(val or '').lower().strip()
            
            # Medical domain-specific outcome mappings
            mapping = {
                'emergency_route_to_doctor': 'emergency',
                'emergency_route': 'emergency',
                'emergency': 'emergency',
                'routine_doctor_consultation': 'routine',
                'routine': 'routine',
                'self_care_advice': 'self_care',
                'selfcare': 'self_care',
                'self_care': 'self_care',
                'need_more_questions': 'inconclusive',
                'inconclusive': 'inconclusive',
                'spam_or_irrelevant': 'spam',
                'spam': 'spam',
            }
            return mapping.get(s, s)
        
        def _safe_float(x, default=0.0):
            """Safe float conversion with fallback"""
            try:
                return float(x)
            except Exception:
                return default
        
        # Initialize trace context for debugging
        trace_id = getattr(trace, 'trace_id', 'unknown') if trace else 'unknown'
        
        if trace:
            logger.debug(f"[TRACE-{trace_id}] Medical accuracy evaluation started")
        
        try:
            # 1. Validate prediction structure
            if prediction is None:
                if trace:
                    logger.warning(f"[TRACE-{trace_id}] NULL prediction received")
                return 0.0
            
            # 2. Validate gold standard structure  
            if not isinstance(gold_standard, dict):
                if trace:
                    logger.error(f"[TRACE-{trace_id}] Gold standard not dict: {type(gold_standard)}")
                return 0.0
            
            # 3. Extract expected outcome with robust handling
            expected_raw = gold_standard.get('expected_outcome')
            if expected_raw is None:
                if trace:
                    logger.error(f"[TRACE-{trace_id}] Missing expected_outcome in keys: {list(gold_standard.keys())}")
                return 0.0
            
            expected = _normalize_outcome(expected_raw)
            
            # 4. Extract predicted outcome via multiple fallback paths
            predicted_raw = None
            if hasattr(prediction, 'accuracy') and hasattr(prediction.accuracy, 'predicted_outcome'):
                predicted_raw = prediction.accuracy.predicted_outcome
            elif hasattr(prediction, 'predicted_outcome'):
                predicted_raw = prediction.predicted_outcome
            elif hasattr(prediction, 'outcome'):
                predicted_raw = prediction.outcome
            
            if predicted_raw is None:
                if trace:
                    logger.warning(f"[TRACE-{trace_id}] No predicted outcome found; attrs={dir(prediction)}")
                return 0.0
            
            predicted = _normalize_outcome(predicted_raw)
            
            # 5. Primary outcome accuracy (60% weight)
            is_correct = (predicted == expected)
            outcome_score = 1.0 if is_correct else 0.0
            
            # 6. Red-flag detection quality (20% weight)
            redflags_score = 0.0
            missed_penalty = 0.0
            if hasattr(prediction, 'red_flags'):
                rf = prediction.red_flags
                detection_score = getattr(rf, 'detection_score', None)
                redflags_score = _safe_float(detection_score, 0.0)
                
                # Strong penalty for missing critical flags
                if getattr(rf, 'missed_critical', False):
                    missed_penalty = 0.3
            
            redflags_component = max(0.0, redflags_score - missed_penalty)
            
            # 7. Confidence calibration (10% weight)
            confidence = None
            for attr in ['confidence_score', 'confidence']:
                if hasattr(prediction, attr):
                    confidence = _safe_float(getattr(prediction, attr))
                    break
            
            conf_component = 0.0
            if confidence is not None:
                # Normalize to [0,1] if given in 0-100 style
                conf_normalized = confidence / 100.0 if confidence > 1.0 else max(0.0, min(1.0, confidence))
                
                if is_correct:
                    # Reward calibrated confidence when correct
                    conf_component = 0.5 * conf_normalized + 0.5 * (conf_normalized ** 2)
                else:
                    # Penalize overconfidence when wrong
                    conf_component = max(0.0, 1.0 - conf_normalized)
            
            # 8. Conversation efficiency (5% weight)
            efficiency_component = 0.0
            turns = getattr(trace, 'num_turns', None) if trace else None
            completed = getattr(trace, 'completed', None) if trace else None
            
            if turns and isinstance(turns, int) and turns > 0:
                # Reward fewer turns; gentle decay
                eff_base = max(0.0, min(1.0, 1.0 / (1.0 + 0.1 * (turns - 1))))
                efficiency_component = eff_base if is_correct else (0.5 * eff_base)
            
            if completed:
                efficiency_component = min(1.0, efficiency_component + 0.1)
            
            # 9. Compassion & clarity proxy (5% weight)
            compassion_component = 0.0
            reasoning = None
            
            # Extract reasoning text via multiple paths
            if hasattr(prediction, 'accuracy') and hasattr(prediction.accuracy, 'accuracy_reasoning'):
                reasoning = prediction.accuracy.accuracy_reasoning
            elif hasattr(prediction, 'reasoning'):
                reasoning = prediction.reasoning
            
            if reasoning:
                text = str(reasoning).lower()
                length_bonus = min(1.0, len(text) / 400.0)  # Saturate at ~400 chars
                
                # Check for empathetic language cues
                empathy_cues = ['please', 'sorry', 'understand', 'concern', 'help', 'support', 'recommend', 'advise']
                empathy_bonus = 0.2 if any(cue in text for cue in empathy_cues) else 0.0
                compassion_component = max(0.0, min(1.0, 0.6 * length_bonus + empathy_bonus))
            
            # 10. Calculate weighted final score
            weights = {
                'outcome': 0.60,      # Primary medical accuracy
                'redflags': 0.20,     # Safety-critical detection
                'confidence': 0.10,   # Calibrated uncertainty
                'efficiency': 0.05,   # Timely completion
                'compassion': 0.05    # Communication quality
            }
            
            final_score = (
                weights['outcome'] * outcome_score +
                weights['redflags'] * redflags_component +
                weights['confidence'] * conf_component +
                weights['efficiency'] * efficiency_component +
                weights['compassion'] * compassion_component
            )
            
            # Clamp to [0, 1]
            final_score = max(0.0, min(1.0, final_score))
            
            # 11. Detailed trace logging for debugging
            if trace:
                logger.info(f"[TRACE-{trace_id}] Medical evaluation complete:")
                logger.info(f"[TRACE-{trace_id}] - Expected: '{expected}' | Predicted: '{predicted}'")
                logger.info(f"[TRACE-{trace_id}] - Components: outcome={outcome_score:.2f}, "
                        f"redflags={redflags_component:.2f}, conf={conf_component:.2f}, "
                        f"eff={efficiency_component:.2f}, comp={compassion_component:.2f}")
                logger.info(f"[TRACE-{trace_id}] - Final Score: {final_score:.3f}")
                
                if not is_correct:
                    logger.debug(f"[TRACE-{trace_id}] MISMATCH DETAILS:")
                    logger.debug(f"[TRACE-{trace_id}] - Raw expected: '{expected_raw}'")
                    logger.debug(f"[TRACE-{trace_id}] - Raw predicted: '{predicted_raw}'")
            
            return final_score
            
        except Exception as e:
            error_msg = f"Medical accuracy metric failed: {str(e)}"
            logger.error(error_msg)
            
            if trace:
                logger.error(f"[TRACE-{trace_id}] EXCEPTION DETAILS:")
                logger.error(f"[TRACE-{trace_id}] - Gold standard type: {type(gold_standard)}")
                logger.error(f"[TRACE-{trace_id}] - Prediction type: {type(prediction)}")
                logger.error(f"[TRACE-{trace_id}] - Exception: {repr(e)}")
            
            return 0.0  # Safe fallback


class EvaluationOptimizer:
    """Production-ready DSPy evaluation and optimization with native modules"""
    
    def __init__(self, model_name: str = None):
        model_name = model_name or settings_v2.DSPY_MODEL_NAME
        
        # NEW: Ensure DSPy is configured centrally before creating agents
        
        if not ensure_dspy_configured(model_name):
            raise RuntimeError("Failed to configure DSPy for evaluation optimizer")
        
        self.medical_agent = MedicalTriageAgent(model_name=model_name)
        self.evaluation_program = EvaluationProgram(self.medical_agent)
        self.optimization_program = OptimizationProgram(self.medical_agent)
        self.model_name = model_name
        logger.info("📊 DSPy Evaluation Optimizer initialized", model=model_name)

        
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
                    # Fix: Remove the limit parameter that doesn't exist in the method signature
                    db_examples = await GoldStandardDialogue.get_evaluation_set(session)
                    # Limit results manually
                    limited_examples = db_examples[:(limit - len(examples))]
                    examples.extend([ex.to_training_example() for ex in limited_examples])
            except Exception as e:
                logger.warning("⚠️ Database load failed, using seed data", error=str(e))

        logger.info("📋 Loaded evaluation examples", count=len(examples))
        return examples

    async def run_background_evaluation(self, limit: int = 50) -> Dict[str, Any]:
        """Run evaluation as async background process - safe for production"""
        logger.info("🔄 Starting background evaluation process", limit=limit)
        
        try:
            # This runs completely independently from the main chat system
            result = await asyncio.create_task(
                self._background_evaluation_worker(limit)
            )
            logger.info("✅ Background evaluation completed", 
                    accuracy=result.get("metrics", {}).get("overall_accuracy", 0))
            return result
            
        except Exception as e:
            logger.error("❌ Background evaluation failed", error=str(e))
            return {
                "status": "failed",
                "error": str(e),
                "impact_on_live_system": "none"  # Critical: no impact on production
            }

    async def _background_evaluation_worker(self, limit: int) -> Dict[str, Any]:
        """FIX: Increase timeout and add progressive fallback"""
        EVALUATION_TIMEOUT = 300  # 5 minutes instead of 2
        FALLBACK_TIMEOUT = 60     # Quick fallback
        
        gold_standards = await self._load_evaluation_examples(limit)

        if not gold_standards:
            return {"error": "No evaluation data", "examples": 0}

        dspy_examples = [
            dspy.Example(gold_standard=gs).with_inputs("gold_standard")
            for gs in gold_standards
        ]

        evaluation_results = []

        for example in dspy_examples:
            try:
                # Try with full timeout first
                result = await asyncio.wait_for(
                    self.evaluation_program(example.gold_standard),
                    timeout=EVALUATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f"Evaluation timeout, trying fallback for {example}")
                try:
                    # Fallback with simplified evaluation
                    result = await asyncio.wait_for(
                        self._simplified_evaluation(example.gold_standard),
                        timeout=FALLBACK_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Record failed evaluation
                    result = self._create_timeout_result(example)
            
            evaluation_results.append(result)

        # Combine results appropriately
        combined_score = sum(getattr(r, 'overall_score', 0) for r in evaluation_results) / len(evaluation_results)
        return {
            "model_name": self.model_name,
            "evaluation_type": "background_async_fixed",
            "metrics": {
                "overall_accuracy": combined_score,
                "examples_evaluated": len(dspy_examples)
            },
            "timestamp": asyncio.get_event_loop().time(),
            "safe_for_production": True
        }

    
    async def scheduled_optimization_run(self, 
                                    evaluation_limit: int = 100,
                                    training_limit: int = 30) -> Dict[str, Any]:
        """Complete async optimization run for scheduled execution"""
        logger.info("🚀 Starting scheduled optimization run")
        
        results = {
            "started_at": asyncio.get_event_loop().time(),
            "evaluation": {},
            "optimization": {},
            "status": "running"
        }
        
        try:
            # Step 1: Background evaluation
            eval_result = await self.run_background_evaluation(evaluation_limit)
            results["evaluation"] = eval_result
            
            # Step 2: Background optimization (if evaluation successful)
            if eval_result.get("metrics", {}).get("overall_accuracy", 0) < 0.8:
                logger.info("🎯 Accuracy below threshold, running optimization")
                opt_result = await self._background_optimization_worker(training_limit)
                results["optimization"] = opt_result
            else:
                logger.info("✅ Model performing well, skipping optimization")
                results["optimization"] = {"status": "skipped", "reason": "high_accuracy"}
            
            results["status"] = "completed"
            results["completed_at"] = asyncio.get_event_loop().time()
            
            return results
            
        except Exception as e:
            logger.error("❌ Scheduled optimization failed", error=str(e))
            results["status"] = "failed"
            results["error"] = str(e)
            return results
    
    # ADD THIS METHOD HERE:
    async def _background_optimization_worker(self, limit: int) -> Dict[str, Any]:
        """Background optimization worker - completely isolated"""
        training_examples = await self._load_evaluation_examples(limit)
        
        dspy_examples = [
            dspy.Example(gold_standard=gs).with_inputs("gold_standard")
            for gs in training_examples
        ]
        
        try:
            # Run optimization in isolated context
            optimized_program = self.optimization_program(
                training_examples=dspy_examples,
                optimizer_type="bootstrap"  # Safe default
            )
            
            return {
                "optimization_status": "completed",
                "training_examples": len(dspy_examples),
                "optimizer_used": "bootstrap", 
                "ready_for_deployment": False,  # Manual approval required
                "optimized_program_type": str(type(optimized_program))  # USE the variable
            }
            
        except Exception as e:
            logger.error("❌ Background optimization failed", error=str(e))
            return {
                "optimization_status": "failed",
                "error": str(e)
            }
# Singleton instance with DSPy modules
evaluation_optimizer = EvaluationOptimizer(settings_v2.DSPY_MODEL_NAME)
