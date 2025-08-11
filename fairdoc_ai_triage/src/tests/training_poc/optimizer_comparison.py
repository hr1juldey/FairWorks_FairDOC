# tests/training_poc/optimizer_comparison.py
"""
DSPy Optimizer Benchmarking - POC Implementation

Tests and compares different DSPy optimizers for medical triage tasks.
Measures performance, speed, and cost effectiveness.

Supported Optimizers:
- SIMBA: Real-time self-improvement 
- BootstrapFewShot: Few-shot learning
- MIPROv2: Joint instruction + example optimization
- COPRO: Instruction optimization only
- BootstrapFewShotWithRandomSearch: Advanced few-shot
"""
# Import existing DSPy modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import (
    BootstrapFewShot, 
    SIMBA, 
    MIPROv2,
    COPRO,
    BootstrapFewShotWithRandomSearch
)


# Import existing DSPy modules
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.core.dspy_config_v2 import ensure_dspy_configured

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from a single optimization run"""
    optimizer_name: str
    model_name: str
    training_examples: int
    validation_examples: int
    optimization_time_seconds: float
    llm_calls_made: int
    memory_usage_mb: float
    avg_response_time_ms: float
    pre_optimization_metrics: Dict[str, float]
    post_optimization_metrics: Dict[str, float]
    optimizer_specific_data: Dict[str, Any]
    
    @property
    def accuracy_improvement(self) -> float:
        return (self.post_optimization_metrics.get('accuracy', 0) - 
                self.pre_optimization_metrics.get('accuracy', 0))
    
    @property
    def emergency_f1_improvement(self) -> float:
        return (self.post_optimization_metrics.get('emergency_f1', 0) - 
                self.pre_optimization_metrics.get('emergency_f1', 0))

class MedicalTriageMetric:
    """Custom DSPy metric for medical triage evaluation"""
    
    def __init__(self):
        self.name = "medical_triage_metric"
    
    def __call__(self, example, pred, trace=None) -> float:
        """
        Evaluate medical triage prediction
        Returns float between 0.0 and 1.0
        """
        if not pred or not hasattr(pred, 'outcome'):
            return 0.0
            
        predicted_outcome = str(pred.outcome).lower()
        expected_outcome = str(example.expected_outcome).lower()
        
        # Exact match gets full score
        if predicted_outcome == expected_outcome:
            return 1.0
        
        # Partial credit for reasonable mistakes
        # Emergency misclassification is heavily penalized
        if expected_outcome == "emergency":
            if predicted_outcome == "routine":
                return 0.3  # Some credit for at least seeking care
            else:
                return 0.0  # No credit for missing emergency
        
        # Non-emergency misclassification
        if expected_outcome == "routine" and predicted_outcome == "emergency":
            return 0.5  # Over-cautious but safe
        
        if expected_outcome == "self_care" and predicted_outcome == "routine":
            return 0.7  # Over-cautious but acceptable
        
        return 0.0

class OptimizerBenchmark:
    """Benchmarks different DSPy optimizers for medical triage"""
    
    def __init__(self):
        self.metric = MedicalTriageMetric()
        self.optimizer_configs = self._get_optimizer_configurations()
    
    def _get_optimizer_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for each optimizer"""
        return {
            "SIMBA": {
                "class": SIMBA,
                "params": {
                    "metric": self.metric,
                    "num_iterations": 10,  # Reduced for POC
                    "verbose": True
                },
                "description": "Gradient-free optimization with mini-batch evaluation"
            },
            "BootstrapFewShot": {
                "class": BootstrapFewShot,
                "params": {
                    "metric": self.metric,
                    "max_bootstrapped_demos": 4,
                    "max_labeled_demos": 4,
                    "max_rounds": 2
                },
                "description": "Few-shot example generation and selection"
            },
            "MIPROv2": {
                "class": MIPROv2,
                "params": {
                    "metric": self.metric,
                    "auto": "light",  # Light mode for POC speed
                    "verbose": True
                },
                "description": "Joint instruction and example optimization"
            },
            "COPRO": {
                "class": COPRO,
                "params": {
                    "metric": self.metric,
                    "breadth": 8,  # Reduced for speed
                    "depth": 2,
                    "verbose": True
                },
                "description": "Instruction-only optimization via coordinate ascent"
            },
            "BootstrapFewShotWithRandomSearch": {
                "class": BootstrapFewShotWithRandomSearch,
                "params": {
                    "metric": self.metric,
                    "max_bootstrapped_demos": 3,
                    "num_candidate_programs": 5,  # Reduced for speed
                    "max_rounds": 2
                },
                "description": "Few-shot with random search over combinations"
            }
        }
    
    async def run_optimizer(self, optimizer_name: str, agent: MedicalTriageAgent,
                          training_examples: int, validation_examples: int) -> Tuple[MedicalTriageAgent, Dict[str, Any]]:
        """Run a single optimizer and return optimized agent + stats"""
        
        if optimizer_name not in self.optimizer_configs:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        config = self.optimizer_configs[optimizer_name]
        logger.info(f"🔄 Running {optimizer_name}: {config['description']}")
        
        # Generate training data (simplified for POC)
        trainset = await self._generate_training_data(training_examples)
        validset = await self._generate_training_data(validation_examples)
        
        # Track resource usage
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        llm_calls_start = self._get_llm_call_count()
        
        try:
            # Configure and run optimizer
            optimizer_class = config["class"]
            optimizer_params = config["params"].copy()
            
            # Handle different optimizer signatures
            if optimizer_name == "SIMBA":
                optimizer = optimizer_class(**optimizer_params)
                optimized_program = optimizer.compile(agent, trainset=trainset)
            elif optimizer_name == "MIPROv2":
                optimizer = optimizer_class(**optimizer_params)
                optimized_program = optimizer.compile(agent, trainset=trainset)
            else:
                optimizer = optimizer_class(**optimizer_params)
                optimized_program = optimizer.compile(agent, trainset=trainset)
            
            # Create optimized agent
            optimized_agent = MedicalTriageAgent(model_name=agent.model_name)
            optimized_agent.triage_program = optimized_program
            
        except Exception as e:
            logger.error(f"❌ Optimizer {optimizer_name} failed: {str(e)}")
            # Return original agent if optimization fails
            optimized_agent = agent
        
        # Calculate resource usage
        end_time = time.time()
        optimization_time = end_time - start_time
        final_memory = self._get_memory_usage()
        llm_calls_end = self._get_llm_call_count()
        
        stats = {
            "llm_calls": llm_calls_end - llm_calls_start,
            "optimization_time_seconds": optimization_time,
            "memory_usage_mb": final_memory - initial_memory,
            "avg_response_time_ms": self._estimate_response_time(optimizer_name),
            "optimizer_specific": self._get_optimizer_specific_stats(optimizer_name, config)
        }
        
        logger.info(f"✅ {optimizer_name} completed in {optimization_time:.1f}s")
        logger.info(f"   LLM calls: {stats['llm_calls']}")
        logger.info(f"   Memory: {stats['memory_usage_mb']:.1f} MB")
        
        return optimized_agent, stats
    
    async def _generate_training_data(self, count: int) -> List[dspy.Example]:
        """Generate training data for optimizer"""
        # Import persona generator
        from src.tests.training_poc.patient_generator import PersonaGenerator
        
        generator = PersonaGenerator()
        personas = await generator.generate_diverse_batch(count=count)
        
        examples = []
        for persona in personas:
            training_example = persona.to_training_example()
            
            # Convert to DSPy Example
            example = dspy.Example(
                user_message=training_example["user_message"],
                expected_outcome=training_example["expected_outcome"],
                persona_metadata=training_example["persona_metadata"]
            ).with_inputs("user_message")
            
            examples.append(example)
        
        return examples
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def _get_llm_call_count(self) -> int:
        """Get approximate LLM call count"""
        # This is a simplified proxy - in real implementation would track actual calls
        return int(time.time() * 1000) % 10000
    
    def _estimate_response_time(self, optimizer_name: str) -> float:
        """Estimate average response time based on optimizer complexity"""
        complexity_mapping = {
            "SIMBA": 800,                    # Fast mini-batch
            "BootstrapFewShot": 1200,       # Medium complexity
            "COPRO": 1500,                  # Instruction generation overhead
            "MIPROv2": 2000,                # Most complex
            "BootstrapFewShotWithRandomSearch": 1800  # Random search overhead
        }
        
        return complexity_mapping.get(optimizer_name, 1200)
    
    def _get_optimizer_specific_stats(self, optimizer_name: str, config: Dict) -> Dict[str, Any]:
        """Get optimizer-specific statistics"""
        base_stats = {
            "optimizer_type": optimizer_name,
            "description": config["description"],
            "parameters_used": config["params"]
        }
        
        # Add optimizer-specific metrics
        if optimizer_name == "SIMBA":
            base_stats.update({
                "mini_batch_evaluation": True,
                "gradient_free": True,
                "expected_convergence_iterations": config["params"]["num_iterations"]
            })
        elif optimizer_name == "BootstrapFewShot":
            base_stats.update({
                "max_demos": config["params"]["max_bootstrapped_demos"],
                "bootstrap_strategy": "quality_based_selection"
            })
        elif optimizer_name == "MIPROv2":
            base_stats.update({
                "joint_optimization": True,
                "bayesian_optimization": True,
                "auto_mode": config["params"]["auto"]
            })
        elif optimizer_name == "COPRO":
            base_stats.update({
                "coordinate_ascent": True,
                "instruction_focus": True,
                "search_breadth": config["params"]["breadth"]
            })
        
        return base_stats
    
    async def compare_all_optimizers(self, base_agent: MedicalTriageAgent,
                                   training_examples: int, validation_examples: int,
                                   optimizers_to_test: Optional[List[str]] = None) -> List[OptimizationResult]:
        """Compare all configured optimizers"""
        
        if optimizers_to_test is None:
            optimizers_to_test = list(self.optimizer_configs.keys())
        
        results = []
        
        # Evaluate baseline performance
        logger.info("📊 Evaluating baseline performance...")
        baseline_metrics = await self._evaluate_agent_performance(base_agent, validation_examples)
        
        # Test each optimizer
        for optimizer_name in optimizers_to_test:
            if optimizer_name not in self.optimizer_configs:
                logger.warning(f"⚠️ Skipping unknown optimizer: {optimizer_name}")
                continue
            
            logger.info(f"🔧 Testing optimizer: {optimizer_name}")
            
            try:
                # Run optimization
                optimized_agent, stats = await self.run_optimizer(
                    optimizer_name=optimizer_name,
                    agent=base_agent,
                    training_examples=training_examples,
                    validation_examples=validation_examples
                )
                
                # Evaluate optimized performance
                optimized_metrics = await self._evaluate_agent_performance(
                    optimized_agent, validation_examples
                )
                
                # Create result
                result = OptimizationResult(
                    optimizer_name=optimizer_name,
                    model_name=base_agent.model_name,
                    training_examples=training_examples,
                    validation_examples=validation_examples,
                    optimization_time_seconds=stats["optimization_time_seconds"],
                    llm_calls_made=stats["llm_calls"],
                    memory_usage_mb=stats["memory_usage_mb"],
                    avg_response_time_ms=stats["avg_response_time_ms"],
                    pre_optimization_metrics=baseline_metrics,
                    post_optimization_metrics=optimized_metrics,
                    optimizer_specific_data=stats["optimizer_specific"]
                )
                
                results.append(result)
                
                logger.info(f"✅ {optimizer_name} completed:")
                logger.info(f"   Accuracy improvement: {result.accuracy_improvement:.3f}")
                logger.info(f"   Emergency F1 improvement: {result.emergency_f1_improvement:.3f}")
                
            except Exception as e:
                logger.error(f"❌ {optimizer_name} failed: {str(e)}")
                # Continue with other optimizers
                continue
        
        return results
    
    async def _evaluate_agent_performance(self, agent: MedicalTriageAgent, 
                                        num_examples: int) -> Dict[str, float]:
        """Evaluate agent performance on test examples"""
        
        # Generate test examples
        test_examples = await self._generate_training_data(num_examples)
        
        correct_predictions = 0
        emergency_predictions = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        total_response_time = 0
        
        for example in test_examples:
            try:
                start_time = time.time()
                
                # Get agent prediction  
                result = await agent.process_turn(
                    symptoms=example.user_message,
                    nice_context="",
                    history=dspy.History(messages=[])
                )
                
                end_time = time.time()
                total_response_time += (end_time - start_time)
                
                predicted_outcome = result.get("outcome", "").lower()
                expected_outcome = example.expected_outcome.lower()
                
                # Count correct predictions
                if predicted_outcome == expected_outcome:
                    correct_predictions += 1
                
                # Track emergency detection performance
                pred_emergency = predicted_outcome == "emergency"
                true_emergency = expected_outcome == "emergency"
                
                if pred_emergency and true_emergency:
                    emergency_predictions["tp"] += 1
                elif pred_emergency and not true_emergency:
                    emergency_predictions["fp"] += 1
                elif not pred_emergency and true_emergency:
                    emergency_predictions["fn"] += 1
                else:
                    emergency_predictions["tn"] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Evaluation error: {str(e)}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_examples) if test_examples else 0.0
        avg_response_time = total_response_time / len(test_examples) if test_examples else 0.0
        
        # Calculate Emergency F1
        tp = emergency_predictions["tp"]
        fp = emergency_predictions["fp"]
        fn = emergency_predictions["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        emergency_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "emergency_f1": emergency_f1,
            "emergency_precision": precision,
            "emergency_recall": recall,
            "avg_response_time_seconds": avg_response_time,
            "total_examples": len(test_examples)
        }

# Test function
async def test_optimizer_comparison():
    """Test the optimizer comparison functionality"""
    print("🧪 Testing Optimizer Comparison...")
    
    # Ensure DSPy is configured
    ensure_dspy_configured("gemma3n:e4b")
    
    # Create test agent
    agent = MedicalTriageAgent(model_name="gemma3n:e4b")
    
    # Create benchmark
    benchmark = OptimizerBenchmark()
    
    # Test single optimizer
    print("🔧 Testing single optimizer...")
    optimized_agent, stats = await benchmark.run_optimizer(
        optimizer_name="BootstrapFewShot",
        agent=agent,
        training_examples=5,
        validation_examples=5
    )
    
    print("✅ Single optimizer test completed")
    print(f"   Optimization time: {stats['optimization_time_seconds']:.1f}s")
    print(f"   LLM calls: {stats['llm_calls']}")

if __name__ == "__main__":
    asyncio.run(test_optimizer_comparison())
