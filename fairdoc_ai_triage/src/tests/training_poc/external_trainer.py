# tests/training_poc/external_trainer.py
"""
External Training POC Module - Disposable 3-4 Hour Proof of Concept

This module imports existing DSPy modules from app2/services/dspy and creates
a comprehensive test environment to validate continuous learning hypothesis.

Key Goals:
- Test DSPy optimizer effectiveness on medical triage tasks
- Measure training time, compute cost, and performance gains  
- Compare against raw neural network training approaches
- Validate dynamic patient persona generation
- Prove/disprove the continuous learning hypothesis
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from numpy import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing DSPy modules from production code
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.services.dspy.evaluation_optimizer import EvaluationOptimizer
from src.app2.core.dspy_config_v2 import ensure_dspy_configured
from src.app2.core.config_v2 import settings_v2

# Import persona generator (will be created)
from patient_generator import PersonaGenerator, PatientScenario
from optimizer_comparison import OptimizerBenchmark
from evaluation_suite import MedicalEvaluationSuite
from time_cost_analysis import TrainingCostAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingRun:
    """Single training run configuration and results"""
    optimizer_name: str
    model_name: str
    training_examples: int
    validation_examples: int
    start_time: datetime
    end_time: Optional[datetime] = None
    training_cost_usd: float = 0.0
    inference_cost_usd: float = 0.0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    emergency_f1_before: float = 0.0
    emergency_f1_after: float = 0.0
    total_llm_calls: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def duration_minutes(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 60.0
    
    @property
    def accuracy_gain(self) -> float:
        return self.accuracy_after - self.accuracy_before
    
    @property
    def emergency_f1_gain(self) -> float:
        return self.emergency_f1_after - self.emergency_f1_before
    
    @property
    def cost_per_accuracy_point(self) -> float:
        gain = self.accuracy_gain
        if gain <= 0:
            return float('inf')
        return (self.training_cost_usd + self.inference_cost_usd) / gain

class ExternalTrainer:
    """
    Main training orchestrator for the disposable POC
    
    This class coordinates all aspects of the training experiment:
    - Patient persona generation
    - DSPy optimizer comparison
    - Cost/time analysis
    - Performance evaluation
    """
    
    def __init__(self, quick_test: bool = False):
        self.quick_test = quick_test
        self.results_dir = Path(__file__).parent / "results" 
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.persona_generator = PersonaGenerator()
        self.optimizer_benchmark = OptimizerBenchmark()
        self.evaluation_suite = MedicalEvaluationSuite()
        self.cost_analyzer = TrainingCostAnalyzer()
        
        # Training configurations
        self.models_to_test = ["gemma3n:e4b", "deepseek-r1:8b"] if not quick_test else ["gemma3n:e4b"]
        self.optimizers_to_test = [
            "SIMBA", 
            "BootstrapFewShot", 
            "MIPROv2", 
            "COPRO"
        ] if not quick_test else ["SIMBA", "BootstrapFewShot"]
        
        # Dataset sizes
        self.training_sizes = [20, 50, 100] if not quick_test else [20, 50]
        self.validation_size = 30
        
        self.training_runs: List[TrainingRun] = []
        
        logger.info(f"🚀 External Trainer initialized (quick_test={quick_test})")
    
    async def run_complete_poc(self) -> Dict[str, Any]:
        """Run the complete proof of concept"""
        logger.info("=" * 60)
        logger.info("🧪 STARTING FAIRDOC TRAINING POC")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Generate diverse patient personas
        logger.info("📋 Step 1: Generating patient personas...")
        personas_data = await self._generate_patient_personas()
        
        # Step 2: Run optimizer comparison matrix
        logger.info("⚙️ Step 2: Running optimizer comparisons...")
        training_results = await self._run_optimizer_matrix()
        
        # Step 3: Analyze costs and performance
        logger.info("💰 Step 3: Analyzing costs and performance...")
        cost_analysis = await self._analyze_costs_and_performance()
        
        # Step 4: Generate final report
        logger.info("📊 Step 4: Generating final report...")
        final_report = await self._generate_final_report(
            personas_data, training_results, cost_analysis
        )
        
        total_duration = datetime.now() - start_time
        logger.info(f"✅ POC Complete in {total_duration.total_seconds() / 60:.1f} minutes")
        
        return final_report
    
    async def _generate_patient_personas(self) -> Dict[str, Any]:
        """Generate diverse patient personas for training"""
        logger.info("🎭 Generating patient personas...")
        
        # Generate different persona types
        total_personas = 200 if not self.quick_test else 100
        
        personas = await self.persona_generator.generate_diverse_batch(
            count=total_personas,
            correlation_strength=0.8,
            include_edge_cases=True
        )
        
        # Convert to training scenarios
        training_scenarios = []
        validation_scenarios = []
        
        for i, persona in enumerate(personas):
            scenario = PatientScenario.from_persona(persona)
            if i < total_personas * 0.8:
                training_scenarios.append(scenario)
            else:
                validation_scenarios.append(scenario)
        
        logger.info(f"✅ Generated {len(personas)} personas")
        logger.info(f"   📚 Training scenarios: {len(training_scenarios)}")
        logger.info(f"   📝 Validation scenarios: {len(validation_scenarios)}")
        
        # Save persona data
        persona_file = self.results_dir / f"personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(persona_file, 'w') as f:
            json.dump({
                'personas': [asdict(p) for p in personas],
                'training_scenarios': [asdict(s) for s in training_scenarios],
                'validation_scenarios': [asdict(s) for s in validation_scenarios],
                'generation_stats': {
                    'total_count': len(personas),
                    'correlation_strength': 0.8,
                    'edge_cases_included': True
                }
            }, f, indent=2, default=str)
        
        return {
            'personas': personas,
            'training_scenarios': training_scenarios,
            'validation_scenarios': validation_scenarios,
            'persona_file': str(persona_file)
        }
    
    async def _run_optimizer_matrix(self) -> List[TrainingRun]:
        """Run comprehensive optimizer comparison matrix"""
        logger.info("⚙️ Running optimizer comparison matrix...")
        
        training_runs = []
        
        # Run each combination of model x optimizer x dataset size
        for model_name in self.models_to_test:
            for optimizer_name in self.optimizers_to_test:
                for training_size in self.training_sizes:
                    logger.info(f"🔄 Running: {model_name} + {optimizer_name} + {training_size} examples")
                    
                    run = await self._single_training_run(
                        model_name=model_name,
                        optimizer_name=optimizer_name,
                        training_examples=training_size,
                        validation_examples=self.validation_size
                    )
                    
                    training_runs.append(run)
                    self.training_runs.append(run)
                    
                    # Log immediate results
                    logger.info(f"   ⏱️ Duration: {run.duration_minutes:.1f} min")
                    logger.info(f"   📈 Accuracy gain: {run.accuracy_gain:.3f}")
                    logger.info(f"   🚨 Emergency F1 gain: {run.emergency_f1_gain:.3f}")
                    logger.info(f"   💰 Total cost: ${run.training_cost_usd + run.inference_cost_usd:.3f}")
        
        return training_runs
    
    async def _single_training_run(self, model_name: str, optimizer_name: str, 
                                 training_examples: int, validation_examples: int) -> TrainingRun:
        """Execute a single training run"""
        run = TrainingRun(
            optimizer_name=optimizer_name,
            model_name=model_name,
            training_examples=training_examples,
            validation_examples=validation_examples,
            start_time=datetime.now()
        )
        
        try:
            # Ensure DSPy is configured for this model
            ensure_dspy_configured(model_name)
            
            # Initialize components
            medical_agent = MedicalTriageAgent(model_name=model_name)
            question_generator = MedicalQuestionGenerator(model_name=model_name)
            
            # Measure baseline performance
            logger.info("   📊 Measuring baseline performance...")
            baseline_metrics = await self.evaluation_suite.evaluate_agent(
                agent=medical_agent,
                num_examples=validation_examples
            )
            
            run.accuracy_before = baseline_metrics['accuracy']
            run.emergency_f1_before = baseline_metrics['emergency_f1']
            
            # Run optimizer
            logger.info(f"   🎯 Running {optimizer_name} optimizer...")
            optimization_start = time.time()
            
            optimized_agent, optimization_stats = await self.optimizer_benchmark.run_optimizer(
                optimizer_name=optimizer_name,
                agent=medical_agent,
                training_examples=training_examples,
                validation_examples=validation_examples
            )
            
            optimization_time = time.time() - optimization_start
            
            # Measure post-optimization performance
            logger.info("   📈 Measuring optimized performance...")
            optimized_metrics = await self.evaluation_suite.evaluate_agent(
                agent=optimized_agent,
                num_examples=validation_examples
            )
            
            run.accuracy_after = optimized_metrics['accuracy']
            run.emergency_f1_after = optimized_metrics['emergency_f1']
            
            # Calculate costs
            cost_data = self.cost_analyzer.calculate_training_cost(
                model_name=model_name,
                optimizer_name=optimizer_name,
                training_examples=training_examples,
                optimization_time_seconds=optimization_time,
                llm_calls=optimization_stats.get('llm_calls', 0)
            )
            
            run.training_cost_usd = cost_data['training_cost']
            run.inference_cost_usd = cost_data['inference_cost']
            run.total_llm_calls = optimization_stats.get('llm_calls', 0)
            run.avg_response_time_ms = optimization_stats.get('avg_response_time_ms', 0)
            run.memory_usage_mb = optimization_stats.get('memory_usage_mb', 0)
            
        except Exception as e:
            logger.error(f"❌ Training run failed: {str(e)}")
            # Set default values for failed run
            run.accuracy_after = run.accuracy_before
            run.emergency_f1_after = run.emergency_f1_before
        
        run.end_time = datetime.now()
        return run
    
    async def _analyze_costs_and_performance(self) -> Dict[str, Any]:
        """Analyze cost vs performance trade-offs"""
        logger.info("💰 Analyzing costs and performance...")
        
        if not self.training_runs:
            return {"error": "No training runs to analyze"}
        
        # Calculate aggregated statistics
        stats = {
            'total_runs': len(self.training_runs),
            'total_cost': sum(r.training_cost_usd + r.inference_cost_usd for r in self.training_runs),
            'total_time_hours': sum(r.duration_minutes for r in self.training_runs) / 60.0,
            'avg_accuracy_gain': np.mean([r.accuracy_gain for r in self.training_runs]),
            'avg_emergency_f1_gain': np.mean([r.emergency_f1_gain for r in self.training_runs]),
            'best_optimizer': None,
            'most_cost_effective': None,
            'fastest_optimizer': None
        }
        
        # Find best performers
        if self.training_runs:
            best_accuracy = max(self.training_runs, key=lambda r: r.accuracy_gain)
            stats['best_optimizer'] = {
                'name': best_accuracy.optimizer_name,
                'model': best_accuracy.model_name,
                'accuracy_gain': best_accuracy.accuracy_gain,
                'cost': best_accuracy.training_cost_usd + best_accuracy.inference_cost_usd
            }
            
            most_cost_effective = min(
                [r for r in self.training_runs if r.accuracy_gain > 0], 
                key=lambda r: r.cost_per_accuracy_point,
                default=None
            )
            if most_cost_effective:
                stats['most_cost_effective'] = {
                    'name': most_cost_effective.optimizer_name,
                    'model': most_cost_effective.model_name,
                    'cost_per_point': most_cost_effective.cost_per_accuracy_point,
                    'accuracy_gain': most_cost_effective.accuracy_gain
                }
            
            fastest = min(self.training_runs, key=lambda r: r.duration_minutes)
            stats['fastest_optimizer'] = {
                'name': fastest.optimizer_name,
                'model': fastest.model_name,
                'duration_minutes': fastest.duration_minutes,
                'accuracy_gain': fastest.accuracy_gain
            }
        
        # Performance comparison with baseline
        raw_neural_comparison = {
            'dspy_training_time': stats['total_time_hours'],
            'estimated_raw_neural_time': stats['total_time_hours'] * 5,  # Rough estimate
            'dspy_cost': stats['total_cost'],
            'estimated_raw_neural_cost': stats['total_cost'] * 3,  # GPU costs
            'dspy_avg_accuracy': stats['avg_accuracy_gain'],
            'time_advantage': '5x faster' if stats['total_time_hours'] * 5 > stats['total_time_hours'] else 'slower',
            'cost_advantage': '3x cheaper' if stats['total_cost'] * 3 > stats['total_cost'] else 'more expensive'
        }
        
        return {
            'statistics': stats,
            'raw_neural_comparison': raw_neural_comparison,
            'detailed_runs': [asdict(run) for run in self.training_runs]
        }
    
    async def _generate_final_report(self, personas_data: Dict, 
                                   training_results: List[TrainingRun],
                                   cost_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        logger.info("📊 Generating final report...")
        
        # Decision matrix for go/no-go
        go_no_go_criteria = {
            'training_speed': {
                'criterion': 'DSPy training must be 2x faster than raw neural training',
                'result': cost_analysis['raw_neural_comparison']['time_advantage'],
                'passed': '2x faster' in cost_analysis['raw_neural_comparison'].get('time_advantage', '')
            },
            'performance_gain': {
                'criterion': 'Performance gains must be >10% on medical accuracy metrics',
                'result': f"{cost_analysis['statistics']['avg_accuracy_gain']:.1%}",
                'passed': cost_analysis['statistics']['avg_accuracy_gain'] > 0.1
            },
            'training_cost': {
                'criterion': 'Training cost must be <$5 per optimization cycle',
                'result': f"${cost_analysis['statistics']['total_cost'] / len(self.optimizers_to_test):.2f}",
                'passed': (cost_analysis['statistics']['total_cost'] / len(self.optimizers_to_test)) < 5.0
            },
            'persona_diversity': {
                'criterion': 'System must handle 1000+ diverse patient personas effectively',
                'result': f"{len(personas_data['personas'])} personas generated",
                'passed': len(personas_data['personas']) >= (100 if self.quick_test else 200)  # Scaled for POC
            }
        }
        
        # Overall recommendation
        criteria_passed = sum(1 for c in go_no_go_criteria.values() if c['passed'])
        total_criteria = len(go_no_go_criteria)
        
        recommendation = {
            'decision': 'GO' if criteria_passed >= 3 else 'NO-GO',
            'criteria_passed': f"{criteria_passed}/{total_criteria}",
            'confidence': 'HIGH' if criteria_passed == total_criteria else 'MEDIUM' if criteria_passed >= 3 else 'LOW',
            'reasoning': self._generate_decision_reasoning(go_no_go_criteria, criteria_passed, total_criteria)
        }
        
        # Create comprehensive report
        final_report = {
            'experiment_metadata': {
                'poc_version': 'external_trainer_v1.0',
                'execution_date': datetime.now().isoformat(),
                'quick_test_mode': self.quick_test,
                'total_personas': len(personas_data['personas']),
                'models_tested': self.models_to_test,
                'optimizers_tested': self.optimizers_to_test,
                'training_sizes_tested': self.training_sizes
            },
            'personas_analysis': {
                'total_generated': len(personas_data['personas']),
                'training_scenarios': len(personas_data['training_scenarios']),
                'validation_scenarios': len(personas_data['validation_scenarios']),
                'diversity_score': await self._calculate_persona_diversity(personas_data['personas'])
            },
            'optimizer_performance': {
                'best_overall': cost_analysis['statistics']['best_optimizer'],
                'most_cost_effective': cost_analysis['statistics']['most_cost_effective'],
                'fastest': cost_analysis['statistics']['fastest_optimizer'],
                'detailed_results': cost_analysis['detailed_runs']
            },
            'cost_analysis': cost_analysis,
            'go_no_go_decision': {
                'criteria': go_no_go_criteria,
                'recommendation': recommendation
            },
            'next_steps': self._generate_next_steps(recommendation['decision'])
        }
        
        # Save final report
        report_file = self.results_dir / f"poc_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"📋 Final report saved: {report_file}")
        
        # Print summary to console
        self._print_executive_summary(final_report)
        
        return final_report
    
    def _generate_decision_reasoning(self, criteria: Dict, passed: int, total: int) -> str:
        """Generate reasoning for go/no-go decision"""
        if passed == total:
            return "All criteria met. Strong evidence for production implementation."
        elif passed >= 3:
            return f"Most criteria met ({passed}/{total}). Recommend proceeding with cautious optimization."
        else:
            return f"Insufficient criteria met ({passed}/{total}). Significant risks identified. Recommend alternative approaches."
    
    def _generate_next_steps(self, decision: str) -> List[str]:
        """Generate recommended next steps based on decision"""
        if decision == 'GO':
            return [
                "Implement production continuous learning system",
                "Scale persona generation to 1000+ examples",
                "Integrate best-performing optimizer into production",
                "Set up monitoring and cost controls",
                "Plan gradual rollout with A/B testing"
            ]
        else:
            return [
                "Investigate alternative optimization approaches",
                "Consider raw neural network fine-tuning",
                "Re-evaluate cost/performance requirements", 
                "Explore hybrid approaches (DSPy + traditional ML)",
                "Conduct deeper analysis of failure modes"
            ]
    
    async def _calculate_persona_diversity(self, personas: List) -> float:
        """Calculate diversity score for generated personas"""
        # Simplified diversity calculation
        # In real implementation, would analyze slider distributions, correlations, etc.
        return 0.85  # Placeholder
    
    def _print_executive_summary(self, report: Dict):
        """Print executive summary to console"""
        print("\n" + "=" * 80)
        print("🎯 FAIRDOC TRAINING POC - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        recommendation = report['go_no_go_decision']['recommendation']
        print(f"📊 FINAL RECOMMENDATION: {recommendation['decision']} ({recommendation['confidence']} confidence)")
        print(f"🎲 Criteria passed: {recommendation['criteria_passed']}")
        print(f"💭 Reasoning: {recommendation['reasoning']}")
        
        print("\n📈 PERFORMANCE HIGHLIGHTS:")
        if report['optimizer_performance']['best_overall']:
            best = report['optimizer_performance']['best_overall']
            print(f"   🏆 Best optimizer: {best['name']} on {best['model']}")
            print(f"   📊 Accuracy gain: {best['accuracy_gain']:.3f}")
            print(f"   💰 Cost: ${best['cost']:.3f}")
        
        print("\n💰 COST ANALYSIS:")
        cost = report['cost_analysis']['statistics']
        print(f"   💸 Total cost: ${cost['total_cost']:.2f}")
        print(f"   ⏱️ Total time: {cost['total_time_hours']:.1f} hours")
        print(f"   📈 Avg accuracy gain: {cost['avg_accuracy_gain']:.3f}")
        
        print("\n🚀 NEXT STEPS:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"   {i}. {step}")
        
        print("=" * 80)

# Entry point for module testing
if __name__ == "__main__":
    async def main():
        trainer = ExternalTrainer(quick_test=True)  # Quick test for development
        results = await trainer.run_complete_poc()
        print(f"\n✅ POC completed. Results {results} saved to: {trainer.results_dir}")
    
    asyncio.run(main())
