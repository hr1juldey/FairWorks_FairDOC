# tests/training_poc/run_poc.py
"""
Single Entry Point for Training POC

Orchestrates the complete 3-4 hour proof of concept for DSPy-based
continuous learning in medical triage systems.

Usage:
    python run_poc.py --quick-test    # 30 min version
    python run_poc.py --full-eval     # Complete 3-4 hour analysis
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import POC modules
from external_trainer import ExternalTrainer
from patient_generator import PersonaGenerator
from optimizer_comparison import OptimizerBenchmark
from evaluation_suite import MedicalEvaluationSuite
from time_cost_analysis import TrainingCostAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print POC banner"""
    print("\n" + "=" * 80)
    print("🚀 FAIRDOC AI TRAINING POC - DISPOSABLE EXTERNAL TRAINER")
    print("=" * 80)
    print("📋 Goals:")
    print("   • Test DSPy optimizer effectiveness on medical triage")
    print("   • Measure training time, compute cost, and performance gains")
    print("   • Compare against raw neural network training approaches")
    print("   • Validate dynamic patient persona generation")
    print("   • Prove/disprove continuous learning hypothesis")
    print("=" * 80)

def print_summary():
    """Print final summary"""
    print("\n" + "=" * 80)
    print("🎯 POC COMPLETE - CHECK RESULTS FOR GO/NO-GO DECISION")
    print("=" * 80)
    print("📊 Decision Criteria:")
    print("   ✓/❌ DSPy training must be 2x faster than raw neural training")
    print("   ✓/❌ Performance gains must be >10% on medical accuracy metrics")
    print("   ✓/❌ Training cost must be <$5 per optimization cycle")
    print("   ✓/❌ System must handle 1000+ diverse patient personas effectively")
    print("\n💡 Next Steps:")
    print("   📁 Check results/ directory for detailed analysis")
    print("   📋 Review final_report_*.json for comprehensive findings")
    print("   🚀 Make production implementation decision based on criteria")
    print("=" * 80)

async def quick_test():
    """Run 30-minute quick test version"""
    logger.info("🚀 Starting QUICK TEST mode (30 minutes)")
    
    print_banner()
    print("⚡ QUICK TEST MODE - 30 minute evaluation")
    
    # Initialize trainer in quick mode
    trainer = ExternalTrainer(quick_test=True)
    
    # Run abbreviated POC
    results = await trainer.run_complete_poc()
    
    return results

async def full_evaluation():
    """Run complete 3-4 hour evaluation"""
    logger.info("🚀 Starting FULL EVALUATION mode (3-4 hours)")
    
    print_banner()
    print("🔬 FULL EVALUATION MODE - Complete 3-4 hour analysis")
    
    # Initialize trainer in full mode
    trainer = ExternalTrainer(quick_test=False)
    
    # Run complete POC
    results = await trainer.run_complete_poc()
    
    return results

async def individual_tests():
    """Run individual component tests for debugging"""
    logger.info("🧪 Running individual component tests...")
    
    print("🧪 COMPONENT TESTING MODE")
    
    # Test persona generation
    print("\n1. Testing Patient Persona Generation...")
    generator = PersonaGenerator()
    personas = await generator.generate_diverse_batch(count=10)
    print(f"   ✅ Generated {len(personas)} personas")
    
    # Test optimizer comparison
    print("\n2. Testing Optimizer Comparison...")
    # Note: This requires proper DSPy setup
    try:
        from src.app2.core.dspy_config_v2 import ensure_dspy_configured
        from src.app2.services.dspy.medical_agent import MedicalTriageAgent
        
        ensure_dspy_configured("gemma3n:e4b")
        agent = MedicalTriageAgent(model_name="gemma3n:e4b")
        
        benchmark = OptimizerBenchmark()
        optimized_agent, stats = await benchmark.run_optimizer(
            optimizer_name="BootstrapFewShot",
            agent=agent,
            training_examples=5,
            validation_examples=5
        )
        print(f"   ✅ Optimizer test completed in {stats['optimization_time_seconds']:.1f}s")
        
    except Exception as e:
        print(f"   ❌ Optimizer test failed: {str(e)}")
    
    # Test evaluation suite
    print("\n3. Testing Evaluation Suite...")
    try:
        suite = MedicalEvaluationSuite()
        if 'agent' in locals():
            metrics = await suite.evaluate_agent(agent, num_examples=5)
            print(f"   ✅ Evaluation completed - Accuracy: {metrics.get('accuracy', 0):.3f}")
        else:
            print("   ⚠️ Skipping evaluation test (no agent available)")
    except Exception as e:
        print(f"   ❌ Evaluation test failed: {str(e)}")
    
    # Test cost analysis
    print("\n4. Testing Cost Analysis...")
    try:
        analyzer = TrainingCostAnalyzer()
        cost = analyzer.calculate_training_cost(
            model_name="gemma3n:e4b",
            optimizer_name="BootstrapFewShot", 
            training_examples=10,
            optimization_time_seconds=60,
            llm_calls=5
        )
        print(f"   ✅ Cost analysis completed - Total cost: ${cost['total_cost']:.4f}")
    except Exception as e:
        print(f"   ❌ Cost analysis test failed: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fairdoc AI Training POC - External Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_poc.py --quick-test     # Quick 30-min test
  python run_poc.py --full-eval      # Complete 3-4 hour analysis
  python run_poc.py --test-components # Test individual components
        """
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick 30-minute test version'
    )
    
    parser.add_argument(
        '--full-eval',
        action='store_true', 
        help='Run complete 3-4 hour evaluation'
    )
    
    parser.add_argument(
        '--test-components',
        action='store_true',
        help='Test individual components for debugging'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    mode_count = sum([args.quick_test, args.full_eval, args.test_components])
    if mode_count != 1:
        print("❌ Error: Please specify exactly one mode (--quick-test, --full-eval, or --test-components)")
        parser.print_help()
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.quick_test:
            results = asyncio.run(quick_test())
        elif args.full_eval:
            results = asyncio.run(full_evaluation())
        elif args.test_components:
            asyncio.run(individual_tests())
            results = {"status": "component_tests_completed"}
        
        print_summary()
        
        # Print key results
        if isinstance(results, dict) and 'go_no_go_decision' in results:
            decision = results['go_no_go_decision']['recommendation']
            print(f"\n🎯 FINAL DECISION: {decision['decision']} ({decision['confidence']} confidence)")
            print(f"📋 Criteria passed: {decision['criteria_passed']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ POC interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ POC failed: {str(e)}")
        print(f"\n❌ POC FAILED: {str(e)}")
        print("💡 Try running with --test-components to debug individual parts")
        sys.exit(1)

if __name__ == "__main__":
    main()
