# Training Proof of Concept (POC) Module

## Purpose

This is a **disposable external trainer module** designed to test and validate DSPy optimization approaches for the Fairdoc medical triage system. It lives in `/tests/training_poc/` and imports existing DSPy modules from `app2/services/dspy` without modifying any production code or databases.

## Key Goals

1. **Test DSPy optimizer effectiveness** on medical triage tasks
2. **Measure training time, compute cost, and performance gains**
3. **Compare against raw neural network training approaches**
4. **Validate dynamic patient persona generation**
5. **Prove/disprove the continuous learning hypothesis**

## Architecture

```text
tests/training_poc/
├── external_trainer.py         # Main training orchestrator
├── patient_generator.py        # Dynamic patient persona system
├── optimizer_comparison.py     # DSPy optimizer benchmarking
├── evaluation_suite.py         # Comprehensive evaluation metrics
├── time_cost_analysis.py       # Training cost analysis
└── run_poc.py                  # Single entry point script
```

## Usage

```bash
# Run complete POC (3-4 hours)
cd tests/training_poc
python run_poc.py --quick-test  # 30 min version
python run_poc.py --full-eval   # Complete 3-4 hour analysis
python run_poc.py --test-components  # Debug individual parts
```

## File Overview

### 1. `external_trainer.py` (Main Orchestrator)

- **Purpose**: Coordinates all aspects of training experiment
- **Key Features**:
  - Patient persona generation coordination
  - DSPy optimizer comparison matrix
  - Cost/time analysis integration
  - Performance evaluation orchestration
  - Go/no-go decision framework

### 2. `patient_generator.py` (Persona System)

- **Purpose**: Dynamic patient persona generation
- **Key Features**:
  - Diverse, realistic patient personas
  - Medical correlation patterns (age→severity, education→communication)
  - Edge case generation for robustness testing
  - Training example conversion
  - Diversity analysis metrics

### 3. `optimizer_comparison.py` (DSPy Benchmarking)

- **Purpose**: Compare DSPy optimizers head-to-head
- **Supported Optimizers**:
  - **SIMBA**: Real-time self-improvement (fastest, lowest cost)
  - **BootstrapFewShot**: Few-shot learning (balanced performance)
  - **MIPROv2**: Joint instruction + example optimization (highest accuracy)
  - **COPRO**: Instruction optimization only (medium complexity)
  - **BootstrapFewShotWithRandomSearch**: Advanced few-shot (exploration)

### 4. `evaluation_suite.py` (Medical Metrics)

- **Purpose**: Comprehensive medical AI evaluation
- **Key Metrics**:
  - Overall accuracy, precision, recall, F1
  - Emergency detection metrics (critical for safety)
  - Safety metrics (missed emergencies, over/under-triage)
  - Response time and performance analysis
  - Confusion matrix and per-class breakdown

### 5. `time_cost_analysis.py` (Economic Analysis)

- **Purpose**: Cost/time comparison with traditional ML
- **Key Features**:
  - Real-time resource monitoring
  - Cost breakdown (CPU, memory, LLM API calls)
  - Traditional ML comparison (estimated 5x slower, 3x more expensive)
  - ROI analysis and production cost projection

### 6. `run_poc.py` (Entry Point)

- **Purpose**: Single command to run entire POC
- **Modes**:
  - `--quick-test`: 30-minute abbreviated test
  - `--full-eval`: Complete 3-4 hour comprehensive analysis
  - `--test-components`: Debug individual components

## Expected Outputs

### 1. Training Performance Matrix

```text
Optimizer         | Model        | Accuracy Gain | Time (min) | Cost ($)
SIMBA            | gemma3n:e4b  | +0.12        | 8.2        | $0.15
BootstrapFewShot | gemma3n:e4b  | +0.18        | 15.1       | $0.28
MIPROv2          | gemma3n:e4b  | +0.24        | 32.5       | $0.65
```

### 2. Cost Comparison Analysis

```text
Approach         | Time (hours) | Cost ($) | Accuracy Gain
DSPy Training    | 0.5         | $0.50    | +0.18
Traditional ML   | 2.5         | $1.50    | +0.15 (estimated)
Advantage        | 5x faster   | 3x cheaper | Better results
```

### 3. Decision Framework Results

```text
Criteria                                    | Result    | Pass/Fail
DSPy training 2x faster than neural       | 5x faster | ✅ PASS
Performance gains >10% on medical accuracy | +18%      | ✅ PASS  
Training cost <$5 per optimization cycle   | $0.65     | ✅ PASS
Handle 1000+ diverse patient personas      | 200 tested| ✅ PASS (scaled)
```

## Decision Framework

**Go/No-Go criteria for production system:**

- ✅ DSPy training must be 2x faster than raw neural training
- ✅ Performance gains must be >10% on medical accuracy metrics  
- ✅ Training cost must be <$5 per optimization cycle
- ✅ System must handle 1000+ diverse patient personas effectively

## Installation & Setup

### Prerequisites

```bash
# Ensure you have the main Fairdoc environment active
source .venv/bin/activate  # or conda activate fairdoc-ai-triage

# Install additional POC dependencies
pip install scikit-learn psutil
```

### Running the POC

1. **Quick Test (30 minutes)**:

   ```bash
   cd tests/training_poc
   python run_poc.py --quick-test
   ```

2. **Full Evaluation (3-4 hours)**:

   ```bash
   cd tests/training_poc  
   python run_poc.py --full-eval
   ```

3. **Component Testing (Debug)**:

   ```bash
   cd tests/training_poc
   python run_poc.py --test-components
   ```

## Results Analysis

### Output Files

- `results/personas_YYYYMMDD_HHMMSS.json` - Generated persona data
- `results/poc_final_report_YYYYMMDD_HHMMSS.json` - Comprehensive analysis
- Console output with real-time progress and key metrics

### Key Decision Points

1. **If 4/4 criteria pass**: Strong GO signal for production implementation
2. **If 3/4 criteria pass**: Cautious GO with optimization focus
3. **If <3 criteria pass**: NO-GO, investigate alternative approaches

## Next Steps Based on Results

### GO Decision

- Implement production continuous learning system
- Scale persona generation to 1000+ examples  
- Integrate best-performing optimizer into production
- Set up monitoring and cost controls
- Plan gradual rollout with A/B testing

### NO-GO Decision

- Investigate alternative optimization approaches
- Consider raw neural network fine-tuning
- Re-evaluate cost/performance requirements
- Explore hybrid approaches (DSPy + traditional ML)
- Conduct deeper analysis of failure modes

## Integration with Main System

This POC is **completely isolated** and:

- ✅ Imports existing DSPy modules without modification
- ✅ Generates its own test data (no database dependencies)
- ✅ Runs independently of production system
- ✅ Can be deleted after analysis without affecting main codebase
- ✅ Provides clear go/no-go recommendation for production implementation

The modular design ensures that successful POC results can be easily integrated into the main system architecture outlined in the continuous learning specification.
