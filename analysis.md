# DSPy Training System Analysis & Solutions

## Major Issues Identified

### 1. **CRITICAL: Async Event Loop Failures (7 instances)**

**Problem**: DSPy bootstrap training fails due to asyncio event loop issues

```text
ERROR: no running event loop. Set `provide_traceback=True` for traceback.
```

**Root Cause**:

- `asyncio.create_task()` called in evaluation without running event loop
- Mixed sync/async DSPy evaluation patterns

**Solution**:

```python
# In evaluation_optimizer.py, Line 102
# REPLACE:
result = asyncio.create_task(self.medical_agent.process_turn(
    symptoms=user_message,
    nice_context=protocols
))

# WITH:
if asyncio.get_running_loop():
    result = await self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    )
else:
    result = asyncio.run(self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    ))
```

### 2. **DATA LEAKAGE: Training/Testing Contamination**

**Problem**: Same data used for training and testing - 100% confidence scores indicate overfitting

**Evidence**:

- High confidence (90-100): 24 instances
- Expected outcomes match confidence patterns exactly
- No proper train/validation/test split

**Solution**:

```python
# Add to evaluation_optimizer.py
class DataSplitter:
    def __init__(self, test_ratio=0.2, val_ratio=0.2):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
    
    def split_data(self, data):
        import random
        random.shuffle(data)
        n = len(data)
        
        test_size = int(n * self.test_ratio)
        val_size = int(n * self.val_ratio)
        
        test_data = data[:test_size]
        val_data = data[test_size:test_size + val_size]
        train_data = data[test_size + val_size:]
        
        return train_data, val_data, test_data
```

### 3. **CONFIDENCE SCORE HARDCODING**

**Problem**: Confidence scores are hardcoded rather than learned

```python
confidence = 85 if emergency_result.is_emergency else 50  # Line 287
confidence = min(85, base_confidence + 15)  # Line 281
```

**Solution**: Replace hardcoded confidence with DSPy-learned confidence

```python
# Add confidence learning signature
class ConfidencePredictionSignature(dspy.Signature):
    """Learn confidence from medical reasoning quality"""
    medical_reasoning: str = dspy.InputField(desc="Quality of medical reasoning")
    red_flags_count: int = dspy.InputField(desc="Number of red flags detected")
    protocol_match_quality: float = dspy.InputField(desc="NICE protocol alignment")
    
    confidence_score: int = dspy.OutputField(desc="Learned confidence 0-100")
```

### 4. **MISSING PROPER TRAINING SEQUENCE**

**Problem**: No structured train → validate → test pipeline

**Current Flow**:

1. Load data ❌
2. Train immediately ❌
3. Test on same data ❌

**Required Flow**:

1. Load data ✅
2. Split train/val/test ✅
3. Train on train set ✅
4. Validate on val set ✅
5. Final test on test set ✅

## Solutions Implementation

### A. Fix Async Issues

```python
# evaluation_optimizer.py - Replace EvaluationProgram.forward()
class EvaluationProgram(dspy.Module):
    def forward(self, gold_standard):
        self.medical_agent.reset_conversation()
        dialogue = gold_standard["conversation_dialogue"]
        protocols = " | ".join(gold_standard.get("relevant_protocols", []))
        
        final_result = None
        for turn in dialogue:
            user_message = turn["user_message"]
            
            # FIX: Remove asyncio.create_task - use direct call
            try:
                if hasattr(self.medical_agent, '_loop') and self.medical_agent._loop.is_running():
                    # Already in async context
                    import asyncio
                    result = asyncio.get_event_loop().run_until_complete(
                        self.medical_agent.process_turn(
                            symptoms=user_message,
                            nice_context=protocols
                        )
                    )
                else:
                    # Sync call
                    result = self.medical_agent.process_turn_sync(
                        symptoms=user_message,
                        nice_context=protocols
                    )
            except Exception as e:
                # Fallback to sync evaluation
                result = self._fallback_sync_evaluation(user_message, protocols)
            
            final_result = result
            if result.get("is_complete") or result.get("outcome") == "emergency":
                break
                
        return self.accuracy_module(final_result, gold_standard)
```

### B. Add Proper Data Splitting

```python
# Add to evaluation_optimizer.py
async def evaluate_model_with_split(self, total_limit: int = 100) -> Dict[str, Any]:
    """Proper train/val/test evaluation"""
    
    # Load all data
    all_data = await self._load_evaluation_examples(total_limit)
    
    # Split data properly
    splitter = DataSplitter(test_ratio=0.3, val_ratio=0.2)  # 50% train, 20% val, 30% test
    train_data, val_data, test_data = splitter.split_data(all_data)
    
    logger.info("Data split completed", 
                train=len(train_data), 
                val=len(val_data), 
                test=len(test_data))
    
    # 1. Baseline evaluation (no training)
    baseline_results = await self._evaluate_on_dataset(test_data, "baseline")
    
    # 2. Train on training set
    trained_program = await self._train_model(train_data)
    
    # 3. Validate on validation set
    val_results = await self._evaluate_on_dataset(val_data, "validation", trained_program)
    
    # 4. Final test on test set
    test_results = await self._evaluate_on_dataset(test_data, "test", trained_program)
    
    return {
        "baseline_accuracy": baseline_results["accuracy"],
        "validation_accuracy": val_results["accuracy"], 
        "test_accuracy": test_results["accuracy"],
        "data_splits": {
            "train": len(train_data),
            "validation": len(val_data), 
            "test": len(test_data)
        },
        "overfitting_detected": val_results["accuracy"] - test_results["accuracy"] > 0.15
    }
```

### C. Remove Confidence Hardcoding

```python
# medical_agent.py - Replace confidence calculation
def _calculate_learned_confidence(self, medical_result, emergency_result) -> int:
    """Learn confidence from reasoning quality, not hardcode it"""
    
    # Extract reasoning quality features
    reasoning_length = len(str(medical_result.medical_reasoning))
    reasoning_words = len(str(medical_result.medical_reasoning).split())
    red_flags_count = len(emergency_result.critical_flags.split(',') if hasattr(emergency_result, 'critical_flags') else [])
    
    # Use DSPy to learn confidence
    confidence_predictor = dspy.ChainOfThought(ConfidencePredictionSignature)
    
    result = confidence_predictor(
        medical_reasoning=str(medical_result.medical_reasoning)[:500],
        red_flags_count=red_flags_count,
        protocol_match_quality=0.8  # From NICE lookup
    )
    
    try:
        confidence = int(result.confidence_score)
        return max(0, min(100, confidence))  # Ensure bounds
    except (ValueError, AttributeError):
        # Fallback based on reasoning quality
        if reasoning_length > 100 and reasoning_words > 20:
            return 75
        elif reasoning_length > 50:
            return 60
        else:
            return 45
```

### D. Structured Training Pipeline

```python
# Add to evaluation_optimizer.py
class StructuredTrainingPipeline:
    def __init__(self, medical_agent):
        self.medical_agent = medical_agent
        self.training_history = []
        
    async def run_full_pipeline(self, data_limit=100):
        """Run complete train->validate->test pipeline"""
        
        logger.info("🚀 Starting structured training pipeline")
        
        # Step 1: Load and split data
        all_data = await self._load_all_data(data_limit)
        train_data, val_data, test_data = self._split_data_deterministic(all_data)
        
        # Step 2: Baseline evaluation (untrained)
        baseline_metrics = await self._evaluate_baseline(test_data)
        
        # Step 3: Training phase
        for optimizer_type in ["bootstrap", "mipro", "copro"]:
            logger.info(f"🎯 Training with {optimizer_type}")
            
            trained_program = await self._train_with_optimizer(
                train_data, optimizer_type
            )
            
            # Validate
            val_metrics = await self._validate_program(
                trained_program, val_data
            )
            
            # Test
            test_metrics = await self._test_program(
                trained_program, test_data  
            )
            
            self.training_history.append({
                "optimizer": optimizer_type,
                "baseline_accuracy": baseline_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "overfitting": val_metrics["accuracy"] - test_metrics["accuracy"]
            })
            
        return self.training_history
    
    def _split_data_deterministic(self, data):
        """Deterministic split to ensure reproducibility"""
        import hashlib
        
        # Sort by hash of title for reproducible splits
        data_sorted = sorted(data, key=lambda x: hashlib.md5(
            x.get("title", "").encode()
        ).hexdigest())
        
        n = len(data_sorted)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)
        
        return (
            data_sorted[:train_end],       # 60% train
            data_sorted[train_end:val_end], # 20% validation  
            data_sorted[val_end:]          # 20% test
        )
```

## Implementation Priority

### Phase 1: Fix Critical Async Issues (Day 1)

1. Replace `asyncio.create_task` with proper async handling
2. Add sync fallback for DSPy evaluation
3. Test basic evaluation pipeline

### Phase 2: Data Splitting (Day 2)

1. Implement deterministic data splitting
2. Add baseline evaluation
3. Separate training/validation/test sets

### Phase 3: Remove Hardcoding (Day 3)

1. Replace confidence hardcoding with learning
2. Add reasoning quality features
3. Train confidence prediction model

### Phase 4: Structured Pipeline (Day 4)

1. Implement full train→validate→test pipeline
2. Add overfitting detection
3. Compare multiple optimizers properly

## Expected Results After Fixes

**Before Fixes**:

- 84.6% test pass rate (mostly trivial tests)
- 100% confidence scores (overfitting)
- Training failures due to async issues
- Data leakage between train/test

**After Fixes**:

- Proper baseline evaluation (likely 45-60% accuracy)
- Training improves validation accuracy to 70-80%
- Test accuracy remains realistic (65-75%)
- Overfitting detection and mitigation
- Confidence scores reflect true uncertainty

## Key Metrics to Track

1. **Baseline Accuracy**: Untrained model performance
2. **Training Improvement**: Validation accuracy gain
3. **Generalization Gap**: Validation - Test accuracy difference
4. **Confidence Calibration**: Predicted confidence vs actual accuracy
5. **Training Stability**: Consistent results across runs

This systematic approach will convert the current overfitted system into a robust, properly trained medical AI agent.
