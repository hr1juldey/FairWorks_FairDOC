# tests/training_poc/time_cost_analysis.py
"""
Training Cost & Time Analysis - POC Implementation

Analyzes the computational cost and time requirements for different
DSPy optimizers in the medical triage context. Provides cost comparison
with traditional neural network training approaches.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class CostEstimate:
    """Cost estimate for a training configuration"""
    
    # Time costs
    wall_clock_time_hours: float
    compute_time_hours: float      # Active compute time
    idle_time_hours: float         # Waiting/IO time
    
    # Computational costs  
    cpu_cost_usd: float
    memory_cost_usd: float
    llm_api_cost_usd: float        # API calls to LLM
    storage_cost_usd: float
    
    # Resource usage
    peak_memory_mb: float
    avg_cpu_usage_percent: float
    total_llm_calls: int
    total_tokens_processed: int
    
    @property
    def total_cost_usd(self) -> float:
        return (self.cpu_cost_usd + self.memory_cost_usd + 
                self.llm_api_cost_usd + self.storage_cost_usd)
    
    @property
    def cost_per_hour(self) -> float:
        if self.wall_clock_time_hours <= 0:
            return 0.0
        return self.total_cost_usd / self.wall_clock_time_hours

class TrainingCostAnalyzer:
    """Analyzes training costs and compares with traditional approaches"""
    
    def __init__(self):
        # Cost constants (realistic estimates)
        self.cpu_cost_per_hour = 0.05        # AWS c5.large equivalent
        self.memory_cost_per_gb_hour = 0.01  # Memory overhead
        self.llm_api_cost_per_1k_tokens = {
            "gemma3n:e4b": 0.0001,           # Local model, minimal cost
            "deepseek-r1:8b": 0.0002,        # Local model, slightly higher
            "gpt-4": 0.03,                   # OpenAI API cost
            "gpt-3.5-turbo": 0.002           # OpenAI API cost
        }
        self.storage_cost_per_gb_hour = 0.0001
        
        # Traditional ML training estimates (for comparison)
        self.neural_training_costs = {
            "gpu_cost_per_hour": 0.90,         # AWS p3.2xlarge
            "training_time_multiplier": 5,     # DSPy is ~5x faster  
            "data_preparation_hours": 8,       # Manual data prep time
            "hyperparameter_tuning_hours": 12  # Manual tuning time
        }
        
        self.baseline_metrics = {}
    
    def calculate_training_cost(self, model_name: str, optimizer_name: str,
                              training_examples: int, optimization_time_seconds: float,
                              llm_calls: int) -> Dict[str, float]:
        """Calculate comprehensive training cost"""
        
        # Convert time to hours
        time_hours = optimization_time_seconds / 3600.0
        
        # Estimate resource usage
        resource_usage = self._estimate_resource_usage(
            optimizer_name, training_examples, time_hours
        )
        
        # Calculate costs
        cpu_cost = time_hours * self.cpu_cost_per_hour
        memory_cost = (resource_usage["memory_gb"] * time_hours * 
                      self.memory_cost_per_gb_hour)
        
        # LLM API costs
        tokens_per_call = self._estimate_tokens_per_call(optimizer_name, model_name)
        total_tokens = llm_calls * tokens_per_call
        api_cost_per_1k = self.llm_api_cost_per_1k_tokens.get(model_name, 0.001)
        llm_cost = (total_tokens / 1000.0) * api_cost_per_1k
        
        # Storage cost (minimal for POC)
        storage_cost = resource_usage["storage_gb"] * time_hours * self.storage_cost_per_gb_hour
        
        return {
            "training_cost": cpu_cost + memory_cost,
            "inference_cost": llm_cost,
            "storage_cost": storage_cost,
            "total_cost": cpu_cost + memory_cost + llm_cost + storage_cost,
            "time_hours": time_hours,
            "resource_usage": resource_usage,
            "cost_breakdown": {
                "cpu": cpu_cost,
                "memory": memory_cost, 
                "llm_api": llm_cost,
                "storage": storage_cost
            }
        }
    
    def _estimate_resource_usage(self, optimizer_name: str, training_examples: int,
                               time_hours: float) -> Dict[str, float]:
        """Estimate resource usage based on optimizer characteristics"""
        
        # Base resource requirements
        base_memory_gb = 1.0
        base_storage_gb = 0.1
        
        # Optimizer-specific multipliers
        optimizer_multipliers = {
            "SIMBA": {"memory": 1.2, "storage": 1.0},             # Lightweight
            "BootstrapFewShot": {"memory": 1.5, "storage": 1.2},  # Moderate
            "MIPROv2": {"memory": 2.0, "storage": 1.5},           # Heavy optimization
            "COPRO": {"memory": 1.3, "storage": 1.1},             # Instruction generation
            "BootstrapFewShotWithRandomSearch": {"memory": 1.8, "storage": 1.4}
        }
        
        multiplier = optimizer_multipliers.get(optimizer_name, {"memory": 1.0, "storage": 1.0})
        
        # Scale with training examples
        example_scale = 1.0 + (training_examples / 1000.0)
        
        memory_gb = base_memory_gb * multiplier["memory"] * example_scale
        storage_gb = base_storage_gb * multiplier["storage"] * example_scale
        
        return {
            "memory_gb": memory_gb,
            "storage_gb": storage_gb,
            "cpu_utilization": 0.7  # Average CPU utilization
        }
    
    def _estimate_tokens_per_call(self, optimizer_name: str, model_name: str) -> int:
        """Estimate tokens per LLM call based on optimizer type"""
        
        base_tokens = {
            "gemma3n:e4b": 800,      # Smaller model, fewer tokens
            "deepseek-r1:8b": 1200,  # Larger context
            "gpt-4": 2000,           # Large context windows
            "gpt-3.5-turbo": 1500
        }
        
        model_base = base_tokens.get(model_name, 1000)
        
        # Optimizer complexity multipliers
        optimizer_multipliers = {
            "SIMBA": 0.8,                    # Simpler prompts
            "BootstrapFewShot": 1.0,        # Standard
            "MIPROv2": 1.5,                 # Complex instruction generation
            "COPRO": 1.2,                   # Instruction optimization
            "BootstrapFewShotWithRandomSearch": 1.1
        }
        
        multiplier = optimizer_multipliers.get(optimizer_name, 1.0)
        return int(model_base * multiplier)
    
    def compare_with_traditional_ml(self, dspy_costs: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compare DSPy training costs with traditional ML approaches"""
        
        if not dspy_costs:
            return {"error": "No DSPy costs provided"}
        
        # Aggregate DSPy costs
        total_dspy_cost = sum(cost["total_cost"] for cost in dspy_costs)
        total_dspy_time = sum(cost["time_hours"] for cost in dspy_costs)
        avg_dspy_cost = total_dspy_cost / len(dspy_costs)
        avg_dspy_time = total_dspy_time / len(dspy_costs)
        
        # Estimate traditional ML costs
        traditional_time = (
            self.neural_training_costs["data_preparation_hours"] +
            avg_dspy_time * self.neural_training_costs["training_time_multiplier"] +
            self.neural_training_costs["hyperparameter_tuning_hours"]
        )
        
        traditional_cost = (
            traditional_time * self.neural_training_costs["gpu_cost_per_hour"]
        )
        
        # Calculate savings/advantages
        time_savings = traditional_time - avg_dspy_time
        cost_savings = traditional_cost - avg_dspy_cost
        
        return {
            "dspy_approach": {
                "avg_cost_usd": avg_dspy_cost,
                "avg_time_hours": avg_dspy_time,
                "total_experiments": len(dspy_costs)
            },
            "traditional_ml": {
                "estimated_cost_usd": traditional_cost,
                "estimated_time_hours": traditional_time,
                "breakdown": {
                    "data_prep_hours": self.neural_training_costs["data_preparation_hours"],
                    "training_hours": avg_dspy_time * self.neural_training_costs["training_time_multiplier"],
                    "hyperparameter_tuning_hours": self.neural_training_costs["hyperparameter_tuning_hours"]
                }
            },
            "comparison": {
                "time_savings_hours": time_savings,
                "cost_savings_usd": cost_savings,
                "speed_advantage": f"{traditional_time / avg_dspy_time:.1f}x faster" if avg_dspy_time > 0 else "N/A",
                "cost_advantage": f"{cost_savings / traditional_cost * 100:.1f}% cheaper" if traditional_cost > 0 else "N/A",
                "roi_analysis": self._calculate_roi_analysis(avg_dspy_cost, traditional_cost, time_savings)
            }
        }
    
    def _calculate_roi_analysis(self, dspy_cost: float, traditional_cost: float, 
                               time_savings: float) -> Dict[str, Any]:
        """Calculate return on investment analysis"""
        
        # Assume time is worth $50/hour for developer
        developer_hourly_rate = 50.0
        time_value = time_savings * developer_hourly_rate
        
        total_savings = (traditional_cost - dspy_cost) + time_value
        roi_percentage = (total_savings / dspy_cost * 100) if dspy_cost > 0 else 0
        
        return {
            "total_savings_usd": total_savings,
            "roi_percentage": roi_percentage,
            "payback_period": "immediate" if total_savings > 0 else "negative",
            "developer_time_value_usd": time_value,
            "cost_differential_usd": traditional_cost - dspy_cost
        }
    
    def monitor_real_time_costs(self, duration_seconds: int = 60) -> Dict[str, float]:
        """Monitor real-time resource usage during training"""
        
        logger.info(f"📊 Starting {duration_seconds}s resource monitoring...")
        
        start_time = time.time()
        measurements = []
        
        while time.time() - start_time < duration_seconds:
            measurement = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
            }
            measurements.append(measurement)
            time.sleep(1)
        
        # Calculate averages
        avg_cpu = sum(m["cpu_percent"] for m in measurements) / len(measurements)
        avg_memory_mb = sum(m["memory_used_mb"] for m in measurements) / len(measurements)
        peak_memory_mb = max(m["memory_used_mb"] for m in measurements)
        
        return {
            "monitoring_duration_seconds": duration_seconds,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "measurements_taken": len(measurements),
            "estimated_cost_usd": self._calculate_monitoring_cost(avg_cpu, avg_memory_mb, duration_seconds)
        }
    
    def _calculate_monitoring_cost(self, avg_cpu: float, avg_memory_mb: float, 
                                 duration_seconds: int) -> float:
        """Calculate cost based on monitoring data"""
        
        hours = duration_seconds / 3600.0
        cpu_cost = hours * self.cpu_cost_per_hour * (avg_cpu / 100.0)
        memory_cost = hours * (avg_memory_mb / 1024.0) * self.memory_cost_per_gb_hour
        
        return cpu_cost + memory_cost
    
    def generate_cost_projection(self, optimizer_results: List[Dict[str, Any]], 
                               scale_factor: int = 10) -> Dict[str, Any]:
        """Project costs for scaled production deployment"""
        
        if not optimizer_results:
            return {"error": "No results to project from"}
        
        # Find best performing optimizer
        best_optimizer = max(optimizer_results, 
                           key=lambda r: r.get("accuracy_improvement", 0))
        
        base_cost = best_optimizer.get("training_cost", 0) + best_optimizer.get("inference_cost", 0)
        base_time = best_optimizer.get("optimization_time_seconds", 0) / 3600.0
        
        # Project scaling
        scaled_cost = base_cost * scale_factor
        scaled_time = base_time * scale_factor * 0.8  # Some efficiency gains at scale
        
        # Monthly/yearly projections
        monthly_cost = scaled_cost * 30  # Assuming daily retraining
        yearly_cost = monthly_cost * 12
        
        return {
            "production_projection": {
                "best_optimizer": best_optimizer.get("optimizer_name", "unknown"),
                "base_experiment": {
                    "cost_usd": base_cost,
                    "time_hours": base_time
                },
                "scaled_deployment": {
                    "scale_factor": scale_factor,
                    "cost_usd": scaled_cost,
                    "time_hours": scaled_time
                },
                "operational_costs": {
                    "monthly_cost_usd": monthly_cost,
                    "yearly_cost_usd": yearly_cost,
                    "cost_per_patient_interaction": scaled_cost / (scale_factor * 100)  # Assume 100 interactions per experiment
                }
            },
            "cost_optimization_recommendations": [
                "Use SIMBA for real-time optimization (lowest cost)",
                "Implement caching for repeated persona patterns",
                "Consider model quantization for inference cost reduction",
                "Implement adaptive retraining based on performance drift"
            ]
        }

# Test function
def test_cost_analysis():
    """Test cost analysis functionality"""
    print("🧪 Testing Cost Analysis...")
    
    analyzer = TrainingCostAnalyzer()
    
    # Test single cost calculation
    cost = analyzer.calculate_training_cost(
        model_name="gemma3n:e4b",
        optimizer_name="BootstrapFewShot",
        training_examples=50,
        optimization_time_seconds=300,  # 5 minutes
        llm_calls=20
    )
    
    print("✅ Single cost calculation:")
    print(f"   Total cost: ${cost['total_cost']:.4f}")
    print(f"   Time: {cost['time_hours']:.2f} hours")
    print(f"   Cost breakdown: {cost['cost_breakdown']}")
    
    # Test comparison with traditional ML
    comparison = analyzer.compare_with_traditional_ml([cost])
    
    print("\n💰 Comparison with traditional ML:")
    print(f"   DSPy cost: ${comparison['dspy_approach']['avg_cost_usd']:.4f}")
    print(f"   Traditional ML cost: ${comparison['traditional_ml']['estimated_cost_usd']:.2f}")
    print(f"   Savings: {comparison['comparison']['cost_advantage']}")
    print(f"   Speed advantage: {comparison['comparison']['speed_advantage']}")

if __name__ == "__main__":
    test_cost_analysis()
