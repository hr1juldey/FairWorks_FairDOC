"""
Machine Learning models for Fairdoc Medical AI Backend.
Handles ML predictions, model metadata, training metrics, and model lifecycle management.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import Field
from enum import Enum

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin, UUIDMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

# ============================================================================
# ML MODEL ENUMS AND TYPES
# ============================================================================

class ModelType(str, Enum):
    """Types of ML models in the system."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"

class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"

class ModelPurpose(str, Enum):
    """Purpose of the ML model."""
    TRIAGE_CLASSIFICATION = "triage_classification"
    RISK_ASSESSMENT = "risk_assessment"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    BIAS_DETECTION = "bias_detection"
    CONVERSATION_AI = "conversation_ai"
    IMAGE_ANALYSIS = "image_analysis"
    VITAL_SIGNS_PREDICTION = "vital_signs_prediction"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"

class TrainingStatus(str, Enum):
    """Training job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class DeploymentEnvironment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"

class DatasetType(str, Enum):
    """Types of datasets used for training."""
    SYNTHETIC = "synthetic"
    REAL_ANONYMIZED = "real_anonymized"
    SIMULATION = "simulation"
    AUGMENTED = "augmented"
    FEDERATED = "federated"

# ============================================================================
# MODEL METADATA AND VERSIONING
# ============================================================================

class ModelMetadata(BaseEntity, ValidationMixin, MetadataMixin):
    """Comprehensive model metadata and versioning."""
    
    # Model identification
    model_name: str = Field(..., max_length=100, description="Human-readable model name")
    model_version: str = Field(..., description="Semantic version (e.g., 1.2.3)")
    model_type: ModelType
    model_purpose: ModelPurpose
    
    # Model architecture details
    architecture_description: str = Field(..., max_length=1000)
    framework: str = Field(..., description="ML framework used (pytorch, tensorflow, etc.)")
    framework_version: str = Field(..., description="Framework version")
    model_size_mb: float = Field(..., ge=0, description="Model size in megabytes")
    parameter_count: int = Field(..., ge=0, description="Number of trainable parameters")
    
    # Training configuration
    training_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    optimization_algorithm: str = Field(..., description="Optimization algorithm used")
    
    # Data requirements
    input_schema: Dict[str, Any] = Field(..., description="Expected input data schema")
    output_schema: Dict[str, Any] = Field(..., description="Model output schema")
    feature_names: List[str] = Field(default_factory=list)
    target_variable: Optional[str] = None
    
    # Performance requirements
    inference_time_ms: Optional[float] = Field(None, ge=0, description="Expected inference time")
    memory_requirements_mb: Optional[float] = Field(None, ge=0, description="Memory requirements")
    gpu_required: bool = Field(default=False)
    min_gpu_memory_gb: Optional[float] = Field(None, ge=0)
    
    # Model status and lifecycle
    status: ModelStatus = Field(default=ModelStatus.TRAINING)
    deployment_environment: Optional[DeploymentEnvironment] = None
    deployment_url: Optional[str] = None
    
    # Training data information
    training_dataset_size: int = Field(..., ge=0)
    validation_dataset_size: int = Field(..., ge=0)
    test_dataset_size: int = Field(..., ge=0)
    dataset_type: DatasetType
    
    # Bias and fairness metadata
    bias_tested: bool = Field(default=False)
    fairness_metrics_computed: bool = Field(default=False)
    demographic_groups_tested: List[str] = Field(default_factory=list)
    
    # Regulatory and compliance
    regulatory_approved: bool = Field(default=False)
    approval_reference: Optional[str] = None
    ethical_review_completed: bool = Field(default=False)
    
    # Documentation
    model_documentation_url: Optional[str] = None
    research_paper_reference: Optional[str] = None
    changelog: List[str] = Field(default_factory=list)
    
    def add_changelog_entry(self, entry: str):
        """Add entry to model changelog."""
        timestamp = datetime.utcnow().isoformat()
        self.changelog.append(f"{timestamp}: {entry}")
        self.update_timestamp()
    
    def update_status(self, new_status: ModelStatus, reason: str = ""):
        """Update model status with audit trail."""
        old_status = self.status
        self.status = new_status
        self.add_changelog_entry(f"Status changed from {old_status} to {new_status}. Reason: {reason}")
        self.update_timestamp()
    
    def add_demographic_test(self, demographic: str):
        """Add demographic group to tested list."""
        if demographic not in self.demographic_groups_tested:
            self.demographic_groups_tested.append(demographic)
            self.update_timestamp()

class ModelVersion(TimestampMixin, UUIDMixin, ValidationMixin):
    """Model version tracking."""
    model_metadata_id: UUID = Field(..., description="Reference to model metadata")
    version_number: str = Field(..., description="Version number")
    parent_version_id: Optional[UUID] = Field(None, description="Parent version for tracking evolution")
    
    # Version-specific details
    changes_summary: str = Field(..., max_length=500)
    breaking_changes: bool = Field(default=False)
    performance_improvement: Optional[float] = Field(None, description="Performance improvement percentage")
    
    # Deployment tracking
    deployment_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    active_deployments: int = Field(default=0, ge=0)
    
    def deploy_version(self):
        """Mark version as deployed."""
        self.deployment_date = datetime.utcnow()
        self.active_deployments += 1
        self.update_timestamp()
    
    def deprecate_version(self):
        """Mark version as deprecated."""
        self.deprecation_date = datetime.utcnow()
        self.update_timestamp()

# ============================================================================
# TRAINING AND EVALUATION MODELS
# ============================================================================

class TrainingJob(BaseEntity, ValidationMixin, MetadataMixin):
    """ML model training job tracking."""
    
    # Job identification
    job_name: str = Field(..., max_length=100)
    model_metadata_id: UUID = Field(..., description="Reference to model being trained")
    training_status: TrainingStatus = Field(default=TrainingStatus.QUEUED)
    
    # Training configuration
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    dataset_config: Dict[str, Any] = Field(..., description="Dataset configuration")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used")
    
    # Resource allocation
    compute_resources: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration_minutes: Optional[int] = Field(None, ge=0)
    actual_duration_minutes: Optional[int] = Field(None, ge=0)
    
    # Training progress
    current_epoch: int = Field(default=0, ge=0)
    total_epochs: int = Field(..., ge=1)
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Timing
    training_started_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None
    
    # Results
    final_metrics: Dict[str, float] = Field(default_factory=dict)
    model_artifacts_path: Optional[str] = None
    logs_path: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    
    def start_training(self):
        """Mark training job as started."""
        self.training_status = TrainingStatus.RUNNING
        self.training_started_at = datetime.utcnow()
        self.update_timestamp()
    
    def complete_training(self, final_metrics: Dict[str, float], artifacts_path: str):
        """Mark training job as completed."""
        self.training_status = TrainingStatus.COMPLETED
        self.training_completed_at = datetime.utcnow()
        self.final_metrics = final_metrics
        self.model_artifacts_path = artifacts_path
        
        if self.training_started_at:
            duration = self.training_completed_at - self.training_started_at
            self.actual_duration_minutes = int(duration.total_seconds() / 60)
        
        self.update_timestamp()
    
    def fail_training(self, error_message: str):
        """Mark training job as failed."""
        self.training_status = TrainingStatus.FAILED
        self.error_message = error_message
        self.training_completed_at = datetime.utcnow()
        self.update_timestamp()

class ModelEvaluation(BaseEntity, ValidationMixin, MetadataMixin):
    """Model evaluation results and metrics."""
    
    # Evaluation context
    model_metadata_id: UUID = Field(..., description="Reference to evaluated model")
    evaluation_dataset_id: UUID = Field(..., description="Dataset used for evaluation")
    evaluation_type: Literal["training", "validation", "test", "production"] = "test"
    
    # Dataset characteristics for bias analysis
    dataset_size: int = Field(..., ge=0)
    gender_distribution: Dict[Gender, int] = Field(default_factory=dict)
    ethnicity_distribution: Dict[Ethnicity, int] = Field(default_factory=dict)
    age_distribution: Dict[str, int] = Field(default_factory=dict)  # age ranges
    
    # Performance metrics
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Risk-specific metrics
    risk_level_accuracy: Dict[RiskLevel, float] = Field(default_factory=dict)
    urgency_level_accuracy: Dict[UrgencyLevel, float] = Field(default_factory=dict)
    
    # Fairness and bias metrics
    demographic_parity: Optional[float] = Field(None, ge=0.0, le=1.0)
    equalized_odds: Optional[float] = Field(None, ge=0.0, le=1.0)
    calibration_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Performance by demographic groups
    accuracy_by_gender: Dict[Gender, float] = Field(default_factory=dict)
    accuracy_by_ethnicity: Dict[Ethnicity, float] = Field(default_factory=dict)
    
    # Confusion matrix and detailed results
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Dict[str, Any] = Field(default_factory=dict)
    
    # Model interpretability
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    shap_values_summary: Optional[Dict[str, Any]] = None
    
    # Performance characteristics
    inference_time_ms: Optional[float] = Field(None, ge=0)
    memory_usage_mb: Optional[float] = Field(None, ge=0)
    
    # Evaluation environment
    evaluation_environment: str = Field(..., description="Environment where evaluation was performed")
    evaluation_framework: str = Field(..., description="Framework used for evaluation")
    
    def record_gender_performance(self, gender: Gender, accuracy: float):
        """Record performance for specific gender."""
        self.accuracy_by_gender[gender] = accuracy
        self.update_timestamp()
    
    def record_ethnicity_performance(self, ethnicity: Ethnicity, accuracy: float):
        """Record performance for specific ethnicity."""
        self.accuracy_by_ethnicity[ethnicity] = accuracy
        self.update_timestamp()
    
    def calculate_bias_score(self) -> float:
        """Calculate overall bias score based on demographic performance differences."""
        bias_scores = []
        
        # Gender bias
        if len(self.accuracy_by_gender) > 1:
            gender_accuracies = list(self.accuracy_by_gender.values())
            gender_bias = max(gender_accuracies) - min(gender_accuracies)
            bias_scores.append(gender_bias)
        
        # Ethnicity bias
        if len(self.accuracy_by_ethnicity) > 1:
            ethnicity_accuracies = list(self.accuracy_by_ethnicity.values())
            ethnicity_bias = max(ethnicity_accuracies) - min(ethnicity_accuracies)
            bias_scores.append(ethnicity_bias)
        
        return max(bias_scores) if bias_scores else 0.0

# ============================================================================
# PREDICTION AND INFERENCE MODELS
# ============================================================================

class MLPrediction(BaseEntity, ValidationMixin, MetadataMixin):
    """Individual ML model prediction with full audit trail."""
    
    # Prediction context
    model_metadata_id: UUID = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version used")
    prediction_request_id: UUID = Field(..., description="Unique request identifier")
    
    # Input data
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    preprocessed_input: Optional[Dict[str, Any]] = Field(None, description="Preprocessed input data")
    
    # Patient context (for bias monitoring)
    patient_id: Optional[UUID] = None
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Prediction results
    prediction_output: Dict[str, Any] = Field(..., description="Raw model output")
    predicted_class: Optional[str] = None
    predicted_risk_level: Optional[RiskLevel] = None
    predicted_urgency_level: Optional[UrgencyLevel] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probability_distribution: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    
    # Model explanation
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    explanation_method: Optional[str] = Field(None, description="Method used for explanation (SHAP, LIME, etc.)")
    top_influential_features: List[str] = Field(default_factory=list)
    
    # Performance metrics
    inference_time_ms: float = Field(..., ge=0, description="Time taken for inference")
    preprocessing_time_ms: Optional[float] = Field(None, ge=0)
    postprocessing_time_ms: Optional[float] = Field(None, ge=0)
    memory_usage_mb: Optional[float] = Field(None, ge=0)
    
    # Bias detection
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detected bias in prediction")
    bias_alerts: List[str] = Field(default_factory=list)
    fairness_checked: bool = Field(default=False)
    
    # Validation and feedback
    human_validation: Optional[bool] = None
    human_validator_id: Optional[UUID] = None
    validation_timestamp: Optional[datetime] = None
    validation_notes: Optional[str] = None
    
    # Ground truth (when available)
    actual_outcome: Optional[str] = None
    outcome_timestamp: Optional[datetime] = None
    prediction_accuracy: Optional[bool] = None
    
    # Quality metrics
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    input_completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness of input data")
    
    def add_human_validation(self, validator_id: UUID, is_correct: bool, notes: str = ""):
        """Add human validation to the prediction."""
        self.human_validation = is_correct
        self.human_validator_id = validator_id
        self.validation_timestamp = datetime.utcnow()
        self.validation_notes = notes
        self.update_timestamp()
    
    def record_actual_outcome(self, outcome: str):
        """Record the actual outcome when it becomes available."""
        self.actual_outcome = outcome
        self.outcome_timestamp = datetime.utcnow()
        
        # Calculate accuracy if we have a predicted class
        if self.predicted_class:
            self.prediction_accuracy = (self.predicted_class == outcome)
        
        self.update_timestamp()
    
    def add_bias_alert(self, alert: str):
        """Add a bias alert to the prediction."""
        self.bias_alerts.append(f"{datetime.utcnow().isoformat()}: {alert}")
        self.update_timestamp()

class PredictionBatch(BaseEntity, ValidationMixin, MetadataMixin):
    """Batch prediction tracking for bulk inference."""
    
    # Batch information
    batch_name: str = Field(..., max_length=100)
    model_metadata_id: UUID = Field(..., description="Model used for batch prediction")
    model_version: str = Field(..., description="Model version used")
    
    # Batch configuration
    batch_size: int = Field(..., ge=1)
    total_samples: int = Field(..., ge=1)
    parallel_workers: int = Field(default=1, ge=1)
    
    # Progress tracking
    samples_processed: int = Field(default=0, ge=0)
    samples_failed: int = Field(default=0, ge=0)
    processing_status: TrainingStatus = Field(default=TrainingStatus.QUEUED)
    
    # Timing
    batch_started_at: Optional[datetime] = None
    batch_completed_at: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    
    # Results summary
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_inference_time_ms: Optional[float] = Field(None, ge=0)
    total_processing_time_minutes: Optional[int] = Field(None, ge=0)
    
    # Demographic distribution in batch
    gender_distribution: Dict[Gender, int] = Field(default_factory=dict)
    ethnicity_distribution: Dict[Ethnicity, int] = Field(default_factory=dict)
    
    # Bias monitoring for batch
    overall_bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias_alerts_count: int = Field(default=0, ge=0)
    
    # Output location
    results_file_path: Optional[str] = None
    error_log_path: Optional[str] = None
    
    def start_batch(self):
        """Mark batch processing as started."""
        self.processing_status = TrainingStatus.RUNNING
        self.batch_started_at = datetime.utcnow()
        self.update_timestamp()
    
    def update_progress(self, processed: int, failed: int = 0):
        """Update batch processing progress."""
        self.samples_processed = processed
        self.samples_failed = failed
        
        if self.batch_started_at:
            elapsed = datetime.utcnow() - self.batch_started_at
            if processed > 0:
                rate = processed / elapsed.total_seconds()
                remaining = self.total_samples - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                self.estimated_completion_time = datetime.utcnow() + timedelta(seconds=eta_seconds)
        
        self.update_timestamp()
    
    def complete_batch(self, results_path: str, summary_metrics: Dict[str, Any]):
        """Mark batch processing as completed."""
        self.processing_status = TrainingStatus.COMPLETED
        self.batch_completed_at = datetime.utcnow()
        self.results_file_path = results_path
        
        # Update summary metrics
        self.average_confidence = summary_metrics.get('average_confidence')
        self.average_inference_time_ms = summary_metrics.get('average_inference_time_ms')
        self.overall_bias_score = summary_metrics.get('overall_bias_score')
        
        if self.batch_started_at:
            duration = self.batch_completed_at - self.batch_started_at
            self.total_processing_time_minutes = int(duration.total_seconds() / 60)
        
        self.update_timestamp()

# ============================================================================
# MODEL DEPLOYMENT AND MONITORING
# ============================================================================

class ModelDeployment(BaseEntity, ValidationMixin, MetadataMixin):
    """Model deployment tracking and management."""
    
    # Deployment identification
    deployment_name: str = Field(..., max_length=100)
    model_metadata_id: UUID = Field(..., description="Deployed model")
    model_version: str = Field(..., description="Deployed model version")
    
    # Deployment configuration
    deployment_environment: DeploymentEnvironment
    deployment_url: str = Field(..., description="Deployment endpoint URL")
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource allocation
    allocated_cpu_cores: Optional[int] = Field(None, ge=1)
    allocated_memory_gb: Optional[float] = Field(None, ge=0)
    allocated_gpu_count: Optional[int] = Field(None, ge=0)
    auto_scaling_enabled: bool = Field(default=False)
    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=1, ge=1)
    
    # Deployment status
    deployment_status: ModelStatus = Field(default=ModelStatus.READY)
    health_check_url: Optional[str] = None
    last_health_check: Optional[datetime] = None
    is_healthy: bool = Field(default=True)
    
    # Usage tracking
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    average_response_time_ms: Optional[float] = Field(None, ge=0)
    
    # Performance monitoring
    current_load_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    peak_load_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    uptime_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Deployment dates
    deployed_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: Optional[datetime] = None
    scheduled_retirement_at: Optional[datetime] = None
    
    def record_request(self, success: bool, response_time_ms: float):
        """Record a request to the deployed model."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        if self.average_response_time_ms is None:
            self.average_response_time_ms = response_time_ms
        else:
            total_time = self.average_response_time_ms * (self.total_requests - 1) + response_time_ms
            self.average_response_time_ms = total_time / self.total_requests
        
        self.update_timestamp()
    
    def update_health_status(self, is_healthy: bool):
        """Update deployment health status."""
        self.is_healthy = is_healthy
        self.last_health_check = datetime.utcnow()
        self.update_timestamp()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ModelListResponse(BaseResponse):
    """Response for model listing requests."""
    models: List[ModelMetadata]
    total_count: int
    deployed_count: int
    training_count: int

class PredictionResponse(BaseResponse):
    """Response for individual prediction requests."""
    prediction: MLPrediction
    explanation: Dict[str, Any]
    confidence_score: float
    bias_score: Optional[float]

class TrainingJobResponse(BaseResponse):
    """Response for training job requests."""
    job: TrainingJob
    estimated_completion: Optional[datetime]
    resource_allocation: Dict[str, Any]

class EvaluationResponse(BaseResponse):
    """Response for model evaluation requests."""
    evaluation: ModelEvaluation
    performance_summary: Dict[str, float]
    bias_analysis: Dict[str, Any]
    recommendations: List[str]

# ============================================================================
# MODEL PERFORMANCE TRACKING
# ============================================================================


PERFORMANCE_THRESHOLDS = {
    ModelPurpose.TRIAGE_CLASSIFICATION: {
        "min_accuracy": 0.85,
        "max_bias_score": 0.1,
        "max_inference_time_ms": 500
    },
    ModelPurpose.RISK_ASSESSMENT: {
        "min_accuracy": 0.90,
        "max_bias_score": 0.05,
        "max_inference_time_ms": 200
    },
    ModelPurpose.BIAS_DETECTION: {
        "min_accuracy": 0.95,
        "max_bias_score": 0.02,
        "max_inference_time_ms": 100
    }
}

# Model purposes that require bias monitoring
BIAS_MONITORED_PURPOSES = [
    ModelPurpose.TRIAGE_CLASSIFICATION,
    ModelPurpose.RISK_ASSESSMENT,
    ModelPurpose.TREATMENT_RECOMMENDATION
]

# Default model configuration templates
DEFAULT_MODEL_CONFIGS = {
    ModelType.CLASSIFICATION: {
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10
        },
        "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    },
    ModelType.TRANSFORMER: {
        "hyperparameters": {
            "learning_rate": 5e-5,
            "batch_size": 16,
            "max_length": 512,
            "warmup_steps": 500
        },
        "evaluation_metrics": ["perplexity", "bleu_score", "accuracy"]
    }
}
