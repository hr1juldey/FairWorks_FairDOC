"""
Comprehensive bias detection and fairness monitoring models for Fairdoc Medical AI Backend.
Handles bias metrics, fairness reports, demographic analysis, and ethical AI compliance.
Fully integrated with all base model components.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import Field, field_validator
from enum import Enum

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

# ============================================================================
# BIAS DETECTION ENUMS AND TYPES
# ============================================================================

class BiasType(str, Enum):
    """Types of bias that can be detected in medical AI systems."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    TREATMENT_EQUALITY = "treatment_equality"
    INTERSECTIONAL = "intersectional"

class ProtectedAttribute(str, Enum):
    """Protected attributes for bias monitoring."""
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    AGE = "age"
    SOCIOECONOMIC_STATUS = "socioeconomic_status"
    GEOGRAPHIC_LOCATION = "geographic_location"
    INSURANCE_STATUS = "insurance_status"
    DISABILITY_STATUS = "disability_status"

class BiasAlertLevel(str, Enum):
    """Bias alert severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class FairnessMetric(str, Enum):
    """Fairness metrics used in bias detection."""
    STATISTICAL_PARITY = "statistical_parity"
    PREDICTIVE_PARITY = "predictive_parity"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    TREATMENT_EQUALITY = "treatment_equality"

class ModelDecisionType(str, Enum):
    """Types of model decisions to monitor for bias."""
    TRIAGE_CLASSIFICATION = "triage_classification"
    RISK_ASSESSMENT = "risk_assessment"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    SPECIALIST_REFERRAL = "specialist_referral"
    RESOURCE_ALLOCATION = "resource_allocation"
    DIAGNOSTIC_PRIORITY = "diagnostic_priority"

# ============================================================================
# BIAS METRICS MODELS
# ============================================================================

class DemographicGroup(TimestampMixin, ValidationMixin):
    """Represents a demographic group for bias analysis."""
    group_id: str = Field(..., description="Unique identifier for demographic group")
    gender: Gender = Field(..., description="Gender demographic")
    ethnicity: Ethnicity = Field(..., description="Ethnicity demographic")
    age_range: str = Field(..., description="Age range in years (e.g., '18-25')")
    population_size: int = Field(..., ge=0, description="Number of individuals in this group")
    
    # Statistical properties
    baseline_prevalence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Baseline condition prevalence")
    representation_ratio: Optional[float] = Field(None, ge=0.0, description="Ratio compared to majority group")
    
    @field_validator('group_id')
    @classmethod
    def validate_group_id(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError('Group ID cannot be empty')
        return v.strip()

class BiasMetric(BaseEntity, ValidationMixin, MetadataMixin):
    """Individual bias metric measurement."""
    
    # Measurement context
    model_decision_type: ModelDecisionType
    fairness_metric: FairnessMetric
    protected_attribute: ProtectedAttribute
    
    # Demographic groups being compared
    reference_group: str = Field(..., description="Reference/majority group identifier")
    comparison_group: str = Field(..., description="Group being compared for bias")
    
    # Demographic details (used from base_models)
    reference_gender: Gender = Field(None, description="Reference group gender")
    reference_ethnicity: Ethnicity = Field(None, description="Reference group ethnicity")
    comparison_gender: Gender = Field(None, description="Comparison group gender")
    comparison_ethnicity: Ethnicity = Field(None, description="Comparison group ethnicity")
    
    # Risk and urgency monitoring (using base_models enums)
    risk_level_distribution: Dict[RiskLevel, float] = Field(
        default_factory=lambda: {
            RiskLevel.LOW: 0.0,
            RiskLevel.MODERATE: 0.0,
            RiskLevel.HIGH: 0.0,
            RiskLevel.CRITICAL: 0.0
        }
    )
    
    urgency_level_distribution: Dict[UrgencyLevel, float] = Field(
        default_factory=lambda: {
            UrgencyLevel.ROUTINE: 0.0,
            UrgencyLevel.URGENT: 0.0,
            UrgencyLevel.EMERGENT: 0.0,
            UrgencyLevel.IMMEDIATE: 0.0
        }
    )
    
    # Metric values
    metric_value: float = Field(..., description="Calculated fairness metric value")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Acceptable threshold for this metric")
    is_biased: bool = Field(..., description="Whether bias is detected based on threshold")
    
    # Statistical significance
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    p_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="Statistical significance")
    sample_size: int = Field(..., ge=1, description="Sample size used for calculation")
    
    # Contextual information
    measurement_period: Dict[str, datetime] = Field(..., description="Start and end time of measurement")
    data_source: str = Field(..., description="Source of data used for bias calculation")
    
    # Alert information
    alert_level: BiasAlertLevel = Field(default=BiasAlertLevel.LOW)
    alert_triggered: bool = Field(default=False)
    alert_message: Optional[str] = None
    
    def calculate_alert_level(self) -> BiasAlertLevel:
        """Calculate alert level based on metric value and threshold."""
        if not self.is_biased:
            return BiasAlertLevel.LOW
        
        bias_severity = abs(self.metric_value - self.threshold) / self.threshold
        
        if bias_severity >= 0.5:
            return BiasAlertLevel.CRITICAL
        elif bias_severity >= 0.3:
            return BiasAlertLevel.HIGH
        elif bias_severity >= 0.1:
            return BiasAlertLevel.MODERATE
        else:
            return BiasAlertLevel.LOW
    
    def update_alert_status(self):
        """Update alert level and trigger status."""
        self.alert_level = self.calculate_alert_level()
        self.alert_triggered = self.alert_level in [BiasAlertLevel.HIGH, BiasAlertLevel.CRITICAL]
        self.update_timestamp()  # Using TimestampMixin method
    
    def record_risk_distribution(self, risk_counts: Dict[RiskLevel, int]):
        """Record distribution of risk levels."""
        total = sum(risk_counts.values())
        if total > 0:
            for risk_level in RiskLevel:
                count = risk_counts.get(risk_level, 0)
                self.risk_level_distribution[risk_level] = count / total
    
    def record_urgency_distribution(self, urgency_counts: Dict[UrgencyLevel, int]):
        """Record distribution of urgency levels."""
        total = sum(urgency_counts.values())
        if total > 0:
            for urgency_level in UrgencyLevel:
                count = urgency_counts.get(urgency_level, 0)
                self.urgency_level_distribution[urgency_level] = count / total

class IntersectionalBiasMetric(BaseEntity, ValidationMixin, MetadataMixin):
    """Bias metric for intersectional analysis (multiple protected attributes)."""
    
    # Multiple protected attributes
    protected_attributes: List[ProtectedAttribute] = Field(..., min_items=2, max_items=5)
    
    # Demographic details
    gender_pairs: List[Dict[str, Gender]] = Field(default_factory=list)
    ethnicity_pairs: List[Dict[str, Ethnicity]] = Field(default_factory=list)
    
    # Intersectional group definition
    intersectional_group: Dict[ProtectedAttribute, str] = Field(..., description="Intersectional group definition")
    
    # Comparison metrics
    individual_metrics: List[UUID] = Field(..., description="Individual bias metrics for each attribute")
    intersectional_metric_value: float = Field(..., description="Combined intersectional bias score")
    
    # Risk level impact
    risk_level_impact: Dict[RiskLevel, float] = Field(
        default_factory=lambda: {
            RiskLevel.LOW: 0.0,
            RiskLevel.MODERATE: 0.0,
            RiskLevel.HIGH: 0.0,
            RiskLevel.CRITICAL: 0.0
        }
    )
    
    # Interaction effects
    interaction_strength: Optional[float] = Field(None, ge=0.0, le=1.0, description="Strength of interaction between attributes")
    amplification_factor: Optional[float] = Field(None, description="How much bias is amplified by intersection")
    
    # Sample statistics
    intersectional_sample_size: int = Field(..., ge=1)
    representation_adequacy: float = Field(..., ge=0.0, le=1.0, description="Adequacy of sample representation")
    
    def add_demographic_pair(self, group_name: str, gender: Gender, ethnicity: Ethnicity):
        """Add a demographic pair for intersectional analysis."""
        self.gender_pairs.append({"group": group_name, "gender": gender})
        self.ethnicity_pairs.append({"group": group_name, "ethnicity": ethnicity})
        self.update_timestamp()  # Using TimestampMixin method

# ============================================================================
# BIAS DETECTION RESULTS
# ============================================================================

class BiasDetectionResult(BaseEntity, ValidationMixin, MetadataMixin):
    """Result of a bias detection analysis."""
    
    # Analysis context
    model_id: str = Field(..., description="ID of the AI model being analyzed")
    model_version: str = Field(..., description="Version of the model")
    analysis_type: str = Field(..., description="Type of bias analysis performed")
    
    # Time period analyzed
    analysis_period: Dict[str, datetime] = Field(..., description="Start and end time of analysis")
    total_predictions: int = Field(..., ge=0, description="Total predictions analyzed")
    
    # Detected bias metrics
    bias_metrics: List[UUID] = Field(default_factory=list, description="Individual bias metrics detected")
    intersectional_metrics: List[UUID] = Field(default_factory=list, description="Intersectional bias metrics")
    
    # Demographic details
    gender_distribution: Dict[Gender, int] = Field(default_factory=dict)
    ethnicity_distribution: Dict[Ethnicity, int] = Field(default_factory=dict)
    
    # Risk and urgency distributions (using base model enums)
    risk_level_by_gender: Dict[Gender, Dict[RiskLevel, float]] = Field(default_factory=dict)
    risk_level_by_ethnicity: Dict[Ethnicity, Dict[RiskLevel, float]] = Field(default_factory=dict)
    
    urgency_level_by_gender: Dict[Gender, Dict[UrgencyLevel, float]] = Field(default_factory=dict)
    urgency_level_by_ethnicity: Dict[Ethnicity, Dict[UrgencyLevel, float]] = Field(default_factory=dict)
    
    # Overall bias assessment
    overall_bias_score: float = Field(..., ge=0.0, le=1.0, description="Aggregate bias score")
    max_bias_severity: BiasAlertLevel = Field(..., description="Highest bias alert level detected")
    biased_attributes: List[ProtectedAttribute] = Field(default_factory=list)
    
    # Statistical summary
    bias_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    demographic_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Recommendations
    mitigation_recommendations: List[str] = Field(default_factory=list)
    immediate_actions_required: bool = Field(default=False)
    
    def add_bias_metric(self, metric_id: UUID, metric_type: BiasType):
        """Add a bias metric to the detection result."""
        self.bias_metrics.append(metric_id)
        self.add_metadata(f"metric_{len(self.bias_metrics)}", {
            "id": str(metric_id),
            "type": metric_type.value,
            "detected_at": datetime.utcnow().isoformat()
        })
        self.update_timestamp()  # Using TimestampMixin method
    
    def calculate_overall_bias_score(self, individual_scores: List[float]) -> float:
        """Calculate aggregate bias score from individual metrics."""
        if not individual_scores:
            return 0.0
        
        # Use weighted average with higher weight for severe biases
        weights = [min(1.0, score * 2) for score in individual_scores]
        weighted_sum = sum(score * weight for score, weight in zip(individual_scores, weights, strict=False))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def record_gender_distribution(self, counts: Dict[Gender, int]):
        """Record distribution of genders in the analyzed dataset."""
        self.gender_distribution = counts
        self.update_timestamp()
    
    def record_ethnicity_distribution(self, counts: Dict[Ethnicity, int]):
        """Record distribution of ethnicities in the analyzed dataset."""
        self.ethnicity_distribution = counts
        self.update_timestamp()
    
    def record_risk_by_gender(self, gender: Gender, risk_counts: Dict[RiskLevel, int]):
        """Record risk level distribution for a specific gender."""
        total = sum(risk_counts.values())
        if total > 0:
            distribution = {}
            for risk_level in RiskLevel:
                count = risk_counts.get(risk_level, 0)
                distribution[risk_level] = count / total
            self.risk_level_by_gender[gender] = distribution
            self.update_timestamp()
    
    def record_urgency_by_ethnicity(self, ethnicity: Ethnicity, urgency_counts: Dict[UrgencyLevel, int]):
        """Record urgency level distribution for a specific ethnicity."""
        total = sum(urgency_counts.values())
        if total > 0:
            distribution = {}
            for urgency_level in UrgencyLevel:
                count = urgency_counts.get(urgency_level, 0)
                distribution[urgency_level] = count / total
            self.urgency_level_by_ethnicity[ethnicity] = distribution
            self.update_timestamp()

# ============================================================================
# FAIRNESS MONITORING MODELS
# ============================================================================

class FairnessThreshold(TimestampMixin, ValidationMixin):
    """Fairness thresholds for different metrics and contexts."""
    
    fairness_metric: FairnessMetric
    protected_attribute: ProtectedAttribute
    model_context: ModelDecisionType
    
    # Threshold values
    acceptable_threshold: float = Field(..., ge=0.0, le=1.0)
    warning_threshold: float = Field(..., ge=0.0, le=1.0)
    critical_threshold: float = Field(..., ge=0.0, le=1.0)
    
    # Demographic-specific thresholds
    gender_specific_thresholds: Dict[Gender, float] = Field(default_factory=dict)
    ethnicity_specific_thresholds: Dict[Ethnicity, float] = Field(default_factory=dict)
    
    # Regulatory compliance
    regulatory_requirement: Optional[str] = Field(None, description="Legal/regulatory requirement source")
    compliance_status: bool = Field(default=True)
    
    # Review and updates
    last_reviewed: datetime = Field(default_factory=datetime.utcnow)
    review_frequency_days: int = Field(default=90, ge=1)
    
    @field_validator('critical_threshold')
    @classmethod
    def validate_threshold_order(cls, v: float, info) -> float:
        """Ensure thresholds are in logical order."""
        if hasattr(info, 'data'):
            acceptable = info.data.get('acceptable_threshold', 0)
            warning = info.data.get('warning_threshold', 0)
            if v <= warning <= acceptable:
                raise ValueError('Thresholds must be: critical > warning > acceptable')
        return v
    
    def set_gender_threshold(self, gender: Gender, threshold: float):
        """Set threshold specific to a gender."""
        self.gender_specific_thresholds[gender] = threshold
        self.update_timestamp()
    
    def set_ethnicity_threshold(self, ethnicity: Ethnicity, threshold: float):
        """Set threshold specific to an ethnicity."""
        self.ethnicity_specific_thresholds[ethnicity] = threshold
        self.update_timestamp()

class FairnessReport(BaseEntity, ValidationMixin, MetadataMixin):
    """Comprehensive fairness assessment report."""
    
    # Report metadata
    report_title: str = Field(..., max_length=200)
    report_type: Literal["daily", "weekly", "monthly", "incident", "compliance"] = "daily"
    reporting_period: Dict[str, datetime] = Field(..., description="Period covered by report")
    
    # System being assessed
    system_name: str = Field(..., description="Name of AI system being assessed")
    system_version: str = Field(..., description="Version of the system")
    deployment_environment: str = Field(..., description="Environment (dev, test, prod)")
    
    # Bias detection results
    bias_detection_results: List[UUID] = Field(default_factory=list)
    overall_fairness_score: float = Field(..., ge=0.0, le=1.0)
    compliance_status: Dict[str, bool] = Field(default_factory=dict)
    
    # Key findings
    critical_issues: List[str] = Field(default_factory=list)
    areas_of_concern: List[str] = Field(default_factory=list)
    improvements_noted: List[str] = Field(default_factory=list)
    
    # Demographics analysis using base models
    gender_fairness_scores: Dict[Gender, float] = Field(default_factory=dict)
    ethnicity_fairness_scores: Dict[Ethnicity, float] = Field(default_factory=dict)
    
    # Risk level fairness across demographics
    risk_level_fairness: Dict[RiskLevel, Dict[str, float]] = Field(default_factory=dict)
    urgency_level_fairness: Dict[UrgencyLevel, Dict[str, float]] = Field(default_factory=dict)
    
    # Demographics analysis
    patient_demographics: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    underrepresented_groups: List[str] = Field(default_factory=list)
    demographic_coverage: Dict[str, float] = Field(default_factory=dict)
    
    # Trend analysis
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    comparison_to_previous: Optional[Dict[str, float]] = None
    
    # Actions and recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    recommended_interventions: List[str] = Field(default_factory=list)
    follow_up_required: bool = Field(default=False)
    next_review_date: Optional[datetime] = None
    
    # Stakeholder information
    prepared_by: str = Field(..., description="Report preparer")
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    distribution_list: List[str] = Field(default_factory=list)
    
    def add_critical_issue(self, issue: str, severity: BiasAlertLevel):
        """Add a critical issue to the report."""
        timestamp = datetime.utcnow().isoformat()
        formatted_issue = f"[{severity.value.upper()}] {timestamp}: {issue}"
        self.critical_issues.append(formatted_issue)
        
        if severity in [BiasAlertLevel.HIGH, BiasAlertLevel.CRITICAL]:
            self.follow_up_required = True
        
        self.update_timestamp()  # Using TimestampMixin method
    
    def record_gender_fairness(self, gender: Gender, fairness_score: float):
        """Record fairness score for a specific gender."""
        self.gender_fairness_scores[gender] = fairness_score
        self.update_timestamp()
    
    def record_ethnicity_fairness(self, ethnicity: Ethnicity, fairness_score: float):
        """Record fairness score for a specific ethnicity."""
        self.ethnicity_fairness_scores[ethnicity] = fairness_score
        self.update_timestamp()
    
    def record_risk_level_fairness(self, risk_level: RiskLevel, metrics: Dict[str, float]):
        """Record fairness metrics for a specific risk level."""
        self.risk_level_fairness[risk_level] = metrics
        self.update_timestamp()
    
    def record_urgency_level_fairness(self, urgency_level: UrgencyLevel, metrics: Dict[str, float]):
        """Record fairness metrics for a specific urgency level."""
        self.urgency_level_fairness[urgency_level] = metrics
        self.update_timestamp()

# ============================================================================
# BIAS MITIGATION MODELS
# ============================================================================

class BiasIntervention(BaseEntity, ValidationMixin, MetadataMixin):
    """Bias mitigation intervention record."""
    
    # Intervention details
    intervention_name: str = Field(..., max_length=100)
    intervention_type: Literal["data", "algorithm", "post_processing", "human_review"]
    target_bias_type: BiasType
    target_attributes: List[ProtectedAttribute]
    
    # Demographic targeting
    target_genders: List[Gender] = Field(default_factory=list)
    target_ethnicities: List[Ethnicity] = Field(default_factory=list)
    
    # Risk and urgency levels targeted
    target_risk_levels: List[RiskLevel] = Field(default_factory=list)
    target_urgency_levels: List[UrgencyLevel] = Field(default_factory=list)
    
    # Implementation details
    implementation_date: datetime = Field(default_factory=datetime.utcnow)
    implementation_status: Literal["planned", "in_progress", "completed", "failed"]
    
    # Effectiveness tracking
    pre_intervention_bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    post_intervention_bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    effectiveness_measure: Optional[float] = Field(None, description="Reduction in bias score")
    
    # Resource requirements
    estimated_cost: Optional[float] = Field(None, ge=0)
    implementation_effort_hours: Optional[int] = Field(None, ge=0)
    staff_required: List[str] = Field(default_factory=list)
    
    # Monitoring and evaluation
    monitoring_metrics: List[FairnessMetric] = Field(default_factory=list)
    evaluation_period_days: int = Field(default=30, ge=1)
    success_criteria: Dict[str, float] = Field(default_factory=dict)
    
    # Side effects and trade-offs
    potential_side_effects: List[str] = Field(default_factory=list)
    performance_impact: Optional[Dict[str, float]] = None
    
    def calculate_effectiveness(self) -> Optional[float]:
        """Calculate intervention effectiveness."""
        if self.pre_intervention_bias_score is None or self.post_intervention_bias_score is None:
            return None
        
        if self.pre_intervention_bias_score == 0:
            return 0.0
        
        reduction = (self.pre_intervention_bias_score - self.post_intervention_bias_score) / self.pre_intervention_bias_score
        return max(0.0, min(1.0, reduction))  # Clamp between 0 and 1
    
    def update_effectiveness(self):
        """Update effectiveness measure based on pre/post scores."""
        self.effectiveness_measure = self.calculate_effectiveness()
        self.update_timestamp()
    
    def add_target_gender(self, gender: Gender):
        """Add a gender to target for intervention."""
        if gender not in self.target_genders:
            self.target_genders.append(gender)
            self.update_timestamp()
    
    def add_target_ethnicity(self, ethnicity: Ethnicity):
        """Add an ethnicity to target for intervention."""
        if ethnicity not in self.target_ethnicities:
            self.target_ethnicities.append(ethnicity)
            self.update_timestamp()
    
    def add_target_risk_level(self, risk_level: RiskLevel):
        """Add a risk level to target for intervention."""
        if risk_level not in self.target_risk_levels:
            self.target_risk_levels.append(risk_level)
            self.update_timestamp()

class BiasAuditTrail(BaseEntity, ValidationMixin):
    """Audit trail for bias-related actions and decisions."""
    
    # Action details
    action_type: Literal["detection", "intervention", "review", "alert", "mitigation"]
    action_description: str = Field(..., max_length=500)
    action_taken_by: str = Field(..., description="User or system that took the action")
    
    # Context
    affected_model: str = Field(..., description="Model affected by the action")
    affected_groups: List[str] = Field(default_factory=list)
    bias_metrics_involved: List[UUID] = Field(default_factory=list)
    
    # Demographic impact
    affected_genders: List[Gender] = Field(default_factory=list)
    affected_ethnicities: List[Ethnicity] = Field(default_factory=list)
    
    # Risk and urgency impact
    affected_risk_levels: List[RiskLevel] = Field(default_factory=list)
    affected_urgency_levels: List[UrgencyLevel] = Field(default_factory=list)
    
    # Outcome
    action_outcome: str = Field(..., max_length=200)
    outcome_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Compliance
    regulatory_compliance: bool = Field(default=True)
    compliance_notes: Optional[str] = None
    
    # Follow-up
    requires_follow_up: bool = Field(default=False)
    follow_up_date: Optional[datetime] = None
    follow_up_assigned_to: Optional[str] = None

# ============================================================================
# REAL-TIME MONITORING MODELS
# ============================================================================

class RealTimeBiasAlert(BaseEntity, ValidationMixin):
    """Real-time bias alert for immediate intervention."""
    
    # Alert details
    alert_id: str = Field(default_factory=lambda: f"BIAS_ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    severity: BiasAlertLevel
    alert_message: str = Field(..., max_length=500)
    
    # Context
    model_prediction_id: Optional[UUID] = None
    protected_attributes: Dict[ProtectedAttribute, str] = Field(default_factory=dict)
    prediction_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Demographic details
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    
    # Risk assessment
    risk_level: Optional[RiskLevel] = None
    urgency_level: Optional[UrgencyLevel] = None
    
    # Bias detection details
    detected_bias_type: BiasType
    bias_score: float = Field(..., ge=0.0, le=1.0)
    threshold_exceeded: float = Field(..., description="How much the threshold was exceeded")
    
    # Response tracking
    alert_acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Resolution
    resolution_action: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Escalation
    escalated: bool = Field(default=False)
    escalated_to: Optional[str] = None
    escalation_reason: Optional[str] = None
    
    def acknowledge_alert(self, acknowledged_by: str):
        """Acknowledge the bias alert."""
        self.alert_acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.utcnow()
        self.update_timestamp()  # Using TimestampMixin method
    
    def resolve_alert(self, resolved_by: str, action: str, notes: str = ""):
        """Resolve the bias alert."""
        self.resolution_action = action
        self.resolved_by = resolved_by
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = notes
        self.update_timestamp()  # Using TimestampMixin method
    
    def set_demographic_details(self, gender: Gender, ethnicity: Ethnicity):
        """Set patient demographic details."""
        self.patient_gender = gender
        self.patient_ethnicity = ethnicity
        self.update_timestamp()
    
    def set_risk_assessment(self, risk_level: RiskLevel, urgency_level: UrgencyLevel):
        """Set risk and urgency assessment details."""
        self.risk_level = risk_level
        self.urgency_level = urgency_level
        self.update_timestamp()

class BiasMonitoringSession(BaseEntity, ValidationMixin, MetadataMixin):
    """Monitoring session for continuous bias tracking."""
    
    # Session details
    session_name: str = Field(..., max_length=100)
    monitoring_start: datetime = Field(default_factory=datetime.utcnow)
    monitoring_end: Optional[datetime] = None
    session_duration: Optional[timedelta] = None
    
    # Monitoring configuration
    monitored_models: List[str] = Field(..., min_items=1)
    monitored_attributes: List[ProtectedAttribute] = Field(..., min_items=1)
    monitoring_frequency: Literal["real_time", "hourly", "daily", "weekly"] = "real_time"
    
    # Demographic monitoring
    monitored_genders: List[Gender] = Field(default_factory=list)
    monitored_ethnicities: List[Ethnicity] = Field(default_factory=list)
    
    # Risk and urgency monitoring
    monitored_risk_levels: List[RiskLevel] = Field(default_factory=list)
    monitored_urgency_levels: List[UrgencyLevel] = Field(default_factory=list)
    
    # Thresholds and alerts
    active_thresholds: List[UUID] = Field(default_factory=list, description="Active fairness thresholds")
    alerts_generated: List[UUID] = Field(default_factory=list, description="Alerts generated during session")
    
    # Statistics
    total_predictions_monitored: int = Field(default=0, ge=0)
    bias_incidents_detected: int = Field(default=0, ge=0)
    false_positive_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Demographic statistics
    predictions_by_gender: Dict[Gender, int] = Field(default_factory=dict)
    predictions_by_ethnicity: Dict[Ethnicity, int] = Field(default_factory=dict)
    alerts_by_gender: Dict[Gender, int] = Field(default_factory=dict)
    alerts_by_ethnicity: Dict[Ethnicity, int] = Field(default_factory=dict)
    
    # Session outcomes
    session_status: Literal["active", "paused", "completed", "terminated"] = "active"
    session_summary: Optional[str] = None
    
    def end_session(self):
        """End the monitoring session."""
        self.monitoring_end = datetime.utcnow()
        self.session_duration = self.monitoring_end - self.monitoring_start
        self.session_status = "completed"
        self.update_timestamp()  # Using TimestampMixin method
    
    def increment_prediction_count(self, gender: Gender, ethnicity: Ethnicity):
        """Increment prediction count by demographics."""
        self.total_predictions_monitored += 1
        
        # Update gender counts
        self.predictions_by_gender[gender] = self.predictions_by_gender.get(gender, 0) + 1
        
        # Update ethnicity counts
        self.predictions_by_ethnicity[ethnicity] = self.predictions_by_ethnicity.get(ethnicity, 0) + 1
        
        self.update_timestamp()
    
    def record_bias_alert(self, alert_id: UUID, gender: Gender, ethnicity: Ethnicity):
        """Record a bias alert with demographic information."""
        self.alerts_generated.append(alert_id)
        self.bias_incidents_detected += 1
        
        # Update alert counts by demographics
        self.alerts_by_gender[gender] = self.alerts_by_gender.get(gender, 0) + 1
        self.alerts_by_ethnicity[ethnicity] = self.alerts_by_ethnicity.get(ethnicity, 0) + 1
        
        self.update_timestamp()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class BiasAnalysisResponse(BaseResponse):
    """Response for bias analysis requests."""
    bias_detected: bool
    bias_metrics: List[Dict[str, Any]]
    overall_bias_score: float
    alert_level: BiasAlertLevel
    recommendations: List[str]
    immediate_action_required: bool
    
    # Demographic breakdown
    gender_analysis: Dict[Gender, float]
    ethnicity_analysis: Dict[Ethnicity, float]
    
    # Risk level analysis
    risk_level_analysis: Dict[RiskLevel, Dict[str, float]]
    urgency_level_analysis: Dict[UrgencyLevel, Dict[str, float]]

class FairnessReportResponse(BaseResponse):
    """Response for fairness report requests."""
    report: FairnessReport
    summary_statistics: Dict[str, Any]
    trend_data: Dict[str, List[float]]
    compliance_status: Dict[str, bool]
    
    # Demographic fairness summaries
    gender_fairness: Dict[Gender, float]
    ethnicity_fairness: Dict[Ethnicity, float]

class BiasAlertResponse(BaseResponse):
    """Response for bias alert notifications."""
    alert: RealTimeBiasAlert
    severity: BiasAlertLevel
    immediate_actions: List[str]
    escalation_contacts: List[str]
    
    # Patient demographics
    patient_gender: Optional[Gender]
    patient_ethnicity: Optional[Ethnicity]
    
    # Risk assessment
    risk_level: Optional[RiskLevel]
    urgency_level: Optional[UrgencyLevel]

# ============================================================================
# BIAS METRIC CALCULATION UTILITIES
# ============================================================================


FAIRNESS_METRIC_DESCRIPTIONS = {
    FairnessMetric.STATISTICAL_PARITY: "Equal positive prediction rates across groups",
    FairnessMetric.PREDICTIVE_PARITY: "Equal positive predictive values across groups",
    FairnessMetric.EQUALITY_OF_OPPORTUNITY: "Equal true positive rates across groups",
    FairnessMetric.EQUALIZED_ODDS: "Equal true positive and false positive rates across groups",
    FairnessMetric.CALIBRATION: "Equal predicted probabilities match actual outcomes across groups",
    FairnessMetric.INDIVIDUAL_FAIRNESS: "Similar individuals receive similar predictions",
    FairnessMetric.TREATMENT_EQUALITY: "Equal error rates across groups"
}

DEFAULT_BIAS_THRESHOLDS = {
    BiasType.DEMOGRAPHIC_PARITY: 0.1,
    BiasType.EQUALIZED_ODDS: 0.1,
    BiasType.CALIBRATION: 0.05,
    BiasType.INDIVIDUAL_FAIRNESS: 0.15,
    BiasType.COUNTERFACTUAL_FAIRNESS: 0.1,
    BiasType.TREATMENT_EQUALITY: 0.1,
    BiasType.INTERSECTIONAL: 0.2
}

# Protected attribute combinations for intersectional analysis
INTERSECTIONAL_COMBINATIONS = [
    [ProtectedAttribute.GENDER, ProtectedAttribute.ETHNICITY],
    [ProtectedAttribute.AGE, ProtectedAttribute.GENDER],
    [ProtectedAttribute.ETHNICITY, ProtectedAttribute.SOCIOECONOMIC_STATUS],
    [ProtectedAttribute.GENDER, ProtectedAttribute.ETHNICITY, ProtectedAttribute.AGE]
]

# Risk level monitoring thresholds by demographic group
RISK_LEVEL_THRESHOLDS = {
    RiskLevel.LOW: 0.05,
    RiskLevel.MODERATE: 0.08,
    RiskLevel.HIGH: 0.12,
    RiskLevel.CRITICAL: 0.15
}

# Urgency level monitoring thresholds by demographic group
URGENCY_LEVEL_THRESHOLDS = {
    UrgencyLevel.ROUTINE: 0.05,
    UrgencyLevel.URGENT: 0.08,
    UrgencyLevel.EMERGENT: 0.12,
    UrgencyLevel.IMMEDIATE: 0.15
}
