# Conversation Evaluator - Comprehensive E2E Test Evaluation
"""
Evaluates the performance of DSPy medical agents in end-to-end conversations
Provides detailed scoring and analysis of medical triage accuracy
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
import asyncio
from datetime import datetime

from src.app2.models.schemas.medical_triage import MedicalOutcome
from patient_profiles import PatientProfile, get_patient_profile
from medical_conditions import MedicalCondition, get_condition

logger = structlog.get_logger(__name__)

class EvaluationCriteria(str, Enum):
    """Evaluation criteria for conversation assessment"""
    OUTCOME_ACCURACY = "outcome_accuracy"
    EMERGENCY_DETECTION = "emergency_detection" 
    TURN_EFFICIENCY = "turn_efficiency"
    RED_FLAG_DETECTION = "red_flag_detection"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    NICE_PROTOCOL_ALIGNMENT = "nice_protocol_alignment"
    PATIENT_EXPERIENCE = "patient_experience"
    COMMUNICATION_QUALITY = "communication_quality"

@dataclass
class ConversationEvaluation:
    """Complete evaluation of a single conversation"""
    
    # Basic identification
    patient_id: str
    condition_id: str
    conversation_id: str
    
    # Core outcomes
    expected_outcome: MedicalOutcome
    actual_outcome: MedicalOutcome
    outcome_match: bool
    
    # Performance metrics
    final_confidence: float
    total_turns: int
    time_to_outcome: float  # seconds
    
    # Detection accuracy
    expected_red_flags: List[str]
    detected_red_flags: List[str]
    missed_red_flags: List[str]
    false_positive_flags: List[str]
    
    # Protocol alignment
    relevant_protocols: List[str]
    used_protocols: List[str]
    protocol_alignment_score: float
    
    # Detailed scoring
    scores: Dict[EvaluationCriteria, float]  # 0-100 for each criteria
    overall_score: float
    
    # Qualitative assessment
    strengths: List[str]
    weaknesses: List[str]
    critical_issues: List[str]
    
    # Raw conversation data
    conversation_turns: List[Dict[str, Any]]
    patient_responses: List[str]
    agent_questions: List[str]
    
    # Metadata
    evaluation_timestamp: datetime
    evaluation_notes: str

class ConversationEvaluator:
    """Comprehensive evaluator for medical triage conversations"""
    
    def __init__(self):
        self.evaluation_weights = self._initialize_evaluation_weights()
        self.critical_thresholds = self._initialize_critical_thresholds()
        
        logger.info("📊 Conversation evaluator initialized")
    
    def _initialize_evaluation_weights(self) -> Dict[EvaluationCriteria, float]:
        """Initialize weights for different evaluation criteria"""
        return {
            EvaluationCriteria.OUTCOME_ACCURACY: 0.25,         # 25% - Most important
            EvaluationCriteria.EMERGENCY_DETECTION: 0.20,      # 20% - Critical for safety
            EvaluationCriteria.RED_FLAG_DETECTION: 0.15,       # 15% - Important for accuracy
            EvaluationCriteria.TURN_EFFICIENCY: 0.10,          # 10% - Efficiency matters
            EvaluationCriteria.CONFIDENCE_CALIBRATION: 0.10,   # 10% - Trust calibration
            EvaluationCriteria.NICE_PROTOCOL_ALIGNMENT: 0.08,  # 8% - Clinical standards
            EvaluationCriteria.PATIENT_EXPERIENCE: 0.07,       # 7% - User experience
            EvaluationCriteria.COMMUNICATION_QUALITY: 0.05     # 5% - Communication style
        }
    
    def _initialize_critical_thresholds(self) -> Dict[str, float]:
        """Initialize thresholds for critical failures"""
        return {
            "emergency_detection_minimum": 90.0,  # Must detect emergencies with 90%+ accuracy
            "false_emergency_maximum": 10.0,      # Max 10% false emergency rate
            "overall_minimum_passing": 60.0,      # 60% minimum to pass
            "critical_emergency_minimum": 95.0,   # Critical emergencies must be 95%+ accurate
            "maximum_acceptable_turns": 10         # Max turns before conversation timeout
        }
    
    async def evaluate_conversation(
        self, 
        patient_id: str,
        conversation_data: Dict[str, Any],
        agent_results: List[Dict[str, Any]]
    ) -> ConversationEvaluation:
        """
        Comprehensive evaluation of a single medical triage conversation
        
        Args:
            patient_id: ID of simulated patient
            conversation_data: Complete conversation record
            agent_results: List of agent responses and decisions
            
        Returns:
            ConversationEvaluation with detailed scoring
        """
        
        logger.info("🔍 Evaluating conversation", 
                   patient_id=patient_id,
                   turns=len(agent_results))
        
        # Get patient and condition context
        profile = get_patient_profile(patient_id)
        condition = get_condition(profile.medical_condition)
        
        # Extract conversation details
        final_result = agent_results[-1] if agent_results else {}
        
        # Initialize evaluation structure
        evaluation = ConversationEvaluation(
            patient_id=patient_id,
            condition_id=profile.medical_condition,
            conversation_id=conversation_data.get("conversation_id", "unknown"),
            
            expected_outcome=condition.expected_outcome,
            actual_outcome=self._extract_actual_outcome(final_result),
            outcome_match=False,  # Will be calculated
            
            final_confidence=final_result.get("confidence", 0.0),
            total_turns=len(agent_results),
            time_to_outcome=conversation_data.get("total_time", 0.0),
            
            expected_red_flags=condition.red_flag_indicators,
            detected_red_flags=[],  # Will be populated
            missed_red_flags=[],
            false_positive_flags=[],
            
            relevant_protocols=condition.relevant_protocols,
            used_protocols=conversation_data.get("protocols_used", []),
            protocol_alignment_score=0.0,
            
            scores={},
            overall_score=0.0,
            
            strengths=[],
            weaknesses=[],
            critical_issues=[],
            
            conversation_turns=agent_results,
            patient_responses=conversation_data.get("patient_responses", []),
            agent_questions=conversation_data.get("agent_questions", []),
            
            evaluation_timestamp=datetime.now(),
            evaluation_notes=""
        )
        
        # Perform detailed evaluation
        await self._evaluate_outcome_accuracy(evaluation, condition)
        await self._evaluate_emergency_detection(evaluation, condition)
        await self._evaluate_turn_efficiency(evaluation, condition)
        await self._evaluate_red_flag_detection(evaluation, agent_results)
        await self._evaluate_confidence_calibration(evaluation, condition)
        await self._evaluate_protocol_alignment(evaluation)
        await self._evaluate_patient_experience(evaluation, profile)
        await self._evaluate_communication_quality(evaluation)
        
        # Calculate overall score
        evaluation.overall_score = self._calculate_overall_score(evaluation)
        
        # Identify critical issues
        self._identify_critical_issues(evaluation, condition)
        
        # Generate qualitative assessment
        self._generate_qualitative_assessment(evaluation, condition, profile)
        
        logger.info("✅ Conversation evaluation completed",
                   patient_id=patient_id,
                   overall_score=evaluation.overall_score,
                   outcome_match=evaluation.outcome_match)
        
        return evaluation
    
    async def _evaluate_outcome_accuracy(self, evaluation: ConversationEvaluation, condition: MedicalCondition):
        """Evaluate accuracy of final medical outcome"""
        
        # Check if outcomes match
        evaluation.outcome_match = (evaluation.expected_outcome == evaluation.actual_outcome)
        
        if evaluation.outcome_match:
            score = 100.0
        else:
            # Partial credit based on severity of mismatch
            expected = evaluation.expected_outcome
            actual = evaluation.actual_outcome
            
            # Emergency missed = critical failure
            if expected == MedicalOutcome.EMERGENCY and actual != MedicalOutcome.EMERGENCY:
                score = 0.0
                evaluation.critical_issues.append("CRITICAL: Emergency condition not detected")
            
            # False emergency = serious but not critical
            elif expected != MedicalOutcome.EMERGENCY and actual == MedicalOutcome.EMERGENCY:
                score = 30.0
                evaluation.weaknesses.append("False emergency detection")
            
            # Routine vs self-care mismatch = moderate issue
            elif ((expected == MedicalOutcome.ROUTINE_DOCTOR and actual == MedicalOutcome.SELF_CARE) or
                  (expected == MedicalOutcome.SELF_CARE and actual == MedicalOutcome.ROUTINE_DOCTOR)):
                score = 60.0
                evaluation.weaknesses.append("Minor outcome classification error")
            
            else:
                score = 20.0  # Other mismatches
        
        evaluation.scores[EvaluationCriteria.OUTCOME_ACCURACY] = score
        
        if score == 100.0:
            evaluation.strengths.append("Perfect outcome accuracy")
    
    async def _evaluate_emergency_detection(self, evaluation: ConversationEvaluation, condition: MedicalCondition):
        """Evaluate emergency detection capability"""
        
        is_emergency = condition.expected_outcome == MedicalOutcome.EMERGENCY
        detected_emergency = evaluation.actual_outcome == MedicalOutcome.EMERGENCY
        
        if is_emergency and detected_emergency:
            # True positive - excellent
            base_score = 100.0
            
            # Bonus for quick detection
            if evaluation.total_turns <= condition.minimum_turns_expected + 1:
                base_score = min(100.0, base_score + 5.0)
                evaluation.strengths.append("Quick emergency detection")
            
            # Check if it was a critical emergency
            if condition.emergency_detection_critical:
                evaluation.strengths.append("Critical emergency correctly identified")
            
            score = base_score
            
        elif is_emergency and not detected_emergency:
            # False negative - critical failure
            score = 0.0
            evaluation.critical_issues.append("CRITICAL: Emergency missed - patient safety risk")
            
        elif not is_emergency and detected_emergency:
            # False positive - problematic but not critical
            score = 40.0
            evaluation.weaknesses.append("False emergency alert")
            
        else:
            # True negative - good
            score = 90.0
            evaluation.strengths.append("Correctly identified non-emergency")
        
        evaluation.scores[EvaluationCriteria.EMERGENCY_DETECTION] = score
    
    async def _evaluate_turn_efficiency(self, evaluation: ConversationEvaluation, condition: MedicalCondition):
        """Evaluate conversation turn efficiency"""
        
        expected_min = condition.minimum_turns_expected
        expected_max = condition.maximum_turns_acceptable
        actual_turns = evaluation.total_turns
        
        if actual_turns <= expected_min:
            # Too fast might indicate rushing
            score = 80.0
            evaluation.weaknesses.append("Conversation may have been too brief")
            
        elif actual_turns <= expected_max:
            # Within acceptable range
            optimal_turns = (expected_min + expected_max) / 2
            distance_from_optimal = abs(actual_turns - optimal_turns) / optimal_turns
            score = max(70.0, 100.0 - (distance_from_optimal * 30))
            
            if score > 90:
                evaluation.strengths.append("Efficient conversation length")
                
        else:
            # Too long - efficiency problem
            excess_turns = actual_turns - expected_max
            penalty = min(50.0, excess_turns * 10)
            score = max(20.0, 70.0 - penalty)
            evaluation.weaknesses.append(f"Conversation too long ({actual_turns} turns)")
        
        evaluation.scores[EvaluationCriteria.TURN_EFFICIENCY] = score
    
    async def _evaluate_red_flag_detection(self, evaluation: ConversationEvaluation, agent_results: List[Dict[str, Any]]):
        """Evaluate red flag detection accuracy"""
        
        # Collect all detected red flags across conversation
        all_detected_flags = set()
        for result in agent_results:
            if "red_flags" in result:
                all_detected_flags.update(result["red_flags"])
        
        evaluation.detected_red_flags = list(all_detected_flags)
        
        expected_flags = set(evaluation.expected_red_flags)
        detected_flags = set(evaluation.detected_red_flags)
        
        # Calculate metrics
        true_positives = expected_flags.intersection(detected_flags)
        false_negatives = expected_flags - detected_flags
        false_positives = detected_flags - expected_flags
        
        evaluation.missed_red_flags = list(false_negatives)
        evaluation.false_positive_flags = list(false_positives)
        
        # Calculate score
        if not expected_flags:
            # No red flags expected
            if not detected_flags:
                score = 100.0  # Perfect - no flags expected or detected
            else:
                score = max(50.0, 100.0 - len(false_positives) * 10)  # Penalty for false positives
        else:
            # Red flags expected
            recall = len(true_positives) / len(expected_flags) if expected_flags else 0
            precision = len(true_positives) / len(detected_flags) if detected_flags else 0
            
            if precision == 0 and recall == 0:
                score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                score = f1_score * 100
        
        evaluation.scores[EvaluationCriteria.RED_FLAG_DETECTION] = score
        
        # Add qualitative feedback
        if score > 90:
            evaluation.strengths.append("Excellent red flag detection")
        elif len(false_negatives) > 0:
            evaluation.weaknesses.append(f"Missed {len(false_negatives)} red flags")
        if len(false_positives) > 2:
            evaluation.weaknesses.append("Too many false positive red flags")
    
    async def _evaluate_confidence_calibration(self, evaluation: ConversationEvaluation, condition: MedicalCondition):
        """Evaluate confidence calibration"""
        
        final_confidence = evaluation.final_confidence
        expected_threshold = condition.confidence_threshold
        outcome_correct = evaluation.outcome_match
        
        # Good calibration means:
        # - High confidence when correct
        # - Low confidence when wrong
        # - Confidence above threshold when outcome is correct
        
        if outcome_correct:
            if final_confidence >= expected_threshold:
                score = 100.0
                evaluation.strengths.append("Well-calibrated confidence")
            else:
                # Underconfident when correct
                confidence_gap = expected_threshold - final_confidence
                penalty = min(40, confidence_gap * 2)
                score = max(60.0, 100.0 - penalty)
                evaluation.weaknesses.append("Underconfident despite correct outcome")
        else:
            if final_confidence < expected_threshold:
                score = 80.0  # Good - low confidence when wrong
                evaluation.strengths.append("Appropriately low confidence for wrong outcome")
            else:
                # Overconfident when wrong
                confidence_excess = final_confidence - expected_threshold
                penalty = min(60, confidence_excess * 3)
                score = max(20.0, 80.0 - penalty)
                evaluation.weaknesses.append("Overconfident despite wrong outcome")
        
        evaluation.scores[EvaluationCriteria.CONFIDENCE_CALIBRATION] = score
    
    async def _evaluate_protocol_alignment(self, evaluation: ConversationEvaluation):
        """Evaluate alignment with NICE protocols"""
        
        expected_protocols = set(evaluation.relevant_protocols)
        used_protocols = set(evaluation.used_protocols)
        
        if not expected_protocols:
            score = 90.0  # No specific protocols required
        else:
            overlap = expected_protocols.intersection(used_protocols)
            coverage = len(overlap) / len(expected_protocols)
            
            # Base score on coverage
            score = coverage * 100
            
            # Bonus for using exactly the right protocols
            if coverage == 1.0 and len(used_protocols) == len(expected_protocols):
                score = min(100.0, score + 10)
                evaluation.strengths.append("Perfect protocol alignment")
            
            # Penalty for using too many irrelevant protocols
            irrelevant_protocols = used_protocols - expected_protocols
            if len(irrelevant_protocols) > 2:
                penalty = min(20, len(irrelevant_protocols) * 5)
                score = max(score - penalty, 40.0)
                evaluation.weaknesses.append("Used too many irrelevant protocols")
        
        evaluation.protocol_alignment_score = score
        evaluation.scores[EvaluationCriteria.NICE_PROTOCOL_ALIGNMENT] = score
    
    async def _evaluate_patient_experience(self, evaluation: ConversationEvaluation, profile: PatientProfile):
        """Evaluate patient experience quality"""
        
        # This is subjective - base on conversation characteristics
        score = 75.0  # Start with neutral
        
        # Positive factors
        if evaluation.total_turns <= 6:
            score += 5  # Efficient
        
        if evaluation.final_confidence > 80:
            score += 5  # Confident agent is reassuring
        
        if len(evaluation.critical_issues) == 0:
            score += 10  # No critical errors
        
        # Negative factors
        if evaluation.total_turns > 10:
            score -= 10  # Too long
        
        if len(evaluation.critical_issues) > 0:
            score -= 20  # Critical issues affect experience
        
        if evaluation.final_confidence < 50:
            score -= 10  # Uncertain agent is concerning
        
        # Adjust for patient characteristics
        if profile.health_anxiety_level > 80:
            # High anxiety patients need more reassurance
            if evaluation.final_confidence > 85:
                score += 5  # Good for anxious patients
            else:
                score -= 5  # Uncertainty worse for anxious patients
        
        score = max(0.0, min(100.0, score))
        evaluation.scores[EvaluationCriteria.PATIENT_EXPERIENCE] = score
        
        if score > 90:
            evaluation.strengths.append("Excellent patient experience")
        elif score < 60:
            evaluation.weaknesses.append("Poor patient experience")
    
    async def _evaluate_communication_quality(self, evaluation: ConversationEvaluation):
        """Evaluate communication quality"""
        
        # This is subjective - base on conversation flow
        score = 80.0  # Start with good baseline
        
        # Check for good questioning patterns
        agent_questions = evaluation.agent_questions
        
        if len(agent_questions) > 0:
            # Variety in questions is good
            unique_question_starts = set()
            for q in agent_questions:
                if q:
                    start = q.split()[:2] if len(q.split()) >= 2 else [q.split()[0]]
                    unique_question_starts.add(' '.join(start).lower())
            
            variety_ratio = len(unique_question_starts) / len(agent_questions)
            if variety_ratio > 0.8:
                score += 10
                evaluation.strengths.append("Good question variety")
            elif variety_ratio < 0.5:
                score -= 10
                evaluation.weaknesses.append("Repetitive questioning")
        
        # Penalize if conversation was too repetitive or stuck
        if evaluation.total_turns > 8:
            # Check for loops or getting stuck
            score -= 5
        
        score = max(0.0, min(100.0, score))
        evaluation.scores[EvaluationCriteria.COMMUNICATION_QUALITY] = score
    
    def _extract_actual_outcome(self, final_result: Dict[str, Any]) -> MedicalOutcome:
        """Extract actual outcome from agent results"""
        
        outcome_str = final_result.get("outcome", "inconclusive")
        
        try:
            # Map string outcomes to enum
            outcome_mapping = {
                "emergency": MedicalOutcome.EMERGENCY,
                "emergency_route_to_doctor": MedicalOutcome.EMERGENCY,
                "routine": MedicalOutcome.ROUTINE_DOCTOR,
                "routine_doctor": MedicalOutcome.ROUTINE_DOCTOR,
                "routine_doctor_consultation": MedicalOutcome.ROUTINE_DOCTOR,
                "self_care": MedicalOutcome.SELF_CARE,
                "self_care_advice": MedicalOutcome.SELF_CARE,
                "inconclusive": MedicalOutcome.INCONCLUSIVE,
                "need_more_questions": MedicalOutcome.INCONCLUSIVE,
                "spam": MedicalOutcome.SPAM,
                "spam_or_irrelevant": MedicalOutcome.SPAM
            }
            
            return outcome_mapping.get(outcome_str.lower(), MedicalOutcome.INCONCLUSIVE)
            
        except Exception:
            return MedicalOutcome.INCONCLUSIVE
    
    def _calculate_overall_score(self, evaluation: ConversationEvaluation) -> float:
        """Calculate weighted overall score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for criteria, weight in self.evaluation_weights.items():
            if criteria in evaluation.scores:
                total_score += evaluation.scores[criteria] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall = total_score / total_weight
        
        # Apply critical failure penalties
        if len(evaluation.critical_issues) > 0:
            # Each critical issue reduces overall score significantly
            penalty = min(40.0, len(evaluation.critical_issues) * 20)
            overall = max(0.0, overall - penalty)
        
        return overall
    
    def _identify_critical_issues(self, evaluation: ConversationEvaluation, condition: MedicalCondition):
        """Identify critical issues that constitute test failures"""
        
        # Emergency detection failures
        if (condition.expected_outcome == MedicalOutcome.EMERGENCY and 
            evaluation.actual_outcome != MedicalOutcome.EMERGENCY):
            evaluation.critical_issues.append("CRITICAL: Emergency condition not detected")
        
        # Critical emergency failures
        if (condition.emergency_detection_critical and 
            evaluation.scores.get(EvaluationCriteria.EMERGENCY_DETECTION, 0) < self.critical_thresholds["critical_emergency_minimum"]):
            evaluation.critical_issues.append("CRITICAL: Critical emergency detection failed")
        
        # Conversation timeout
        if evaluation.total_turns > self.critical_thresholds["maximum_acceptable_turns"]:
            evaluation.critical_issues.append("CRITICAL: Conversation exceeded maximum turns")
        
        # Overall failure
        if evaluation.overall_score < self.critical_thresholds["overall_minimum_passing"]:
            evaluation.critical_issues.append("CRITICAL: Overall performance below minimum threshold")
    
    def _generate_qualitative_assessment(self, evaluation: ConversationEvaluation, 
                                       condition: MedicalCondition, profile: PatientProfile):
        """Generate qualitative strengths and weaknesses"""
        
        # Add context-specific feedback
        if condition.expected_outcome == MedicalOutcome.EMERGENCY:
            if evaluation.outcome_match and evaluation.total_turns <= 3:
                evaluation.strengths.append("Rapid emergency identification")
            elif not evaluation.outcome_match:
                evaluation.weaknesses.append("Failed to identify emergency condition")
        
        # Communication style feedback
        if profile.communication_style.english_proficiency == "basic":
            if evaluation.scores.get(EvaluationCriteria.COMMUNICATION_QUALITY, 0) > 80:
                evaluation.strengths.append("Good communication with basic English speaker")
        
        # Anxiety handling
        if profile.health_anxiety_level > 80:
            if evaluation.scores.get(EvaluationCriteria.PATIENT_EXPERIENCE, 0) > 80:
                evaluation.strengths.append("Effective handling of anxious patient")
            else:
                evaluation.weaknesses.append("Could improve anxiety management")
        
        # Generate summary note
        if evaluation.overall_score >= 90:
            evaluation.evaluation_notes = "Excellent performance across all metrics"
        elif evaluation.overall_score >= 80:
            evaluation.evaluation_notes = "Good performance with minor areas for improvement"
        elif evaluation.overall_score >= 70:
            evaluation.evaluation_notes = "Acceptable performance but needs improvement"
        elif evaluation.overall_score >= 60:
            evaluation.evaluation_notes = "Marginal performance - significant issues identified"
        else:
            evaluation.evaluation_notes = "Poor performance - major issues require attention"
    
    def is_passing_score(self, evaluation: ConversationEvaluation) -> bool:
        """Determine if evaluation represents a passing test"""
        
        # Must meet minimum overall score
        if evaluation.overall_score < self.critical_thresholds["overall_minimum_passing"]:
            return False
        
        # Must not have critical issues
        if len(evaluation.critical_issues) > 0:
            return False
        
        # Emergency detection must be excellent
        if (evaluation.expected_outcome == MedicalOutcome.EMERGENCY and 
            evaluation.scores.get(EvaluationCriteria.EMERGENCY_DETECTION, 0) < self.critical_thresholds["emergency_detection_minimum"]):
            return False
        
        return True

async def evaluate_conversation_batch(evaluator: ConversationEvaluator,
                                    conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate batch of conversations and return aggregate results"""
    
    logger.info("📊 Starting batch evaluation", count=len(conversations))
    
    evaluations = []
    for conv_data in conversations:
        try:
            evaluation = await evaluator.evaluate_conversation(
                conv_data["patient_id"],
                conv_data["conversation_data"],
                conv_data["agent_results"]
            )
            evaluations.append(evaluation)
        except Exception as e:
            logger.error("❌ Conversation evaluation failed", error=str(e))
    
    # Aggregate results
    if not evaluations:
        return {"error": "No successful evaluations"}
    
    total_score = sum(e.overall_score for e in evaluations)
    average_score = total_score / len(evaluations)
    
    passing_count = sum(1 for e in evaluations if evaluator.is_passing_score(e))
    passing_rate = passing_count / len(evaluations) * 100
    
    # Category breakdown
    emergency_evals = [e for e in evaluations if e.expected_outcome == MedicalOutcome.EMERGENCY]
    routine_evals = [e for e in evaluations if e.expected_outcome == MedicalOutcome.ROUTINE_DOCTOR]
    self_care_evals = [e for e in evaluations if e.expected_outcome == MedicalOutcome.SELF_CARE]
    
    category_scores = {
        "emergency": sum(e.overall_score for e in emergency_evals) / len(emergency_evals) if emergency_evals else 0,
        "routine": sum(e.overall_score for e in routine_evals) / len(routine_evals) if routine_evals else 0,
        "self_care": sum(e.overall_score for e in self_care_evals) / len(self_care_evals) if self_care_evals else 0
    }
    
    # Critical issue summary
    all_critical_issues = []
    for e in evaluations:
        all_critical_issues.extend(e.critical_issues)
    
    return {
        "total_conversations": len(evaluations),
        "average_score": average_score,
        "passing_rate": passing_rate,
        "passing_count": passing_count,
        "failed_count": len(evaluations) - passing_count,
        "category_scores": category_scores,
        "critical_issues_count": len(all_critical_issues),
        "detailed_evaluations": evaluations,
        "summary": f"Average: {average_score:.1f}%, Passing: {passing_rate:.1f}%"
    }
