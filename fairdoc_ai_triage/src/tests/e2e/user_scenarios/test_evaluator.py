# Test Results Evaluator - E2E Test Evaluation and Scoring
"""
Evaluates E2E conversation test results using fuzzy logic and comprehensive scoring
Provides detailed analysis of DSPy medical agent performance
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from patient_profiles import get_patient_profile
from medical_conditions import get_condition
from conversation_orchestrator import ConversationResult, ConversationMetrics

logger = structlog.get_logger(__name__)

class ScoreCategory(str, Enum):
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    SATISFACTORY = "satisfactory"  # 60-74
    POOR = "poor"          # 40-59
    FAILED = "failed"      # 0-39

@dataclass
class EvaluationCriteria:
    """Comprehensive evaluation criteria for E2E tests"""
    
    # Core medical accuracy (40 points)
    correct_outcome: int = 25      # Correct final medical outcome
    emergency_detection: int = 15  # Emergency correctly identified/ruled out
    
    # Clinical reasoning (30 points) 
    red_flag_detection: int = 10   # Appropriate red flags identified
    nice_protocol_usage: int = 10  # Relevant NICE protocols used
    confidence_accuracy: int = 10  # Confidence scores match situation
    
    # Conversation quality (20 points)
    appropriate_turns: int = 8   # Reasonable number of conversation turns
    question_relevance: int = 7  # Medical questions are relevant
    patient_experience: int = 5  # Patient satisfaction estimated
    
    # Efficiency (10 points)
    time_to_decision: int = 5     # Reasonable time to reach decision
    resource_efficiency: int = 5  # Efficient use of conversation turns
    
    @property
    def total_points(self) -> int:
        return sum([
            self.correct_outcome, self.emergency_detection,
            self.red_flag_detection, self.nice_protocol_usage, self.confidence_accuracy,
            self.appropriate_turns, self.question_relevance, self.patient_experience,
            self.time_to_decision, self.resource_efficiency
        ])

@dataclass
class DetailedEvaluation:
    """Detailed evaluation result for single conversation"""
    patient_id: str
    conversation_result: ConversationResult
    
    # Individual category scores
    medical_accuracy_score: float
    clinical_reasoning_score: float
    conversation_quality_score: float
    efficiency_score: float
    
    # Overall evaluation
    total_score: float
    category: ScoreCategory
    passed: bool
    
    # Detailed feedback
    strengths: List[str]
    weaknesses: List[str]
    critical_issues: List[str]
    recommendations: List[str]

class FuzzyLogicEvaluator:
    """Applies fuzzy logic to evaluate conversation quality"""
    
    def __init__(self):
        self.criteria = EvaluationCriteria()
    
    def evaluate_conversation(self, result: ConversationResult) -> DetailedEvaluation:
        """Comprehensive evaluation of single conversation"""
        
        logger.info("📊 Evaluating conversation",
                   patient_id=result.patient_id,
                   turns=result.metrics.total_turns,
                   outcome=result.metrics.final_outcome)
        
        patient_profile = get_patient_profile(result.patient_id)
        medical_condition = get_condition(patient_profile.medical_condition)
        
        # Evaluate each category
        medical_accuracy = self._evaluate_medical_accuracy(result, medical_condition)
        clinical_reasoning = self._evaluate_clinical_reasoning(result, medical_condition)
        conversation_quality = self._evaluate_conversation_quality(result, medical_condition)
        efficiency = self._evaluate_efficiency(result, medical_condition)
        
        # Calculate total score
        total_score = (
            medical_accuracy.score + clinical_reasoning.score + 
            conversation_quality.score + efficiency.score
        )
        
        # Determine category and pass/fail
        category = self._score_to_category(total_score)
        passed = total_score >= 60 and not self._has_critical_failures(result, medical_condition)
        
        # Collect feedback
        strengths = medical_accuracy.strengths + clinical_reasoning.strengths + conversation_quality.strengths + efficiency.strengths
        weaknesses = medical_accuracy.weaknesses + clinical_reasoning.weaknesses + conversation_quality.weaknesses + efficiency.weaknesses
        critical_issues = self._identify_critical_issues(result, medical_condition)
        recommendations = self._generate_recommendations(result, medical_condition, total_score)
        
        return DetailedEvaluation(
            patient_id=result.patient_id,
            conversation_result=result,
            medical_accuracy_score=medical_accuracy.score,
            clinical_reasoning_score=clinical_reasoning.score,
            conversation_quality_score=conversation_quality.score,
            efficiency_score=efficiency.score,
            total_score=total_score,
            category=category,
            passed=passed,
            strengths=strengths,
            weaknesses=weaknesses,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _evaluate_medical_accuracy(self, result: ConversationResult, condition) -> "CategoryEvaluation":
        """Evaluate medical accuracy (40 points)"""
        
        score = 0.0
        strengths = []
        weaknesses = []
        
        # Correct outcome (25 points)
        if result.metrics.final_outcome == condition.expected_outcome.value:
            score += 25
            strengths.append("Correct final medical outcome identified")
        else:
            # Partial credit for related outcomes
            if condition.expected_outcome.value == "emergency_route_to_doctor":
                if result.metrics.final_outcome == "routine_doctor_consultation":
                    score += 10  # At least referred to doctor
                    weaknesses.append("Emergency treated as routine - serious issue")
                else:
                    weaknesses.append("Emergency completely missed - critical failure")
            elif condition.expected_outcome.value == "routine_doctor_consultation":
                if result.metrics.final_outcome == "emergency_route_to_doctor":
                    score += 15  # Over-cautious but safe
                    weaknesses.append("Routine case treated as emergency - over-cautious")
                elif result.metrics.final_outcome == "self_care_advice":
                    score += 5   # Missed need for doctor
                    weaknesses.append("Medical consultation needed but not recommended")
            elif condition.expected_outcome.value == "self_care_advice":
                if result.metrics.final_outcome == "routine_doctor_consultation":
                    score += 15  # Over-cautious but acceptable
                    strengths.append("Conservative approach - safe if not efficient")
        
        # Emergency detection (15 points)
        if result.metrics.emergency_correctly_identified:
            score += 15
            strengths.append("Emergency situations correctly identified")
        else:
            if condition.emergency_detection_critical:
                weaknesses.append("Critical emergency detection failure")
            else:
                score += 8  # Partial credit for non-critical cases
        
        return CategoryEvaluation(score, strengths, weaknesses)
    
    def _evaluate_clinical_reasoning(self, result: ConversationResult, condition) -> "CategoryEvaluation":
        """Evaluate clinical reasoning (30 points)"""
        
        score = 0.0
        strengths = []
        weaknesses = []
        
        # Red flag detection (10 points)
        expected_flags = set(condition.red_flag_indicators)
        detected_flags = set(result.metrics.red_flags_detected)
        
        if expected_flags:
            detection_rate = len(expected_flags & detected_flags) / len(expected_flags)
            red_flag_score = 10 * detection_rate
            score += red_flag_score
            
            if detection_rate >= 0.8:
                strengths.append("Excellent red flag detection")
            elif detection_rate >= 0.5:
                strengths.append("Good red flag detection")
            else:
                weaknesses.append("Poor red flag detection")
        else:
            # No false positives expected
            if not detected_flags:
                score += 10
                strengths.append("No inappropriate red flags detected")
            else:
                score += 7  # Minor penalty for false positives
                weaknesses.append("Some unnecessary red flags detected")
        
        # NICE protocol usage (10 points)
        expected_protocols = set(condition.relevant_protocols)
        used_protocols = set(result.metrics.nice_protocols_triggered)
        
        if expected_protocols:
            protocol_rate = len(expected_protocols & used_protocols) / len(expected_protocols)
            protocol_score = 10 * protocol_rate
            score += protocol_score
            
            if protocol_rate >= 0.7:
                strengths.append("Appropriate NICE protocols followed")
            else:
                weaknesses.append("NICE protocol usage could be improved")
        else:
            score += 8  # Default score if no specific protocols expected
        
        # Confidence accuracy (10 points)
        if result.metrics.confidence_scores:
            final_confidence = result.metrics.confidence_scores[-1]
            
            # High confidence should match correct outcomes
            if result.metrics.final_outcome == condition.expected_outcome.value:
                if final_confidence >= 80:
                    score += 10
                    strengths.append("High confidence with correct outcome")
                elif final_confidence >= 60:
                    score += 7
                    strengths.append("Reasonable confidence with correct outcome")
                else:
                    score += 4
                    weaknesses.append("Low confidence despite correct outcome")
            else:
                # Wrong outcome should have lower confidence
                if final_confidence < 60:
                    score += 6  # At least recognized uncertainty
                    strengths.append("Appropriately uncertain about wrong outcome")
                else:
                    score += 2
                    weaknesses.append("High confidence with wrong outcome")
        
        return CategoryEvaluation(score, strengths, weaknesses)
    
    def _evaluate_conversation_quality(self, result: ConversationResult, condition) -> "CategoryEvaluation":
        """Evaluate conversation quality (20 points)"""
        
        score = 0.0
        strengths = []
        weaknesses = []
        
        # Appropriate turns (8 points)
        expected_min = condition.minimum_turns_expected
        expected_max = condition.maximum_turns_acceptable
        actual_turns = result.metrics.total_turns
        
        if expected_min <= actual_turns <= expected_max:
            score += 8
            strengths.append("Appropriate number of conversation turns")
        elif actual_turns < expected_min:
            score += 4  # Too quick
            weaknesses.append("Conversation ended too quickly - may lack thoroughness")
        elif actual_turns <= expected_max + 2:
            score += 6  # Slightly long but acceptable
            strengths.append("Thorough conversation, slightly longer than optimal")
        else:
            score += 2  # Too long
            weaknesses.append("Conversation too lengthy - inefficient")
        
        # Question relevance (7 points)
        relevant_questions = 0
        total_questions = len(result.metrics.medical_agent_questions)
        
        if total_questions > 0:
            # Analyze question relevance (simplified heuristic)
            for question in result.metrics.medical_agent_questions:
                if any(keyword in question.lower() for keyword in [
                    'pain', 'symptom', 'when', 'how', 'where', 'severity',
                    'medication', 'history', 'emergency', 'doctor'
                ]):
                    relevant_questions += 1
            
            relevance_rate = relevant_questions / total_questions
            score += 7 * relevance_rate
            
            if relevance_rate >= 0.8:
                strengths.append("Highly relevant medical questions")
            elif relevance_rate >= 0.6:
                strengths.append("Generally relevant questions")
            else:
                weaknesses.append("Some irrelevant or poorly focused questions")
        
        # Patient experience (5 points)
        satisfaction = result.metrics.patient_satisfaction_estimated
        if satisfaction >= 80:
            score += 5
            strengths.append("Excellent estimated patient experience")
        elif satisfaction >= 60:
            score += 4
            strengths.append("Good patient experience")
        elif satisfaction >= 40:
            score += 2
            weaknesses.append("Patient experience could be improved")
        else:
            score += 1
            weaknesses.append("Poor estimated patient experience")
        
        return CategoryEvaluation(score, strengths, weaknesses)
    
    def _evaluate_efficiency(self, result: ConversationResult, condition) -> "CategoryEvaluation":
        """Evaluate efficiency (10 points)"""
        
        score = 0.0
        strengths = []
        weaknesses = []
        
        # Time to decision (5 points)
        duration = result.metrics.conversation_duration_seconds
        
        if condition.expected_outcome.value == "emergency_route_to_doctor":
            # Emergency should be quick
            if duration <= 60:  # 1 minute
                score += 5
                strengths.append("Rapid emergency identification")
            elif duration <= 120:  # 2 minutes
                score += 3
                strengths.append("Quick emergency identification")
            else:
                score += 1
                weaknesses.append("Emergency took too long to identify")
        else:
            # Non-emergency should be reasonable
            if duration <= 180:  # 3 minutes
                score += 5
                strengths.append("Efficient consultation time")
            elif duration <= 300:  # 5 minutes
                score += 4
                strengths.append("Reasonable consultation time")
            else:
                score += 2
                weaknesses.append("Consultation took longer than optimal")
        
        # Resource efficiency (5 points)
        turns_per_minute = result.metrics.total_turns / (duration / 60) if duration > 0 else 0
        
        if 0.5 <= turns_per_minute <= 2.0:  # Reasonable pace
            score += 5
            strengths.append("Good conversation pacing")
        elif turns_per_minute > 2.0:
            score += 3
            weaknesses.append("Conversation pace too rapid")
        elif turns_per_minute > 0:
            score += 3
            weaknesses.append("Conversation pace too slow")
        
        return CategoryEvaluation(score, strengths, weaknesses)
    
    def _score_to_category(self, score: float) -> ScoreCategory:
        """Convert numerical score to category"""
        if score >= 90:
            return ScoreCategory.EXCELLENT
        elif score >= 75:
            return ScoreCategory.GOOD
        elif score >= 60:
            return ScoreCategory.SATISFACTORY
        elif score >= 40:
            return ScoreCategory.POOR
        else:
            return ScoreCategory.FAILED
    
    def _has_critical_failures(self, result: ConversationResult, condition) -> bool:
        """Check for critical failures that override score"""
        
        # Emergency missed completely
        if (condition.expected_outcome.value == "emergency_route_to_doctor" and
            result.metrics.final_outcome not in ["emergency_route_to_doctor", "routine_doctor_consultation"]):
            return True
        
        # Critical emergency detection failure
        if condition.emergency_detection_critical and not result.metrics.emergency_correctly_identified:
            return True
        
        # Conversation failed to complete
        if result.metrics.total_turns == 0:
            return True
        
        return False
    
    def _identify_critical_issues(self, result: ConversationResult, condition) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        
        issues = []
        
        if condition.emergency_detection_critical and not result.metrics.emergency_correctly_identified:
            issues.append("CRITICAL: Emergency condition not detected")
        
        if (condition.expected_outcome.value == "emergency_route_to_doctor" and
            result.metrics.final_outcome == "self_care_advice"):
            issues.append("CRITICAL: Emergency treated as self-care")
        
        if result.metrics.total_turns > condition.maximum_turns_acceptable + 3:
            issues.append("CRITICAL: Excessive conversation length")
        
        if not result.metrics.confidence_scores or max(result.metrics.confidence_scores) < 30:
            issues.append("CRITICAL: Extremely low confidence throughout")
        
        return issues
    
    def _generate_recommendations(self, result: ConversationResult, condition, score: float) -> List[str]:
        """Generate specific improvement recommendations"""
        
        recommendations = []
        
        if score < 60:
            recommendations.append("Overall performance below threshold - requires DSPy model optimization")
        
        if not result.metrics.emergency_correctly_identified and condition.emergency_detection_critical:
            recommendations.append("Improve emergency detection - review red flag training data")
        
        if result.metrics.total_turns > condition.maximum_turns_acceptable:
            recommendations.append("Optimize question generation to reduce conversation length")
        
        if len(result.metrics.red_flags_detected) < len(condition.red_flag_indicators) / 2:
            recommendations.append("Enhance red flag detection sensitivity")
        
        if result.metrics.patient_satisfaction_estimated < 50:
            recommendations.append("Improve patient communication and empathy in responses")
        
        confidence_trend = self._analyze_confidence_trend(result.metrics.confidence_scores)
        if confidence_trend == "decreasing":
            recommendations.append("Review conversation flow - confidence should generally increase")
        
        if not recommendations:
            recommendations.append("Performance acceptable - continue monitoring and fine-tuning")
        
        return recommendations
    
    def _analyze_confidence_trend(self, confidence_scores: List[float]) -> str:
        """Analyze trend in confidence scores"""
        if len(confidence_scores) < 2:
            return "insufficient_data"
        
        if confidence_scores[-1] > confidence_scores[0] + 10:
            return "increasing"
        elif confidence_scores[-1] < confidence_scores[0] - 10:
            return "decreasing"
        else:
            return "stable"

@dataclass
class CategoryEvaluation:
    """Evaluation result for a single category"""
    score: float
    strengths: List[str]
    weaknesses: List[str]

class BatchEvaluator:
    """Evaluates multiple conversation results in batch"""
    
    def __init__(self):
        self.fuzzy_evaluator = FuzzyLogicEvaluator()
    
    def evaluate_batch(self, results: List[ConversationResult]) -> Dict[str, Any]:
        """Evaluate batch of conversation results"""
        
        logger.info("📊 Starting batch evaluation", total_conversations=len(results))
        
        detailed_evaluations = []
        
        for result in results:
            try:
                evaluation = self.fuzzy_evaluator.evaluate_conversation(result)
                detailed_evaluations.append(evaluation)
            except Exception as e:
                logger.error("❌ Evaluation failed", patient_id=result.patient_id, error=str(e))
        
        # Generate aggregate statistics
        aggregate_stats = self._generate_aggregate_statistics(detailed_evaluations)
        performance_analysis = self._analyze_performance_patterns(detailed_evaluations)
        final_recommendations = self._generate_final_recommendations(detailed_evaluations)
        
        logger.info("✅ Batch evaluation completed",
                   evaluated=len(detailed_evaluations),
                   average_score=aggregate_stats.get("average_total_score", 0))
        
        return {
            "detailed_evaluations": detailed_evaluations,
            "aggregate_statistics": aggregate_stats,
            "performance_analysis": performance_analysis,
            "recommendations": final_recommendations
        }
    
    def _generate_aggregate_statistics(self, evaluations: List[DetailedEvaluation]) -> Dict[str, Any]:
        """Generate aggregate statistics across all evaluations"""
        
        if not evaluations:
            return {}
        
        # Score statistics
        total_scores = [e.total_score for e in evaluations]
        medical_scores = [e.medical_accuracy_score for e in evaluations]
        reasoning_scores = [e.clinical_reasoning_score for e in evaluations]
        quality_scores = [e.conversation_quality_score for e in evaluations]
        efficiency_scores = [e.efficiency_score for e in evaluations]
        
        # Category distribution
        category_dist = {}
        for e in evaluations:
            category_dist[e.category.value] = category_dist.get(e.category.value, 0) + 1
        
        # Pass/fail analysis
        passed = len([e for e in evaluations if e.passed])
        
        return {
            "total_evaluations": len(evaluations),
            "passed_evaluations": passed,
            "pass_rate": passed / len(evaluations) * 100,
            
            "score_statistics": {
                "average_total_score": sum(total_scores) / len(total_scores),
                "min_total_score": min(total_scores),
                "max_total_score": max(total_scores),
                "average_medical_accuracy": sum(medical_scores) / len(medical_scores),
                "average_clinical_reasoning": sum(reasoning_scores) / len(reasoning_scores),
                "average_conversation_quality": sum(quality_scores) / len(quality_scores),
                "average_efficiency": sum(efficiency_scores) / len(efficiency_scores)
            },
            
            "category_distribution": category_dist,
            
            "critical_issues_count": sum(len(e.critical_issues) for e in evaluations)
        }
    
    def _analyze_performance_patterns(self, evaluations: List[DetailedEvaluation]) -> Dict[str, Any]:
        """Analyze patterns in performance across evaluations"""
        
        patterns = {
            "emergency_detection": {"total": 0, "correct": 0},
            "conversation_length": {"too_short": 0, "optimal": 0, "too_long": 0},
            "confidence_patterns": {"low_start_high_end": 0, "consistently_low": 0, "consistently_high": 0},
            "common_weaknesses": {},
            "common_strengths": {}
        }
        
        for eval_result in evaluations:
            result = eval_result.conversation_result
            condition = get_condition(get_patient_profile(result.patient_id).medical_condition)
            
            # Emergency detection analysis
            if condition.expected_outcome.value == "emergency_route_to_doctor":
                patterns["emergency_detection"]["total"] += 1
                if result.metrics.emergency_correctly_identified:
                    patterns["emergency_detection"]["correct"] += 1
            
            # Conversation length patterns
            if result.metrics.total_turns < condition.minimum_turns_expected:
                patterns["conversation_length"]["too_short"] += 1
            elif result.metrics.total_turns <= condition.maximum_turns_acceptable:
                patterns["conversation_length"]["optimal"] += 1
            else:
                patterns["conversation_length"]["too_long"] += 1
            
            # Confidence patterns
            if len(result.metrics.confidence_scores) >= 2:
                start_confidence = result.metrics.confidence_scores[0]
                end_confidence = result.metrics.confidence_scores[-1]
                
                if start_confidence < 50 and end_confidence > 70:
                    patterns["confidence_patterns"]["low_start_high_end"] += 1
                elif max(result.metrics.confidence_scores) < 50:
                    patterns["confidence_patterns"]["consistently_low"] += 1
                elif min(result.metrics.confidence_scores) > 70:
                    patterns["confidence_patterns"]["consistently_high"] += 1
            
            # Aggregate strengths and weaknesses
            for weakness in eval_result.weaknesses:
                patterns["common_weaknesses"][weakness] = patterns["common_weaknesses"].get(weakness, 0) + 1
            
            for strength in eval_result.strengths:
                patterns["common_strengths"][strength] = patterns["common_strengths"].get(strength, 0) + 1
        
        return patterns
    
    def _generate_final_recommendations(self, evaluations: List[DetailedEvaluation]) -> List[str]:
        """Generate final recommendations based on all evaluations"""
        
        recommendations = []
        
        if not evaluations:
            return ["No evaluations available for analysis"]
        
        pass_rate = len([e for e in evaluations if e.passed]) / len(evaluations)
        
        if pass_rate < 0.6:
            recommendations.append("❗ CRITICAL: Overall pass rate below 60% - requires immediate DSPy model retraining")
        elif pass_rate < 0.8:
            recommendations.append("⚠️ Pass rate below 80% - consider DSPy optimization and additional training data")
        
        # Emergency detection issues
        emergency_evals = [e for e in evaluations 
                          if get_condition(get_patient_profile(e.patient_id).medical_condition).expected_outcome.value == "emergency_route_to_doctor"]
        if emergency_evals:
            emergency_pass_rate = len([e for e in emergency_evals if e.passed]) / len(emergency_evals)
            if emergency_pass_rate < 0.9:
                recommendations.append("🚨 CRITICAL: Emergency detection pass rate below 90% - review emergency training examples")
        
        # Performance category analysis
        category_scores = {}
        for eval_result in evaluations:
            if eval_result.category.value not in category_scores:
                category_scores[eval_result.category.value] = 0
            category_scores[eval_result.category.value] += 1
        
        failed_count = category_scores.get("failed", 0)
        if failed_count > len(evaluations) * 0.2:
            recommendations.append("❌ High failure rate - review DSPy signatures and examples")
        
        # Common issues analysis
        all_critical_issues = []
        for eval_result in evaluations:
            all_critical_issues.extend(eval_result.critical_issues)
        
        if len(all_critical_issues) > len(evaluations) * 0.3:
            recommendations.append("⚠️ Many critical issues detected - comprehensive system review needed")
        
        # Score-based recommendations
        avg_score = sum(e.total_score for e in evaluations) / len(evaluations)
        if avg_score < 70:
            recommendations.append("📊 Average score below 70 - focus on DSPy prompt optimization")
        
        if not recommendations:
            recommendations.append("✅ System performing within acceptable parameters - continue monitoring")
        
        return recommendations
