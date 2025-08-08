# Enhanced NICE Lookup Service - Production Ready Implementation
# Fixes critical SEV_BURN misclassification bug while maintaining 100% backward compatibility
# File: src/app2/services/context/nice_lookup.py

"""
Production-Ready Enhanced NICE Lookup Service
Maintains 100% backward compatibility while fixing critical bugs
"""

from __future__ import annotations
import re
import structlog
from typing import Dict, List, Optional, Tuple, Set, Any
from functools import cached_property
from dataclasses import dataclass
from enum import Enum

# Import existing dependencies (maintaining compatibility)
from src.app2.models.database.nice_protocols import NICE_SEED_DATA

logger = structlog.get_logger(__name__)

class MedicalContextAnalyzerLite:
    """
    Lightweight medical context analyzer for production deployment
    Optimized for speed and reliability without external dependencies
    """
    
    def __init__(self):
        # Anatomical region mapping
        self.anatomical_regions = {
            'head': ['head', 'skull', 'face', 'brain', 'headache', 'migraine'],
            'chest': ['chest', 'thorax', 'heart', 'cardiac', 'breast', 'sternum'],
            'abdomen': ['stomach', 'belly', 'abdomen', 'gut', 'liver', 'epigastric', 'gastric'],
            'limbs': ['arm', 'leg', 'hand', 'foot', 'joint', 'knee', 'elbow', 'wrist'],
            'back': ['back', 'spine', 'neck', 'shoulder', 'lumbar', 'cervical'],
            'skin': ['skin', 'rash', 'burn', 'wound', 'cut', 'blister', 'burn']
        }
        
        # Severity indicators
        self.severity_keywords = {
            'severe': ['severe', 'unbearable', 'excruciating', 'intense', 'crushing', 'worst'],
            'moderate': ['moderate', 'significant', 'troubling', 'concerning', 'bad'],
            'mild': ['mild', 'slight', 'minor', 'dull', 'aching', 'little']
        }
        
        # Red flag patterns by system
        self.red_flag_patterns = {
            'cardiovascular': ['crushing', 'radiating', 'sweating', 'nausea', 'left arm'],
            'neurological': ['thunderclap', 'worst ever', 'neck stiffness', 'sudden'],
            'respiratory': ['cannot breathe', 'blue', 'choking', 'gasping'],
            'trauma': ['accident', 'fall', 'hit', 'injured', 'bleeding', 'burn']
        }
        
        # Burn-specific indicators (prevents false positives)
        self.burn_indicators = ['burn', 'burned', 'burnt', 'scald', 'fire', 'flame', 'hot oil', 'steam', 'blister']
    
    def extract_context(self, text: str) -> Dict[str, Any]:
        """Extract medical context from patient text"""
        text_lower = text.lower()
        
        return {
            'anatomical_location': self._identify_anatomical_region(text_lower),
            'severity_level': self._identify_severity(text_lower),
            'red_flags': self._identify_red_flags(text_lower),
            'burn_indicators': self._has_burn_indicators(text_lower),
            'primary_symptom': self._extract_primary_symptom(text_lower)
        }
    
    def _identify_anatomical_region(self, text: str) -> str:
        """Identify primary anatomical region"""
        region_scores = {}
        for region, keywords in self.anatomical_regions.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                region_scores[region] = score
        
        return max(region_scores, key=region_scores.get) if region_scores else "general"
    
    def _identify_severity(self, text: str) -> str:
        """Identify severity level"""
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text for keyword in keywords):
                return severity
        return "unspecified"
    
    def _identify_red_flags(self, text: str) -> List[str]:
        """Identify medical red flags with associated system"""
        red_flags = []
        for system, patterns in self.red_flag_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    red_flags.append(f"{system}:{pattern}")
        return red_flags

    
    def _has_burn_indicators(self, text: str) -> bool:
        """Check for specific burn indicators"""
        return any(indicator in text for indicator in self.burn_indicators)
    
    def _extract_primary_symptom(self, text: str) -> str:
        """Extract primary symptom type"""
        symptom_patterns = {
            'pain': ['pain', 'ache', 'hurt', 'sore', 'throb'],
            'breathing': ['breath', 'breathing', 'dyspnea', 'shortness'],
            'headache': ['headache', 'head pain', 'migraine'],
            'nausea': ['nausea', 'sick', 'vomit', 'queasy'],
            'fever': ['fever', 'hot', 'temperature'],
            'dizziness': ['dizzy', 'lightheaded', 'vertigo']
        }
        
        for symptom, patterns in symptom_patterns.items():
            if any(pattern in text for pattern in patterns):
                return symptom
        
        return "unspecified"

class EnhancedNICELookupService:
    """
    Enhanced NICE Protocol Lookup Service
    Fixes SEV_BURN misclassification while maintaining 100% API compatibility
    """
    
    def __init__(self, seed_data: Optional[List[Dict]] = None):
        self._protocols = seed_data or NICE_SEED_DATA
        self.context_analyzer = MedicalContextAnalyzerLite()
        
        # Build optimized indices for fast lookup
        self._build_indices()
        
        logger.info("Enhanced NICE Lookup Service initialized", 
                   protocol_count=len(self._protocols))
    
    def _build_indices(self):
        """Build optimized lookup indices"""
        # Category index for fast anatomical matching
        self._category_index = {}
        for protocol in self._protocols:
            category = protocol.get('category', 'General')
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(protocol)
        
        # Symptom index for fast symptom matching
        self._symptom_index = {}
        for protocol in self._protocols:
            for symptom in protocol.get('primary_symptoms', []):
                if symptom not in self._symptom_index:
                    self._symptom_index[symptom] = []
                self._symptom_index[symptom].append(protocol)
        
        logger.debug("NICE indices built", 
                    categories=len(self._category_index),
                    symptoms=len(self._symptom_index))
    
    def find_relevant_protocols(self, symptom_text: str) -> Dict[str, str]:
        """
        Enhanced protocol matching with context awareness
        
        MAINTAINS EXACT API COMPATIBILITY with original implementation
        Returns: Dict with keys 'protocol_code' and 'protocol_text'
        """
        if not symptom_text or not isinstance(symptom_text, str):
            return self._get_fallback_response()
        
        # Extract medical context
        context = self.context_analyzer.extract_context(symptom_text)
        
        # Multi-strategy protocol matching
        candidates = self._get_protocol_candidates(context, symptom_text)
        
        if not candidates:
            logger.warning("No protocol candidates found", symptom_text=symptom_text[:50])
            return self._get_fallback_response()
        
        # Select best match with confidence scoring
        best_match = self._select_best_match(candidates, context)
        
        # Format response (maintains exact compatibility)
        response = {
            "protocol_code": best_match['protocol_code'],
            "protocol_text": self._format_protocol(best_match)
        }
        
        logger.info("Protocol matched", 
                   protocol_code=best_match['protocol_code'],
                   confidence=best_match.get('confidence', 'N/A'))
        
        return response
    
    def _get_protocol_candidates(self, context: Dict[str, Any], 
                               original_text: str) -> List[Dict[str, Any]]:
        """Get ranked list of protocol candidates"""
        candidates = []
        
        # Strategy 1: Anatomical region matching
        candidates.extend(self._anatomical_matching(context))
        
        # Strategy 2: Symptom-based matching  
        candidates.extend(self._symptom_matching(context, original_text))
        
        # Strategy 3: Red flag emergency matching
        candidates.extend(self._red_flag_matching(context))
        
        # Strategy 4: Burn-specific matching (prevents false positives)
        if context.get('burn_indicators', False):
            candidates.extend(self._burn_specific_matching(context))
        
        # Remove duplicates and add confidence scores
        return self._deduplicate_and_score(candidates, context, original_text)
    
    def _anatomical_matching(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match protocols based on anatomical location"""
        anatomical_location = context.get('anatomical_location', 'general')
        
        # Map anatomical locations to protocol categories
        location_to_category = {
            'head': ['Neurological'],
            'chest': ['Cardiovascular', 'Respiratory'],
            'abdomen': ['Gastrointestinal'],
            'limbs': ['Trauma'],
            'back': ['Neurological'],
            'skin': ['Trauma']
        }
        
        relevant_categories = location_to_category.get(anatomical_location, [])
        candidates = []
        
        for category in relevant_categories:
            if category in self._category_index:
                candidates.extend(self._category_index[category])
        
        return candidates
    
    def _symptom_matching(self, context: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        """Match protocols based on primary symptoms"""
        primary_symptom = context.get('primary_symptom', 'unspecified')
        candidates = []
        
        # Direct symptom matching
        if primary_symptom in self._symptom_index:
            candidates.extend(self._symptom_index[primary_symptom])
        
        # Fuzzy symptom matching for common patterns
        symptom_mappings = {
            'pain': ['chest_pain', 'headache', 'back_pain', 'epigastric_pain'],
            'breathing': ['dyspnea', 'wheeze', 'cough'],
            'headache': ['headache', 'photophobia']
        }
        
        if primary_symptom in symptom_mappings:
            for mapped_symptom in symptom_mappings[primary_symptom]:
                if mapped_symptom in self._symptom_index:
                    candidates.extend(self._symptom_index[mapped_symptom])
        
        return candidates
    
    def _red_flag_matching(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match protocols based on red flag symptoms"""
        red_flags = context.get('red_flags', [])
        if not red_flags:
            return []
        
        candidates = []
        
        # High priority red flag matching
        red_flag_protocols = {
            'crushing': ['NG136_MI', 'CG95_CHEST_PAIN'],
            'thunderclap': ['SAH', 'NG127_HEADACHE'],
            'worst ever': ['SAH', 'NG127_HEADACHE'],
            'neck stiffness': ['MENINGITIS_ACUTE', 'NG127_HEADACHE'],
            'cannot breathe': ['NG80_ASTHMA_EXAC', 'ANAPHYLAXIS_AIRWAY']
        }
        
        for flag in red_flags:
            if flag in red_flag_protocols:
                for protocol_code in red_flag_protocols[flag]:
                    protocol = self._find_protocol_by_code(protocol_code)
                    if protocol:
                        # Mark as high confidence due to red flag
                        protocol_copy = protocol.copy()
                        protocol_copy['red_flag_boost'] = True
                        candidates.append(protocol_copy)
        
        return candidates
    
    def _burn_specific_matching(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match burn protocols ONLY when burn indicators are present"""
        # This prevents the SEV_BURN misclassification bug
        if not context.get('burn_indicators', False):
            return []
        
        # Only return burn protocol if we have actual burn indicators
        burn_protocol = self._find_protocol_by_code('SEV_BURN')
        if burn_protocol:
            burn_copy = burn_protocol.copy()
            burn_copy['burn_specific_match'] = True
            return [burn_copy]
        
        return []
    
    def _deduplicate_and_score(self, candidates: List[Dict[str, Any]], 
                             context: Dict[str, Any], 
                             original_text: str) -> List[Dict[str, Any]]:
        """Remove duplicates and add confidence scores"""
        # Deduplicate by protocol_code
        unique_candidates = {}
        for candidate in candidates:
            code = candidate['protocol_code']
            if code not in unique_candidates:
                unique_candidates[code] = candidate
        
        # Add confidence scores
        scored_candidates = []
        for candidate in unique_candidates.values():
            confidence = self._calculate_confidence(candidate, context, original_text)
            candidate['confidence'] = confidence
            scored_candidates.append(candidate)
        
        # Sort by confidence descending
        return sorted(scored_candidates, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_confidence(self, protocol: Dict[str, Any], 
                            context: Dict[str, Any], 
                            original_text: str) -> float:
        """Calculate confidence score for protocol match"""
        confidence = 0.5  # Base confidence
        
        # Boost for red flag matches
        if protocol.get('red_flag_boost', False):
            confidence += 0.3
        
        # Boost for burn-specific matches
        if protocol.get('burn_specific_match', False):
            confidence += 0.4
        
        # Boost for anatomical alignment
        anatomical_location = context.get('anatomical_location', 'general')
        protocol_category = protocol.get('category', 'General')
        
        alignment_map = {
            ('head', 'Neurological'): 0.2,
            ('chest', 'Cardiovascular'): 0.2,
            ('chest', 'Respiratory'): 0.2,
            ('abdomen', 'Gastrointestinal'): 0.2,
            ('skin', 'Trauma'): 0.2
        }
        
        if (anatomical_location, protocol_category) in alignment_map:
            confidence += alignment_map[(anatomical_location, protocol_category)]
        
        # Penalty for misaligned matches (prevents SEV_BURN for non-burns)
        if (protocol['protocol_code'] == 'SEV_BURN' and 
            not context.get('burn_indicators', False)):
            confidence *= 0.1  # Heavy penalty for burn misclassification
        
        # Boost for severity alignment
        severity = context.get('severity_level', 'unspecified')
        emergency_criteria = protocol.get('emergency_criteria', '').lower()
        
        if severity == 'severe' and any(word in emergency_criteria 
                                       for word in ['emergency', 'severe', 'shock']):
            confidence += 0.1
        
        # FIX: Round to 2 decimal places to avoid floating point precision issues
        final_confidence = min(confidence, 1.0)
        return round(final_confidence, 2)

    
    def _select_best_match(self, candidates: List[Dict[str, Any]], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best protocol match from candidates"""
        if not candidates:
            return self._get_fallback_protocol_dict()
        
        # Return highest confidence match
        best_match = candidates[0]  # Already sorted by confidence
        
        # Additional validation for SEV_BURN
        if (best_match['protocol_code'] == 'SEV_BURN' and 
            not context.get('burn_indicators', False)):
            # Find next best non-burn match
            for candidate in candidates[1:]:
                if candidate['protocol_code'] != 'SEV_BURN':
                    return candidate
            # If no other match, use fallback
            return self._get_fallback_protocol_dict()
        
        return best_match
    
    def _find_protocol_by_code(self, protocol_code: str) -> Optional[Dict[str, Any]]:
        """Find protocol by code"""
        for protocol in self._protocols:
            if protocol['protocol_code'] == protocol_code:
                return protocol
        return None
    
    def _get_fallback_response(self) -> Dict[str, str]:
        """Fallback response maintaining API compatibility"""
        return {
            "protocol_code": "GENERAL_TRIAGE",
            "protocol_text": "General medical triage protocol for symptom assessment"
        }
    
    def _get_fallback_protocol_dict(self) -> Dict[str, Any]:
        """Fallback protocol dictionary"""
        return {
            'protocol_code': 'GENERAL_TRIAGE',
            'condition_name': 'General Medical Assessment',
            'primary_symptoms': ['general'],
            'initial_questions': ['What are your main symptoms?'],
            'follow_up_questions': ['How long have you had these symptoms?'],
            'emergency_criteria': 'Severe symptoms requiring immediate attention',
            'routine_criteria': 'Symptoms suitable for routine medical care',
            'self_care_criteria': 'Minor symptoms manageable at home',
            'confidence': 0.5
        }
    
    def _format_protocol(self, protocol: Dict[str, Any]) -> str:
        """Format protocol for output (maintains compatibility)"""
        questions = (protocol.get('initial_questions', []) + 
                    protocol.get('follow_up_questions', []))
        
        return (
            f"Condition: {protocol.get('condition_name', 'Unknown')}\n"
            f"Emergency criteria: {protocol.get('emergency_criteria', 'N/A')}\n"
            f"Routine criteria: {protocol.get('routine_criteria', 'N/A')}\n"
            f"Self-care criteria: {protocol.get('self_care_criteria', 'N/A')}\n"
            "Questions:\n • " + "\n • ".join(questions[:5])
        )

# BACKWARD COMPATIBILITY WRAPPER
class NICELookupService(EnhancedNICELookupService):
    """
    Legacy compatibility wrapper
    
    This class maintains 100% backward compatibility with existing code
    while providing all the enhanced functionality internally
    """
    
    def __init__(self, seed_data: Optional[List[Dict]] = None):
        """Initialize with same signature as original"""
        super().__init__(seed_data)
        logger.info("Legacy NICE Lookup Service initialized with enhanced backend")
    
    # All existing methods work unchanged - enhanced implementation is transparent
    
    @cached_property
    def _index(self) -> Dict[str, Dict]:
        """Legacy index property for compatibility"""
        # Build legacy-style index for any code that might access it directly
        index = {}
        for protocol in self._protocols:
            for symptom in protocol.get('primary_symptoms', []):
                index[symptom.lower()] = protocol
        return index
    
    @staticmethod
    def _clean_text(text: str) -> List[str]:
        """Legacy text cleaning method for compatibility"""
        return re.findall(r"[a-zA-Z_]+", text.lower())
    
    @staticmethod
    def _format_protocol(proto: Dict) -> str:
        """Legacy protocol formatting for compatibility"""
        questions = proto.get("initial_questions", []) + proto.get("follow_up_questions", [])
        return (
            f"Condition: {proto.get('condition_name', 'Unknown')}\n"
            f"Emergency criteria: {proto.get('emergency_criteria', 'N/A')}\n"
            f"Routine criteria: {proto.get('routine_criteria', 'N/A')}\n"
            f"Self-care criteria: {proto.get('self_care_criteria', 'N/A')}\n"
            "Questions:\n • " + "\n • ".join(questions[:5])
        )

# Export the enhanced service as the default implementation
__all__ = ['NICELookupService', 'EnhancedNICELookupService']
