"""
Fairdoc AI Thinking Process Processor
Handles extraction and analysis of AI reasoning chains
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)

def utcnow():
    """Return timezone-aware UTC now"""
    return datetime.now(timezone.utc)


class ThinkingProcessor:
    """
    Processes and analyzes AI thinking processes for safety and observability
    """
    
    def __init__(self):
        self.safety_keywords = {
            'emergency': ['emergency', 'urgent', 'critical', 'severe', 'life-threatening'],
            'caution': ['careful', 'concerning', 'worry', 'serious', 'monitor'],
            'uncertainty': ['not sure', 'unclear', 'uncertain', 'might be', 'could be'],
            'disclaimers': ['cannot diagnose', 'not a doctor', 'seek medical attention']
        }
    
    def extract_thinking_process(self, raw_response: str) -> Tuple[str, Optional[Dict]]:
        """
        Extract thinking process from response and return clean response
        
        Args:
            raw_response: Raw AI response with potential <think> tags
            
        Returns:
            Tuple of (clean_response, thinking_data)
        """
        try:
            # Extract thinking content using regex
            think_pattern = r'<think>(.*?)</think>'
            thinking_matches = re.findall(think_pattern, raw_response, re.DOTALL)
            
            if not thinking_matches:
                # No thinking process found, return original response
                return raw_response.strip(), None
            
            # Remove thinking tags from response
            clean_response = re.sub(think_pattern, '', raw_response, flags=re.DOTALL).strip()
            
            # Process thinking content
            thinking_content = thinking_matches[0].strip()
            thinking_data = self._analyze_thinking_content(thinking_content)
            
            logger.info("🧠 Thinking process extracted", 
                       thinking_length=len(thinking_content),
                       safety_flags=len(thinking_data.get('safety_flags', [])))
            
            return clean_response, thinking_data
            
        except Exception as e:
            logger.error("Error extracting thinking process", error=str(e))
            return raw_response.strip(), None
    
    def _analyze_thinking_content(self, thinking_content: str) -> Dict[str, Any]:
        """Analyze thinking content for safety and quality metrics"""
        
        analysis = {
            'content': thinking_content,
            'word_count': len(thinking_content.split()),
            'safety_flags': [],
            'reasoning_steps': self._extract_reasoning_steps(thinking_content),
            'confidence_indicators': self._extract_confidence_indicators(thinking_content),
            'medical_considerations': self._extract_medical_considerations(thinking_content),
            'timestamp': utcnow().isoformat()
        }
        
        # Safety flag analysis
        thinking_lower = thinking_content.lower()
        for category, keywords in self.safety_keywords.items():
            for keyword in keywords:
                if keyword in thinking_lower:
                    analysis['safety_flags'].append({
                        'category': category,
                        'keyword': keyword,
                        'severity': self._assess_severity(category)
                    })
        
        return analysis
    
    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """Extract key reasoning steps from thinking process"""
        
        # Look for numbered steps, bullet points, or logical flow indicators
        step_patterns = [
            r'(\d+[\.\)]\s*[^.!?]*[.!?])',  # Numbered steps
            r'(First[^.!?]*[.!?])',          # First, Second, etc.
            r'(Next[^.!?]*[.!?])',
            r'(Then[^.!?]*[.!?])',
            r'(Finally[^.!?]*[.!?])'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])
        
        return steps[:10]  # Limit to first 10 steps
    
    def _extract_confidence_indicators(self, content: str) -> Dict[str, Any]:
        """Extract confidence and uncertainty indicators"""
        
        confidence_patterns = {
            'high_confidence': ['definitely', 'clearly', 'obviously', 'certainly'],
            'medium_confidence': ['likely', 'probably', 'seems like', 'appears to'],
            'low_confidence': ['might', 'possibly', 'uncertain', 'not sure', 'unclear']
        }
        
        indicators = {}
        content_lower = content.lower()
        
        for level, words in confidence_patterns.items():
            count = sum(1 for word in words if word in content_lower)
            if count > 0:
                indicators[level] = count
        
        return indicators
    
    def _extract_medical_considerations(self, content: str) -> List[str]:
        """Extract medical terms and considerations mentioned"""
        
        medical_patterns = [
            r'\b(cardiac|heart|chest pain|angina|symptoms|diagnosis|treatment)\b',
            r'\b(patient|medical|clinical|healthcare|doctor|physician)\b',
            r'\b(risk|safety|urgent|emergency|serious|concerning)\b'
        ]
        
        considerations = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            considerations.extend(matches)
        
        return list(set(considerations))  # Remove duplicates
    
    def _assess_severity(self, category: str) -> str:
        """Assess severity level of safety flags"""
        
        severity_map = {
            'emergency': 'high',
            'caution': 'medium', 
            'uncertainty': 'low',
            'disclaimers': 'informational'
        }
        
        return severity_map.get(category, 'low')
    
    def generate_safety_summary(self, thinking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate safety summary for monitoring dashboard"""
        
        if not thinking_data:
            return {'status': 'no_thinking_data'}
        
        safety_flags = thinking_data.get('safety_flags', [])
        high_severity_flags = [f for f in safety_flags if f.get('severity') == 'high']
        
        summary = {
            'overall_safety_level': 'safe',
            'total_flags': len(safety_flags),
            'high_severity_count': len(high_severity_flags),
            'reasoning_quality': self._assess_reasoning_quality(thinking_data),
            'requires_review': len(high_severity_flags) > 0,
            'generated_at': utcnow().isoformat()
        }
        
        # Determine overall safety level
        if len(high_severity_flags) > 2:
            summary['overall_safety_level'] = 'needs_review'
        elif len(high_severity_flags) > 0:
            summary['overall_safety_level'] = 'caution'
        
        return summary
    
    def _assess_reasoning_quality(self, thinking_data: Dict[str, Any]) -> str:
        """Assess quality of reasoning process"""
        
        word_count = thinking_data.get('word_count', 0)
        reasoning_steps = len(thinking_data.get('reasoning_steps', []))
        medical_considerations = len(thinking_data.get('medical_considerations', []))
        
        # Simple quality scoring
        quality_score = 0
        if word_count > 50:
            quality_score += 1
        if reasoning_steps > 2:
            quality_score += 1  
        if medical_considerations > 3:
            quality_score += 1
        
        if quality_score >= 3:
            return 'high'
        elif quality_score >= 2:
            return 'medium'
        else:
            return 'basic'
