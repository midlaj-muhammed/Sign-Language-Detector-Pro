"""
Fallback Sign Language Classifier

This module provides basic sign language classification without requiring OpenAI API.
It uses rule-based pattern matching to identify common ASL letters and gestures.
"""

from typing import Dict, Any, Optional
import re


class FallbackSignLanguageClassifier:
    """
    Fallback classifier for basic ASL recognition using pattern matching.
    """
    
    def __init__(self):
        """Initialize the fallback classifier."""
        self.debug = True
        print("Fallback classifier initialized (no API required)")
    
    def classify_gesture(self, gesture_description: str, 
                        sign_language: str = "ASL", 
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify gesture using rule-based pattern matching.
        
        Args:
            gesture_description: Description of the hand gesture
            sign_language: Sign language type (default: ASL)
            context: Additional context (optional)
            
        Returns:
            Classification result dictionary
        """
        if self.debug:
            print(f"\n=== Fallback Classification Debug ===")
            print(f"Input: {gesture_description}")
        
        try:
            # Analyze the gesture description
            result = self._analyze_gesture_patterns(gesture_description)
            result['success'] = True
            result['method'] = 'fallback_pattern_matching'
            
            if self.debug:
                print(f"Result: {result}")
                print("=== End Fallback Debug ===\n")
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Fallback classification error: {str(e)}")
                print("=== End Fallback Debug ===\n")
            
            return {
                'success': False,
                'error': str(e),
                'letter': None,
                'word': None,
                'confidence': 0.0,
                'description': 'Fallback classification failed',
                'method': 'fallback_pattern_matching'
            }
    
    def _analyze_gesture_patterns(self, description: str) -> Dict[str, Any]:
        """
        Analyze gesture description using pattern matching rules.
        
        Args:
            description: Gesture description string
            
        Returns:
            Classification result
        """
        desc_lower = description.lower()
        
        # Extract key information
        extended_fingers = self._extract_extended_fingers(desc_lower)
        closed_fingers = self._extract_closed_fingers(desc_lower)
        patterns = self._extract_patterns(desc_lower)
        
        # Rule-based classification
        letter, word, confidence, explanation = self._apply_classification_rules(
            extended_fingers, closed_fingers, patterns, desc_lower
        )
        
        return {
            'letter': letter,
            'word': word,
            'confidence': confidence,
            'description': explanation,
            'extended_fingers': extended_fingers,
            'closed_fingers': closed_fingers,
            'patterns': patterns
        }
    
    def _extract_extended_fingers(self, description: str) -> list:
        """Extract extended fingers from description."""
        extended = []
        if 'extended fingers:' in description:
            # Find the extended fingers section
            start = description.find('extended fingers:') + len('extended fingers:')
            end = description.find(';', start)
            if end == -1:
                end = len(description)
            
            fingers_text = description[start:end].strip()
            
            # Extract individual fingers
            if 'thumb' in fingers_text:
                extended.append('thumb')
            if 'index' in fingers_text:
                extended.append('index')
            if 'middle' in fingers_text:
                extended.append('middle')
            if 'ring' in fingers_text:
                extended.append('ring')
            if 'pinky' in fingers_text:
                extended.append('pinky')
        
        return extended
    
    def _extract_closed_fingers(self, description: str) -> list:
        """Extract closed fingers from description."""
        closed = []
        if 'closed fingers:' in description:
            # Find the closed fingers section
            start = description.find('closed fingers:') + len('closed fingers:')
            end = description.find(';', start)
            if end == -1:
                end = len(description)
            
            fingers_text = description[start:end].strip()
            
            # Extract individual fingers
            if 'thumb' in fingers_text:
                closed.append('thumb')
            if 'index' in fingers_text:
                closed.append('index')
            if 'middle' in fingers_text:
                closed.append('middle')
            if 'ring' in fingers_text:
                closed.append('ring')
            if 'pinky' in fingers_text:
                closed.append('pinky')
        
        return closed
    
    def _extract_patterns(self, description: str) -> list:
        """Extract gesture patterns from description."""
        patterns = []
        
        if 'closed fist' in description:
            patterns.append('closed_fist')
        if 'open hand' in description:
            patterns.append('open_hand')
        if 'pointing gesture' in description:
            patterns.append('pointing')
        if 'pinch gesture' in description:
            patterns.append('pinch')
        
        return patterns
    
    def _apply_classification_rules(self, extended: list, closed: list,
                                  patterns: list, description: str) -> tuple:
        """
        Apply enhanced ASL-specific classification logic.

        Returns:
            (letter, word, confidence, explanation)
        """

        # PRECISE ASL RULES based on exact finger positions

        # Rule 1: Single finger extended
        if len(extended) == 1:
            if 'index' in extended:
                return '1', None, 0.9, "Index finger only = Number 1"
            elif 'pinky' in extended:
                return None, 'I', 0.9, "Pinky finger only = Pronoun I"
            elif 'thumb' in extended:
                return None, 'GOOD', 0.8, "Thumb up = GOOD"
            elif 'middle' in extended:
                return None, 'BAD', 0.6, "Middle finger = BAD (rude gesture)"

        # Rule 2: Two fingers extended
        if len(extended) == 2:
            if 'index' in extended and 'middle' in extended:
                return '2', None, 0.9, "Index and middle = Number 2"
            elif 'index' in extended and 'thumb' in extended:
                return 'L', None, 0.8, "Index and thumb = Letter L"
            elif 'index' in extended and 'pinky' in extended:
                return None, 'I LOVE YOU', 0.9, "Index and pinky = I LOVE YOU sign"
            elif 'thumb' in extended and 'pinky' in extended:
                return None, 'CALL', 0.7, "Thumb and pinky = CALL/PHONE"

        # Rule 3: Three fingers extended
        if len(extended) == 3:
            if 'index' in extended and 'middle' in extended and 'ring' in extended:
                return '3', None, 0.9, "Three middle fingers = Number 3"
            elif 'thumb' in extended and 'index' in extended and 'pinky' in extended:
                return None, 'I LOVE YOU', 0.9, "Thumb, index, pinky = I LOVE YOU"

        # Rule 4: Four fingers extended (thumb closed)
        if len(extended) == 4 and 'thumb' in closed:
            return '4', None, 0.9, "Four fingers, thumb closed = Number 4"

        # Rule 5: All five fingers extended
        if len(extended) == 5:
            return '5', None, 0.9, "All fingers extended = Number 5"

        # Rule 6: Closed fist (no fingers extended)
        if len(extended) == 0 or 'closed_fist' in patterns:
            return 'A', None, 0.8, "Closed fist = Letter A"

        # Rule 7: Four fingers extended (index, middle, ring, pinky) - thumb closed
        if (len(extended) == 4 and 'index' in extended and 'middle' in extended
            and 'ring' in extended and 'pinky' in extended and 'thumb' in closed):
            return None, 'HELLO', 0.8, "Four fingers extended = HELLO"

        # Rule 8: Pinch gesture pattern
        if 'pinch' in patterns:
            return 'F', None, 0.7, "Pinch gesture = Letter F"

        # Rule 9: Pointing gesture pattern
        if 'pointing' in patterns:
            if 'index' in extended and len(extended) == 1:
                return '1', None, 0.8, "Pointing with index = Number 1"
            else:
                return None, 'YOU', 0.6, "Pointing gesture = YOU"

        # Rule 10: Open hand pattern
        if 'open_hand' in patterns:
            if len(extended) == 5:
                return '5', None, 0.8, "Open hand = Number 5"
            else:
                return None, 'HELLO', 0.7, "Open hand = HELLO"

        # Default fallback based on finger count with lower confidence
        finger_count = len(extended)
        if finger_count == 0:
            return 'A', None, 0.4, f"No extended fingers, default to A"
        elif finger_count == 1:
            return '1', None, 0.4, f"One finger extended, default to 1"
        elif finger_count == 2:
            return '2', None, 0.4, f"Two fingers extended, default to 2"
        elif finger_count == 3:
            return '3', None, 0.4, f"Three fingers extended, default to 3"
        elif finger_count == 4:
            return '4', None, 0.4, f"Four fingers extended, default to 4"
        elif finger_count == 5:
            return '5', None, 0.4, f"Five fingers extended, default to 5"
        else:
            return None, None, 0.1, "Unable to classify gesture"
    
    def classify_sequence(self, gesture_descriptions: list, 
                         sign_language: str = "ASL") -> Dict[str, Any]:
        """
        Classify a sequence of gestures (fallback implementation).
        
        Args:
            gesture_descriptions: List of gesture descriptions
            sign_language: Sign language type
            
        Returns:
            Sequence classification result
        """
        # Simple implementation: classify each gesture and combine
        letters = []
        words = []
        
        for desc in gesture_descriptions:
            result = self.classify_gesture(desc, sign_language)
            if result.get('success'):
                if result.get('letter'):
                    letters.append(result['letter'])
                if result.get('word'):
                    words.append(result['word'])
        
        # Try to form words from letters
        if letters and not words:
            letter_sequence = ''.join(letters)
            # Check for common words
            common_words = {
                'HI': 'HI',
                'NO': 'NO',
                'OK': 'OK',
                'YES': 'YES'
            }
            
            if letter_sequence in common_words:
                words.append(common_words[letter_sequence])
        
        return {
            'success': True,
            'word': words[0] if words else None,
            'sentence': ' '.join(words) if len(words) > 1 else None,
            'confidence': 0.6,
            'individual_letters': letters,
            'method': 'fallback_sequence_matching'
        }
