"""
OpenAI API Integration for Sign Language Classification
"""

from openai import OpenAI
import os
from typing import List, Dict, Any, Optional
import json
import time
from dotenv import load_dotenv
from .fallback_classifier import FallbackSignLanguageClassifier

# Load environment variables
load_dotenv()


class SignLanguageClassifier:
    """
    A class for classifying sign language gestures using OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the SignLanguageClassifier.

        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI model to use for classification
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Initialize OpenAI client with new format
        self.client = OpenAI(api_key=self.api_key)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

        # Debug mode
        self.debug = True

        # Initialize fallback classifier
        self.fallback_classifier = FallbackSignLanguageClassifier()

        print(f"OpenAI classifier initialized with fallback support")
        
    def _rate_limit(self):
        """Implement simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def classify_gesture(self, gesture_description: str, 
                        sign_language: str = "ASL",
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a gesture using OpenAI API.
        
        Args:
            gesture_description: Textual description of the gesture
            sign_language: Type of sign language (ASL, ISL, etc.)
            context: Additional context for classification
            
        Returns:
            Dictionary containing classification results
        """
        self._rate_limit()

        # Create the prompt
        prompt = self._create_classification_prompt(gesture_description, sign_language, context)

        if self.debug:
            print(f"\n=== OpenAI Classification Debug ===")
            print(f"Input gesture description: {gesture_description}")
            print(f"Prompt sent to OpenAI: {prompt}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(sign_language)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                top_p=0.9
            )

            response_content = response.choices[0].message.content

            if self.debug:
                print(f"OpenAI response: {response_content}")

            result = self._parse_response(response_content)
            result['raw_response'] = response_content
            result['success'] = True

            if self.debug:
                print(f"Parsed result: {result}")
                print("=== End Debug ===\n")

            return result

        except Exception as e:
            error_msg = str(e)
            if self.debug:
                print(f"OpenAI API Error: {error_msg}")
                print("Falling back to pattern-based classification...")

            # Use fallback classifier when OpenAI API fails
            try:
                fallback_result = self.fallback_classifier.classify_gesture(
                    gesture_description, sign_language, context
                )
                fallback_result['fallback_used'] = True
                fallback_result['openai_error'] = error_msg

                if self.debug:
                    print(f"Fallback result: {fallback_result}")
                    print("=== End Debug ===\n")

                return fallback_result

            except Exception as fallback_error:
                if self.debug:
                    print(f"Fallback also failed: {str(fallback_error)}")
                    print("=== End Debug ===\n")

                return {
                    'success': False,
                    'error': error_msg,
                    'fallback_error': str(fallback_error),
                    'letter': None,
                    'word': None,
                    'confidence': 0.0,
                    'description': None
                }
    
    def classify_sequence(self, gesture_descriptions: List[str],
                         sign_language: str = "ASL") -> Dict[str, Any]:
        """
        Classify a sequence of gestures to form words or sentences.
        
        Args:
            gesture_descriptions: List of gesture descriptions
            sign_language: Type of sign language
            
        Returns:
            Dictionary containing sequence classification results
        """
        self._rate_limit()
        
        # Create sequence prompt
        prompt = self._create_sequence_prompt(gesture_descriptions, sign_language)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_sequence_system_prompt(sign_language)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3,
                top_p=0.9
            )
            
            result = self._parse_sequence_response(response.choices[0].message.content)
            result['raw_response'] = response.choices[0].message.content
            result['success'] = True
            
            return result
            
        except Exception as e:
            # Use fallback for sequence classification too
            try:
                fallback_result = self.fallback_classifier.classify_sequence(
                    gesture_descriptions, sign_language
                )
                fallback_result['fallback_used'] = True
                fallback_result['openai_error'] = str(e)
                return fallback_result

            except Exception as fallback_error:
                return {
                    'success': False,
                    'error': str(e),
                    'fallback_error': str(fallback_error),
                    'word': None,
                    'sentence': None,
                    'confidence': 0.0
                }
    
    def _get_system_prompt(self, sign_language: str) -> str:
        """Get system prompt for gesture classification."""
        return f"""You are an expert in {sign_language} (American Sign Language) recognition.
        Your task is to provide ONE CLEAR PREDICTION for each hand gesture.

        PRIORITY ORDER:
        1. If it's a complete word sign (like "HELLO", "HUNGRY", "THANK YOU"), identify the WORD
        2. If it's a letter/number sign, identify the LETTER or NUMBER
        3. If uncertain, provide your best single guess

        Respond in JSON format:
        {{
            "letter": "A" or null,
            "word": "HUNGRY" or null,
            "confidence": 0.85,
            "description": "Brief explanation"
        }}

        IMPORTANT RULES:
        - Provide either a letter OR a word, not both
        - Words take priority over letters
        - Be decisive - give your best single prediction
        - Common words: HELLO, HUNGRY, THANK YOU, PLEASE, SORRY, YES, NO, I, YOU, LOVE, etc.
        - Letters: A-Z, Numbers: 0-9
        - Confidence should reflect your certainty (0.1 = very uncertain, 0.9 = very certain)

        Focus on the most likely single interpretation of the gesture."""
    
    def _get_sequence_system_prompt(self, sign_language: str) -> str:
        """Get system prompt for sequence classification."""
        return f"""You are an expert in {sign_language} recognition specializing in interpreting sequences of gestures.
        Your task is to analyze a sequence of hand gestures and determine if they form a word or sentence.
        
        Respond in JSON format:
        {{
            "word": "HELLO" or null,
            "sentence": "HELLO WORLD" or null,
            "confidence": 0.85,
            "individual_letters": ["H", "E", "L", "L", "O"]
        }}
        
        Consider:
        - Sequential letter spelling
        - Common {sign_language} words and phrases
        - Context and flow between gestures"""
    
    def _create_classification_prompt(self, gesture_description: str,
                                    sign_language: str, context: Optional[str]) -> str:
        """Create enhanced prompt for single gesture classification."""
        prompt = f"""You are an expert ASL (American Sign Language) interpreter. Analyze this hand gesture and provide ONE CLEAR PREDICTION.

GESTURE DATA:
{gesture_description}

TASK: Identify what this gesture represents. Respond with EXACTLY ONE of these:
- A single letter (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z)
- A single number (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
- A complete word (HELLO, HUNGRY, THANK YOU, PLEASE, SORRY, YES, NO, I, YOU, LOVE, HELP, MORE, WATER, EAT, DRINK, etc.)

PRIORITY: If this could be a word sign, choose the WORD. If it's clearly a letter/number, choose that.

COMMON ASL PATTERNS:
- Closed fist = A, S, or numbers
- Open hand = 5, HELLO, or STOP
- Pointing = 1, I, or YOU
- Pinch gesture = F, 9, or SMALL

"""

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Respond in this EXACT JSON format:
{
    "letter": "A" or null,
    "word": "HELLO" or null,
    "confidence": 0.85,
    "description": "Brief explanation"
}

Be decisive and confident in your single prediction."""

        return prompt
    
    def _create_sequence_prompt(self, gesture_descriptions: List[str], 
                              sign_language: str) -> str:
        """Create prompt for gesture sequence classification."""
        prompt = f"""Analyze this sequence of {sign_language} hand gestures:

"""
        
        for i, description in enumerate(gesture_descriptions, 1):
            prompt += f"Gesture {i}: {description}\n"
        
        prompt += f"""
What word or sentence do these {sign_language} gestures spell out when combined?
Consider the sequence and flow of the gestures."""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenAI response for single gesture classification."""
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Ensure required fields exist
                return {
                    'letter': result.get('letter'),
                    'word': result.get('word'),
                    'confidence': float(result.get('confidence', 0.0)),
                    'description': result.get('description', '')
                }
            else:
                # Fallback: simple text parsing
                return self._parse_text_response(response_text)
                
        except (json.JSONDecodeError, ValueError):
            return self._parse_text_response(response_text)
    
    def _parse_sequence_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenAI response for sequence classification."""
        try:
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                return {
                    'word': result.get('word'),
                    'sentence': result.get('sentence'),
                    'confidence': float(result.get('confidence', 0.0)),
                    'individual_letters': result.get('individual_letters', [])
                }
            else:
                return self._parse_sequence_text_response(response_text)
                
        except (json.JSONDecodeError, ValueError):
            return self._parse_sequence_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Enhanced fallback text parsing for single gesture."""
        response_lower = response_text.lower()

        # Common ASL words to look for
        common_words = ['hello', 'hungry', 'thank you', 'please', 'sorry', 'yes', 'no',
                       'i', 'you', 'love', 'help', 'more', 'water', 'eat', 'drink',
                       'good', 'bad', 'happy', 'sad', 'stop', 'go', 'come', 'home']

        # Look for words first (priority)
        word = None
        for w in common_words:
            if w in response_lower:
                word = w.upper()
                break

        # Look for letter patterns
        letter = None
        if not word:  # Only look for letters if no word found
            import re
            letter_match = re.search(r'letter\s*[:\-]?\s*([a-z])', response_lower)
            if letter_match:
                letter = letter_match.group(1).upper()
        
        # Look for word patterns
        word = None
        if 'word' in response_lower:
            word_match = re.search(r'word\s*[:\-]?\s*([a-z]+)', response_lower)
            if word_match:
                word = word_match.group(1).upper()
        
        return {
            'letter': letter,
            'word': word,
            'confidence': 0.5,  # Default confidence for text parsing
            'description': response_text[:100]  # First 100 chars
        }
    
    def _parse_sequence_text_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback text parsing for sequence."""
        return {
            'word': None,
            'sentence': None,
            'confidence': 0.5,
            'individual_letters': []
        }
