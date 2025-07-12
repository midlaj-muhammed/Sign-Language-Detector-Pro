"""
Google Gemini Sign Language Classifier

This module provides sign language classification using Google's Gemini AI API.
"""

import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
import json
import time
from dotenv import load_dotenv
from .fallback_classifier import FallbackSignLanguageClassifier

# Load environment variables
load_dotenv()


class GeminiSignLanguageClassifier:
    """
    Sign language classifier using Google Gemini AI.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini classifier.
        
        Args:
            api_key: Gemini API key (if None, will use environment variable)
            model: Gemini model to use for classification
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Enhanced rate limiting for free tier
        self.last_request_time = 0
        self.min_request_interval = 5.0  # 5 seconds between requests for free tier
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = 10  # Conservative limit for free tier
        
        # Initialize fallback classifier
        self.fallback_classifier = FallbackSignLanguageClassifier()
        
        # Debug mode
        self.debug = True
        
        print(f"Gemini classifier initialized with fallback support")
    
    def classify_gesture(self, gesture_description: str, 
                        sign_language: str = "ASL", 
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a single gesture using Gemini AI.
        
        Args:
            gesture_description: Description of the hand gesture
            sign_language: Sign language type (default: ASL)
            context: Additional context (optional)
            
        Returns:
            Classification result dictionary
        """
        self._rate_limit()
        
        # Create the prompt
        prompt = self._create_classification_prompt(gesture_description, sign_language, context)
        
        if self.debug:
            print(f"\n=== Gemini Classification Debug ===")
            print(f"Input gesture description: {gesture_description}")
            print(f"Prompt sent to Gemini: {prompt[:200]}...")
        
        try:
            response = self.model.generate_content(prompt)
            response_content = response.text
            
            if self.debug:
                print(f"Gemini response: {response_content}")
            
            result = self._parse_response(response_content)
            result['raw_response'] = response_content
            result['success'] = True
            result['method'] = 'gemini_ai'
            
            if self.debug:
                print(f"Parsed result: {result}")
                print("=== End Gemini Debug ===\n")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            if self.debug:
                print(f"Gemini API Error: {error_msg}")
                print("Falling back to pattern-based classification...")
            
            # Use fallback classifier when Gemini API fails
            try:
                fallback_result = self.fallback_classifier.classify_gesture(
                    gesture_description, sign_language, context
                )
                fallback_result['fallback_used'] = True
                fallback_result['gemini_error'] = error_msg
                
                if self.debug:
                    print(f"Fallback result: {fallback_result}")
                    print("=== End Gemini Debug ===\n")
                
                return fallback_result
                
            except Exception as fallback_error:
                if self.debug:
                    print(f"Fallback also failed: {str(fallback_error)}")
                    print("=== End Gemini Debug ===\n")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'fallback_error': str(fallback_error),
                    'letter': None,
                    'word': None,
                    'confidence': 0.0,
                    'description': None,
                    'method': 'gemini_ai'
                }
    
    def classify_sequence(self, gesture_descriptions: List[str], 
                         sign_language: str = "ASL") -> Dict[str, Any]:
        """
        Classify a sequence of gestures using Gemini AI.
        
        Args:
            gesture_descriptions: List of gesture descriptions
            sign_language: Sign language type
            
        Returns:
            Sequence classification result
        """
        self._rate_limit()
        
        # Create sequence prompt
        prompt = self._create_sequence_prompt(gesture_descriptions, sign_language)
        
        try:
            response = self.model.generate_content(prompt)
            response_content = response.text
            
            result = self._parse_sequence_response(response_content)
            result['raw_response'] = response_content
            result['success'] = True
            result['method'] = 'gemini_ai'
            
            return result
            
        except Exception as e:
            # Use fallback for sequence classification too
            try:
                fallback_result = self.fallback_classifier.classify_sequence(
                    gesture_descriptions, sign_language
                )
                fallback_result['fallback_used'] = True
                fallback_result['gemini_error'] = str(e)
                return fallback_result
                
            except Exception as fallback_error:
                return {
                    'success': False,
                    'error': str(e),
                    'fallback_error': str(fallback_error),
                    'word': None,
                    'sentence': None,
                    'confidence': 0.0,
                    'method': 'gemini_ai'
                }
    
    def _rate_limit(self):
        """Enhanced rate limiting for Gemini free tier."""
        current_time = time.time()

        # Reset request count every minute
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time

        # Check if we've hit the per-minute limit
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_window_start) + 1
            if self.debug:
                print(f"⏳ Rate limit reached, sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            self.request_count = 0
            self.request_window_start = time.time()

        # Ensure minimum interval between requests
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            if self.debug:
                print(f"⏳ Waiting {sleep_time:.1f} seconds between requests...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1
    
    def _create_classification_prompt(self, gesture_description: str,
                                    sign_language: str, context: Optional[str]) -> str:
        """Create enhanced prompt for single gesture classification."""
        prompt = f"""You are an expert ASL (American Sign Language) interpreter. Analyze this hand gesture and provide ONE CLEAR PREDICTION.

GESTURE DATA:
{gesture_description}

COMMON ASL PATTERNS TO RECOGNIZE:
• Index finger pointing = Number "1"
• Pinky finger only = Pronoun "I"
• Thumb up = "GOOD" or "YES"
• All fingers extended = Number "5" or "HELLO"
• Closed fist = Letter "A" or "S"
• Index + middle = Number "2"
• Three fingers = Number "3"
• Four fingers = Number "4"
• Index + pinky = "I LOVE YOU"
• Thumb + index = Letter "L"

TASK: Based on the finger positions described, identify what this gesture most likely represents:
- A single letter (A-Z)
- A single number (0-9)
- A complete word (HELLO, GOOD, I, YOU, LOVE, etc.)

Even if not a perfect match, provide your best interpretation based on ASL knowledge.

"""
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += """Respond in this EXACT JSON format (choose ONE prediction):
{
    "letter": "1",
    "word": null,
    "confidence": 0.85,
    "description": "Index finger pointing = Number 1"
}

OR for a word:
{
    "letter": null,
    "word": "GOOD",
    "confidence": 0.85,
    "description": "Thumb up = GOOD"
}

IMPORTANT: Always provide either a letter OR a word, never both null. Make your best guess based on ASL knowledge."""
        
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
Consider the sequence and flow of the gestures.

Respond in JSON format:
{{
    "word": "HELLO" or null,
    "sentence": "HELLO WORLD" or null,
    "confidence": 0.85,
    "individual_letters": ["H", "E", "L", "L", "O"]
}}"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response for single gesture classification."""
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Extract values
                letter = result.get('letter')
                word = result.get('word')
                confidence = float(result.get('confidence', 0.0))
                description = result.get('description', '')

                # If both are null, try to extract from description
                if not letter and not word:
                    if self.debug:
                        print("⚠️ Gemini returned null values, trying to extract from description...")

                    # Try to extract prediction from description
                    desc_lower = description.lower()

                    # Look for numbers
                    for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                        if f"number '{num}'" in desc_lower or f"number {num}" in desc_lower:
                            letter = num
                            break

                    # Look for letters
                    if not letter:
                        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                            if f"letter '{char.lower()}'" in desc_lower or f"letter {char.lower()}" in desc_lower:
                                letter = char
                                break

                    # Look for words
                    if not letter and not word:
                        common_words = ['good', 'hello', 'i', 'you', 'love', 'yes', 'no', 'please', 'thank you']
                        for w in common_words:
                            if w in desc_lower:
                                word = w.upper()
                                break

                return {
                    'letter': letter,
                    'word': word,
                    'confidence': confidence,
                    'description': description
                }
            else:
                # Fallback: simple text parsing
                return self._parse_text_response(response_text)

        except (json.JSONDecodeError, ValueError):
            return self._parse_text_response(response_text)
    
    def _parse_sequence_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response for sequence classification."""
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
            # Look for single letters
            letter_match = re.search(r'\b([A-Z])\b', response_text.upper())
            if letter_match:
                letter = letter_match.group(1)
            
            # Look for numbers
            number_match = re.search(r'\b([0-9])\b', response_text)
            if number_match:
                letter = number_match.group(1)
        
        # Extract confidence if mentioned
        confidence = 0.5  # Default
        conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100
        
        return {
            'letter': letter,
            'word': word,
            'confidence': confidence,
            'description': f"Parsed from text: {response_text[:100]}..."
        }
    
    def _parse_sequence_text_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback text parsing for sequence."""
        # Simple implementation for sequence parsing
        return {
            'word': None,
            'sentence': None,
            'confidence': 0.3,
            'individual_letters': [],
            'description': f"Text parsing fallback: {response_text[:100]}..."
        }
