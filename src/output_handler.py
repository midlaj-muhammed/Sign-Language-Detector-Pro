"""
Output Display and Speech Synthesis Module
Handles text display and text-to-speech functionality
"""

import pyttsx3
import threading
import time
import os
from typing import List, Dict, Any, Optional, Callable
from queue import Queue, Empty
import json
from datetime import datetime


class OutputHandler:
    """
    Handles text display and speech synthesis for sign language detection results.
    """
    
    def __init__(self, 
                 enable_speech: bool = True,
                 speech_rate: int = 150,
                 speech_volume: float = 0.9,
                 save_transcript: bool = True,
                 transcript_file: str = "sign_language_transcript.txt"):
        """
        Initialize the OutputHandler.
        
        Args:
            enable_speech: Whether to enable text-to-speech
            speech_rate: Speech rate (words per minute)
            speech_volume: Speech volume (0.0 to 1.0)
            save_transcript: Whether to save transcript to file
            transcript_file: Path to transcript file
        """
        self.enable_speech = enable_speech
        self.speech_rate = speech_rate
        self.speech_volume = speech_volume
        self.save_transcript = save_transcript
        self.transcript_file = transcript_file
        
        # Initialize TTS engine
        self.tts_engine = None
        self.tts_thread = None
        self.speech_queue = Queue()
        self.is_speaking = False
        
        # Transcript storage
        self.transcript = []
        self.current_session_start = datetime.now()
        
        # Display callbacks
        self.display_callbacks = []
        
        # Initialize TTS if enabled
        if self.enable_speech:
            self._initialize_tts()
    
    def _initialize_tts(self) -> bool:
        """
        Initialize the text-to-speech engine.
        
        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            self.tts_engine = pyttsx3.init()
            
            # Set properties
            self.tts_engine.setProperty('rate', self.speech_rate)
            self.tts_engine.setProperty('volume', self.speech_volume)
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Start TTS thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            print("Text-to-speech initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            self.enable_speech = False
            return False
    
    def _tts_worker(self):
        """TTS worker thread that processes speech queue."""
        while True:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text is None:  # Shutdown signal
                    break
                
                self.is_speaking = True
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.is_speaking = False
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in TTS worker: {e}")
                self.is_speaking = False
    
    def add_display_callback(self, callback: Callable):
        """
        Add a callback function for display updates.
        
        Args:
            callback: Function to call when display should be updated
        """
        self.display_callbacks.append(callback)
    
    def display_detection(self, detection: Dict[str, Any], speak: bool = True):
        """
        Display and optionally speak a gesture detection result.
        
        Args:
            detection: Detection result dictionary
            speak: Whether to speak the result
        """
        # Extract relevant information
        hand_label = detection.get('hand_label', 'Unknown')
        classification = detection.get('classification', {})
        
        if not classification.get('success', False):
            return
        
        # Format display text
        display_text = self._format_detection_text(detection)
        
        # Add to transcript
        if self.save_transcript:
            self._add_to_transcript(detection, display_text)
        
        # Call display callbacks
        for callback in self.display_callbacks:
            try:
                callback(display_text, detection)
            except Exception as e:
                print(f"Error in display callback: {e}")
        
        # Speak if enabled and requested
        if speak and self.enable_speech:
            speech_text = self._format_speech_text(detection)
            self.speak(speech_text)
        
        # Print to console
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {display_text}")
    
    def display_sequence(self, sequence_result: Dict[str, Any], speak: bool = True):
        """
        Display and optionally speak a gesture sequence result.
        
        Args:
            sequence_result: Sequence classification result
            speak: Whether to speak the result
        """
        if not sequence_result.get('success', False):
            return
        
        # Format display text
        display_text = self._format_sequence_text(sequence_result)
        
        # Add to transcript
        if self.save_transcript:
            self._add_sequence_to_transcript(sequence_result, display_text)
        
        # Call display callbacks
        for callback in self.display_callbacks:
            try:
                callback(display_text, sequence_result)
            except Exception as e:
                print(f"Error in display callback: {e}")
        
        # Speak if enabled and requested
        if speak and self.enable_speech:
            speech_text = self._format_sequence_speech_text(sequence_result)
            self.speak(speech_text)
        
        # Print to console
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SEQUENCE: {display_text}")
    
    def speak(self, text: str):
        """
        Add text to speech queue.
        
        Args:
            text: Text to speak
        """
        if self.enable_speech and not self.is_speaking:
            self.speech_queue.put(text)
    
    def _format_detection_text(self, detection: Dict[str, Any]) -> str:
        """Format detection result for display."""
        classification = detection.get('classification', {})
        hand_label = detection.get('hand_label', 'Unknown')
        
        parts = [f"{hand_label} hand:"]
        
        if classification.get('letter'):
            parts.append(f"Letter '{classification['letter']}'")
        
        if classification.get('word'):
            parts.append(f"Word '{classification['word']}'")
        
        confidence = classification.get('confidence', 0.0)
        if confidence > 0:
            parts.append(f"({confidence:.1%} confidence)")
        
        return " ".join(parts)
    
    def _format_sequence_text(self, sequence_result: Dict[str, Any]) -> str:
        """Format sequence result for display."""
        parts = []
        
        if sequence_result.get('word'):
            parts.append(f"Word: '{sequence_result['word']}'")
        
        if sequence_result.get('sentence'):
            parts.append(f"Sentence: '{sequence_result['sentence']}'")
        
        if sequence_result.get('individual_letters'):
            letters = " ".join(sequence_result['individual_letters'])
            parts.append(f"Letters: {letters}")
        
        confidence = sequence_result.get('confidence', 0.0)
        if confidence > 0:
            parts.append(f"({confidence:.1%} confidence)")
        
        return " | ".join(parts)
    
    def _format_speech_text(self, detection: Dict[str, Any]) -> str:
        """Format detection result for speech."""
        classification = detection.get('classification', {})
        
        if classification.get('word'):
            return classification['word']
        elif classification.get('letter'):
            return f"Letter {classification['letter']}"
        else:
            return "Gesture detected"
    
    def _format_sequence_speech_text(self, sequence_result: Dict[str, Any]) -> str:
        """Format sequence result for speech."""
        if sequence_result.get('sentence'):
            return sequence_result['sentence']
        elif sequence_result.get('word'):
            return sequence_result['word']
        else:
            return "Sequence detected"
    
    def _add_to_transcript(self, detection: Dict[str, Any], display_text: str):
        """Add detection to transcript."""
        transcript_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'detection',
            'display_text': display_text,
            'detection': detection
        }
        self.transcript.append(transcript_entry)
        
        # Save to file periodically
        if len(self.transcript) % 10 == 0:
            self._save_transcript()
    
    def _add_sequence_to_transcript(self, sequence_result: Dict[str, Any], display_text: str):
        """Add sequence to transcript."""
        transcript_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'sequence',
            'display_text': display_text,
            'sequence_result': sequence_result
        }
        self.transcript.append(transcript_entry)
        self._save_transcript()
    
    def _save_transcript(self):
        """Save transcript to file."""
        if not self.save_transcript:
            return
        
        try:
            # Create transcript data
            transcript_data = {
                'session_start': self.current_session_start.isoformat(),
                'last_updated': datetime.now().isoformat(),
                'entries': self.transcript
            }
            
            # Save as JSON
            json_file = os.path.splitext(self.transcript_file)[0] + '.json'
            with open(json_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            # Save as readable text
            with open(self.transcript_file, 'w') as f:
                f.write(f"Sign Language Detection Transcript\n")
                f.write(f"Session started: {self.current_session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for entry in self.transcript:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    f.write(f"[{timestamp.strftime('%H:%M:%S')}] {entry['display_text']}\n")
                
        except Exception as e:
            print(f"Error saving transcript: {e}")
    
    def get_transcript_summary(self) -> Dict[str, Any]:
        """
        Get summary of current transcript.
        
        Returns:
            Dictionary containing transcript summary
        """
        if not self.transcript:
            return {'total_entries': 0, 'detections': 0, 'sequences': 0}
        
        detections = sum(1 for entry in self.transcript if entry['type'] == 'detection')
        sequences = sum(1 for entry in self.transcript if entry['type'] == 'sequence')
        
        # Extract detected words and letters
        detected_words = []
        detected_letters = []
        
        for entry in self.transcript:
            if entry['type'] == 'detection':
                classification = entry.get('detection', {}).get('classification', {})
                if classification.get('word'):
                    detected_words.append(classification['word'])
                if classification.get('letter'):
                    detected_letters.append(classification['letter'])
            elif entry['type'] == 'sequence':
                sequence_result = entry.get('sequence_result', {})
                if sequence_result.get('word'):
                    detected_words.append(sequence_result['word'])
                if sequence_result.get('sentence'):
                    detected_words.extend(sequence_result['sentence'].split())
        
        return {
            'total_entries': len(self.transcript),
            'detections': detections,
            'sequences': sequences,
            'detected_words': list(set(detected_words)),
            'detected_letters': list(set(detected_letters)),
            'session_duration': (datetime.now() - self.current_session_start).total_seconds()
        }
    
    def clear_transcript(self):
        """Clear the current transcript."""
        self.transcript = []
        self.current_session_start = datetime.now()
        print("Transcript cleared")
    
    def set_speech_enabled(self, enabled: bool):
        """Enable or disable speech synthesis."""
        self.enable_speech = enabled
        if not enabled and self.is_speaking:
            # Stop current speech
            if self.tts_engine:
                self.tts_engine.stop()
    
    def cleanup(self):
        """Clean up resources."""
        # Save final transcript
        if self.save_transcript and self.transcript:
            self._save_transcript()
        
        # Stop TTS
        if self.tts_thread:
            self.speech_queue.put(None)  # Shutdown signal
            self.tts_thread.join(timeout=2.0)
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
