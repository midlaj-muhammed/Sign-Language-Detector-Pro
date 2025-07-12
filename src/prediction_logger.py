"""
Comprehensive Prediction Logging System

This module provides detailed logging for the sign language prediction pipeline
to help identify where predictions are failing and track performance.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


class PredictionLogger:
    """
    Comprehensive logging system for sign language predictions.
    """
    
    def __init__(self, log_file: str = "prediction_logs.json", debug: bool = True):
        """
        Initialize the prediction logger.
        
        Args:
            log_file: Path to the log file
            debug: Whether to print debug information
        """
        self.log_file = log_file
        self.debug = debug
        self.session_id = f"session_{int(time.time())}"
        self.logs = []
        
        if self.debug:
            print(f"🔍 Prediction Logger initialized - Session: {self.session_id}")
    
    def log_hand_detection(self, image_info: Dict[str, Any], hands_detected: int, 
                          detection_confidence: List[float] = None) -> str:
        """
        Log hand detection results.
        
        Args:
            image_info: Information about the processed image
            hands_detected: Number of hands detected
            detection_confidence: List of confidence scores for detected hands
            
        Returns:
            Log entry ID
        """
        log_entry = {
            "id": f"hand_det_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "stage": "hand_detection",
            "image_info": image_info,
            "hands_detected": hands_detected,
            "detection_confidence": detection_confidence or [],
            "success": hands_detected > 0
        }
        
        self.logs.append(log_entry)
        
        if self.debug:
            status = "✅" if hands_detected > 0 else "❌"
            print(f"{status} Hand Detection: {hands_detected} hands detected")
            if detection_confidence:
                for i, conf in enumerate(detection_confidence):
                    print(f"   Hand {i+1}: {conf:.1%} confidence")
        
        return log_entry["id"]
    
    def log_gesture_extraction(self, hand_data: Dict[str, Any], 
                             gesture_description: str) -> str:
        """
        Log gesture extraction results.
        
        Args:
            hand_data: Hand landmark data
            gesture_description: Generated gesture description
            
        Returns:
            Log entry ID
        """
        log_entry = {
            "id": f"gest_ext_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "stage": "gesture_extraction",
            "hand_label": hand_data.get('label', 'Unknown'),
            "hand_confidence": hand_data.get('confidence', 0.0),
            "gesture_description": gesture_description,
            "description_length": len(gesture_description),
            "success": len(gesture_description) > 0
        }
        
        self.logs.append(log_entry)
        
        if self.debug:
            print(f"✅ Gesture Extraction: {len(gesture_description)} chars description")
            print(f"   Hand: {hand_data.get('label', 'Unknown')} ({hand_data.get('confidence', 0):.1%})")
        
        return log_entry["id"]
    
    def log_ai_classification(self, gesture_description: str, ai_provider: str,
                            response: Dict[str, Any], success: bool, 
                            error_message: str = None) -> str:
        """
        Log AI classification attempts.
        
        Args:
            gesture_description: Input gesture description
            ai_provider: AI provider used (gemini, openai, etc.)
            response: AI response data
            success: Whether the classification succeeded
            error_message: Error message if failed
            
        Returns:
            Log entry ID
        """
        log_entry = {
            "id": f"ai_class_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "stage": "ai_classification",
            "ai_provider": ai_provider,
            "input_description": gesture_description,
            "response": response,
            "success": success,
            "error_message": error_message,
            "prediction": response.get('word') or response.get('letter') if success else None,
            "confidence": response.get('confidence', 0.0) if success else 0.0
        }
        
        self.logs.append(log_entry)
        
        if self.debug:
            status = "✅" if success else "❌"
            if success:
                prediction = response.get('word') or response.get('letter') or 'No prediction'
                confidence = response.get('confidence', 0.0)
                print(f"{status} AI Classification ({ai_provider}): {prediction} ({confidence:.1%})")
            else:
                print(f"{status} AI Classification ({ai_provider}) Failed: {error_message}")
        
        return log_entry["id"]
    
    def log_fallback_classification(self, gesture_description: str, 
                                  response: Dict[str, Any], success: bool) -> str:
        """
        Log fallback classification results.
        
        Args:
            gesture_description: Input gesture description
            response: Fallback classifier response
            success: Whether the classification succeeded
            
        Returns:
            Log entry ID
        """
        log_entry = {
            "id": f"fallback_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "stage": "fallback_classification",
            "input_description": gesture_description,
            "response": response,
            "success": success,
            "prediction": response.get('word') or response.get('letter') if success else None,
            "confidence": response.get('confidence', 0.0) if success else 0.0
        }
        
        self.logs.append(log_entry)
        
        if self.debug:
            status = "✅" if success else "❌"
            if success:
                prediction = response.get('word') or response.get('letter') or 'No prediction'
                confidence = response.get('confidence', 0.0)
                print(f"{status} Fallback Classification: {prediction} ({confidence:.1%})")
            else:
                print(f"{status} Fallback Classification Failed")
        
        return log_entry["id"]
    
    def log_final_prediction(self, file_path: str, final_prediction: str, 
                           confidence: float, method_used: str, 
                           processing_time: float) -> str:
        """
        Log final prediction results.
        
        Args:
            file_path: Path to the processed file
            final_prediction: Final prediction result
            confidence: Prediction confidence
            method_used: Method that provided the final prediction
            processing_time: Total processing time in seconds
            
        Returns:
            Log entry ID
        """
        log_entry = {
            "id": f"final_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "stage": "final_prediction",
            "file_path": file_path,
            "final_prediction": final_prediction,
            "confidence": confidence,
            "method_used": method_used,
            "processing_time": processing_time,
            "success": final_prediction is not None and final_prediction != "No prediction"
        }
        
        self.logs.append(log_entry)
        
        if self.debug:
            status = "🎯" if log_entry["success"] else "❌"
            print(f"{status} Final Prediction: {final_prediction} ({confidence:.1%}) via {method_used}")
            print(f"   Processing time: {processing_time:.2f}s")
        
        return log_entry["id"]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session.
        
        Returns:
            Session summary statistics
        """
        total_predictions = len([log for log in self.logs if log["stage"] == "final_prediction"])
        successful_predictions = len([log for log in self.logs 
                                    if log["stage"] == "final_prediction" and log["success"]])
        
        hand_detections = len([log for log in self.logs if log["stage"] == "hand_detection"])
        successful_hand_detections = len([log for log in self.logs 
                                        if log["stage"] == "hand_detection" and log["success"]])
        
        ai_attempts = len([log for log in self.logs if log["stage"] == "ai_classification"])
        successful_ai = len([log for log in self.logs 
                           if log["stage"] == "ai_classification" and log["success"]])
        
        fallback_attempts = len([log for log in self.logs if log["stage"] == "fallback_classification"])
        
        summary = {
            "session_id": self.session_id,
            "total_files_processed": total_predictions,
            "successful_predictions": successful_predictions,
            "prediction_success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
            "hand_detection_success_rate": successful_hand_detections / hand_detections if hand_detections > 0 else 0,
            "ai_classification_success_rate": successful_ai / ai_attempts if ai_attempts > 0 else 0,
            "fallback_usage_rate": fallback_attempts / total_predictions if total_predictions > 0 else 0,
            "total_logs": len(self.logs)
        }
        
        return summary
    
    def save_logs(self) -> bool:
        """
        Save logs to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    "session_summary": self.get_session_summary(),
                    "logs": self.logs
                }, f, indent=2)
            
            if self.debug:
                print(f"💾 Logs saved to {self.log_file}")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to save logs: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of the current session."""
        summary = self.get_session_summary()
        
        print("\n" + "="*50)
        print("📊 PREDICTION SESSION SUMMARY")
        print("="*50)
        print(f"Session ID: {summary['session_id']}")
        print(f"Files Processed: {summary['total_files_processed']}")
        print(f"Successful Predictions: {summary['successful_predictions']}")
        print(f"Prediction Success Rate: {summary['prediction_success_rate']:.1%}")
        print(f"Hand Detection Success Rate: {summary['hand_detection_success_rate']:.1%}")
        print(f"AI Classification Success Rate: {summary['ai_classification_success_rate']:.1%}")
        print(f"Fallback Usage Rate: {summary['fallback_usage_rate']:.1%}")
        print(f"Total Log Entries: {summary['total_logs']}")
        print("="*50)
