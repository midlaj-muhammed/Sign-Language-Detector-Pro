"""
Real-time Camera Input Handler for Sign Language Detection
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Callable, Dict, Any, List
from queue import Queue, Empty

from .hand_detector import HandDetector
from .gesture_extractor import GestureExtractor
from .openai_classifier import SignLanguageClassifier


class CameraHandler:
    """
    Handles real-time camera input for sign language detection.
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 fps: int = 30,
                 detection_interval: float = 2.0):
        """
        Initialize the CameraHandler.
        
        Args:
            camera_index: Index of the camera to use
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
            fps: Frames per second for camera capture
            detection_interval: Seconds between gesture classifications
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.detection_interval = detection_interval
        
        # Initialize components
        self.hand_detector = HandDetector()
        self.gesture_extractor = GestureExtractor()
        self.classifier = None  # Will be initialized when needed
        
        # Camera and threading
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.detection_thread = None
        
        # Frame and detection queues
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=5)
        
        # Callbacks
        self.on_frame_callback = None
        self.on_detection_callback = None
        
        # Detection state
        self.last_detection_time = 0
        self.gesture_history = []
        self.max_history_length = 10
        
    def initialize_camera(self) -> bool:
        """
        Initialize the camera.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def initialize_classifier(self, api_key: Optional[str] = None) -> bool:
        """
        Initialize the OpenAI classifier.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            True if classifier initialized successfully, False otherwise
        """
        try:
            self.classifier = SignLanguageClassifier(api_key=api_key)
            print("OpenAI classifier initialized")
            return True
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            return False
    
    def set_callbacks(self, 
                     on_frame: Optional[Callable] = None,
                     on_detection: Optional[Callable] = None):
        """
        Set callback functions for frame and detection events.
        
        Args:
            on_frame: Callback for each processed frame
            on_detection: Callback for gesture detections
        """
        self.on_frame_callback = on_frame
        self.on_detection_callback = on_detection
    
    def start_capture(self) -> bool:
        """
        Start the camera capture and detection threads.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized")
            return False
        
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        print("Camera capture started")
        return True
    
    def stop_capture(self):
        """Stop the camera capture and detection threads."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        print("Camera capture stopped")
    
    def _capture_loop(self):
        """Main camera capture loop (runs in separate thread)."""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error reading frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            annotated_frame, hand_landmarks = self.hand_detector.detect_hands(frame)
            
            # Add frame to queue for detection processing
            if not self.frame_queue.full():
                self.frame_queue.put((frame.copy(), hand_landmarks))
            
            # Call frame callback if set
            if self.on_frame_callback:
                self.on_frame_callback(annotated_frame, hand_landmarks)
            
            # Small delay to control frame rate
            time.sleep(1.0 / self.fps)
    
    def _detection_loop(self):
        """Gesture detection and classification loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Get frame from queue
                frame, hand_landmarks = self.frame_queue.get(timeout=1.0)
                
                # Check if enough time has passed since last detection
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_interval:
                    continue
                
                # Process gestures if hands detected
                if hand_landmarks and self.classifier:
                    self._process_gestures(hand_landmarks)
                    self.last_detection_time = current_time
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in detection loop: {e}")
    
    def _process_gestures(self, hand_landmarks: List[Dict[str, Any]]):
        """
        Process detected hand landmarks and classify gestures.
        
        Args:
            hand_landmarks: List of detected hand landmarks
        """
        detections = []
        
        for hand_data in hand_landmarks:
            try:
                # Extract gesture features
                gesture_description = self.gesture_extractor.create_gesture_description(hand_data)
                
                # Classify gesture
                classification = self.classifier.classify_gesture(gesture_description)
                
                if classification['success']:
                    detection = {
                        'hand_label': hand_data['label'],
                        'gesture_description': gesture_description,
                        'classification': classification,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
                    
                    # Add to gesture history
                    self.gesture_history.append(detection)
                    if len(self.gesture_history) > self.max_history_length:
                        self.gesture_history.pop(0)
            
            except Exception as e:
                print(f"Error processing gesture: {e}")
        
        # Call detection callback if detections found
        if detections and self.on_detection_callback:
            self.on_detection_callback(detections)
    
    def get_recent_gestures(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent gesture detections.
        
        Args:
            count: Number of recent gestures to return
            
        Returns:
            List of recent gesture detections
        """
        return self.gesture_history[-count:] if self.gesture_history else []
    
    def classify_gesture_sequence(self, count: int = 5) -> Optional[Dict[str, Any]]:
        """
        Classify a sequence of recent gestures.
        
        Args:
            count: Number of recent gestures to include in sequence
            
        Returns:
            Sequence classification result or None
        """
        if not self.classifier or len(self.gesture_history) < 2:
            return None
        
        recent_gestures = self.get_recent_gestures(count)
        gesture_descriptions = [g['gesture_description'] for g in recent_gestures]
        
        try:
            return self.classifier.classify_sequence(gesture_descriptions)
        except Exception as e:
            print(f"Error classifying gesture sequence: {e}")
            return None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Captured frame or None if error
        """
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return cv2.flip(frame, 1)  # Mirror effect
        return None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_capture()
        
        if self.hand_detector:
            self.hand_detector.cleanup()
        
        cv2.destroyAllWindows()
