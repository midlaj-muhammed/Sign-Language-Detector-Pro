"""
Hand Landmark Detection Module using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class HandDetector:
    """
    A class for detecting hand landmarks using MediaPipe Hands.
    """
    
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,  # Lowered for better detection
                 min_tracking_confidence: float = 0.3):  # Lowered for better detection
        """
        Initialize the HandDetector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect hands in the given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_image, hand_landmarks_list)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification (Left/Right)
                hand_label = "Unknown"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                
                # Draw landmarks on the image
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                hand_data = {
                    'label': hand_label,
                    'landmarks': landmarks,
                    'confidence': results.multi_handedness[idx].classification[0].score if results.multi_handedness else 0.0
                }
                
                hand_landmarks_list.append(hand_data)
        
        return annotated_image, hand_landmarks_list
    
    def get_landmark_positions(self, hand_landmarks: List[Dict[str, Any]], 
                             image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            hand_landmarks: List of hand landmark data
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            List of (x, y) pixel coordinates
        """
        positions = []
        for hand_data in hand_landmarks:
            hand_positions = []
            for landmark in hand_data['landmarks']:
                x = int(landmark['x'] * image_width)
                y = int(landmark['y'] * image_height)
                hand_positions.append((x, y))
            positions.append(hand_positions)
        
        return positions
    
    def get_bounding_box(self, hand_landmarks: Dict[str, Any], 
                        image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Get bounding box for detected hand.
        
        Args:
            hand_landmarks: Hand landmark data
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = [landmark['x'] * image_width for landmark in hand_landmarks['landmarks']]
        y_coords = [landmark['y'] * image_height for landmark in hand_landmarks['landmarks']]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        
        return x_min, y_min, x_max, y_max
    
    def is_hand_closed(self, hand_landmarks: Dict[str, Any]) -> bool:
        """
        Simple heuristic to determine if hand is closed (fist).
        
        Args:
            hand_landmarks: Hand landmark data
            
        Returns:
            Boolean indicating if hand appears closed
        """
        landmarks = hand_landmarks['landmarks']
        
        # Check if fingertips are below their respective PIP joints
        # Thumb: tip (4) vs IP (3)
        # Index: tip (8) vs PIP (6)
        # Middle: tip (12) vs PIP (10)
        # Ring: tip (16) vs PIP (14)
        # Pinky: tip (20) vs PIP (18)
        
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        closed_fingers = 0
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip]['y'] > landmarks[pip]['y']:  # tip below pip
                closed_fingers += 1
        
        # Consider hand closed if 4 or more fingers are closed
        return closed_fingers >= 4
    
    def cleanup(self):
        """
        Clean up MediaPipe resources.
        """
        if hasattr(self, 'hands'):
            self.hands.close()


# Landmark indices for reference
HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
}
