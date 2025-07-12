"""
Gesture Feature Extraction Module
Processes hand landmark data into simplified format for OpenAI API classification
"""

import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional


class GestureExtractor:
    """
    A class for extracting gesture features from hand landmarks.
    """
    
    def __init__(self):
        """Initialize the GestureExtractor."""
        # Define finger tip and base indices for easier processing
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_bases = [2, 5, 9, 13, 17]  # Finger base joints
        self.finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        
    def normalize_landmarks(self, hand_landmarks: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Normalize hand landmarks relative to wrist position and hand size.
        
        Args:
            hand_landmarks: Hand landmark data from MediaPipe
            
        Returns:
            List of normalized landmark coordinates
        """
        landmarks = hand_landmarks['landmarks']
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0]
        wrist_x, wrist_y = wrist['x'], wrist['y']
        
        # Calculate hand size (distance from wrist to middle finger MCP)
        middle_mcp = landmarks[9]
        hand_size = math.sqrt(
            (middle_mcp['x'] - wrist_x) ** 2 + 
            (middle_mcp['y'] - wrist_y) ** 2
        )
        
        # Avoid division by zero
        if hand_size == 0:
            hand_size = 1.0
        
        # Normalize all landmarks
        normalized_landmarks = []
        for landmark in landmarks:
            normalized = {
                'x': (landmark['x'] - wrist_x) / hand_size,
                'y': (landmark['y'] - wrist_y) / hand_size,
                'z': landmark['z'] / hand_size
            }
            normalized_landmarks.append(normalized)
        
        return normalized_landmarks
    
    def extract_finger_states(self, normalized_landmarks: List[Dict[str, float]]) -> Dict[str, bool]:
        """
        Determine which fingers are extended or closed.
        
        Args:
            normalized_landmarks: Normalized landmark coordinates
            
        Returns:
            Dictionary with finger states (True = extended, False = closed)
        """
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_states = {}
        
        for i, finger_name in enumerate(finger_names):
            tip_idx = self.finger_tips[i]
            pip_idx = self.finger_pips[i]
            
            # For thumb, use different logic (horizontal movement)
            if finger_name == 'thumb':
                # Compare thumb tip with thumb IP joint
                tip_x = normalized_landmarks[tip_idx]['x']
                ip_x = normalized_landmarks[3]['x']  # Thumb IP joint
                finger_states[finger_name] = abs(tip_x - ip_x) > 0.1
            else:
                # For other fingers, compare tip Y with PIP Y
                tip_y = normalized_landmarks[tip_idx]['y']
                pip_y = normalized_landmarks[pip_idx]['y']
                finger_states[finger_name] = tip_y < pip_y  # Extended if tip is above PIP
        
        return finger_states
    
    def calculate_angles(self, normalized_landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate angles between key landmarks.
        
        Args:
            normalized_landmarks: Normalized landmark coordinates
            
        Returns:
            Dictionary of calculated angles
        """
        angles = {}
        
        # Calculate angle between thumb and index finger
        thumb_tip = normalized_landmarks[4]
        index_tip = normalized_landmarks[8]
        wrist = normalized_landmarks[0]
        
        # Vector from wrist to thumb tip
        thumb_vector = np.array([thumb_tip['x'] - wrist['x'], thumb_tip['y'] - wrist['y']])
        # Vector from wrist to index tip
        index_vector = np.array([index_tip['x'] - wrist['x'], index_tip['y'] - wrist['y']])
        
        # Calculate angle between vectors
        dot_product = np.dot(thumb_vector, index_vector)
        norms = np.linalg.norm(thumb_vector) * np.linalg.norm(index_vector)
        
        if norms > 0:
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angles['thumb_index_angle'] = math.degrees(math.acos(cos_angle))
        else:
            angles['thumb_index_angle'] = 0.0
        
        # Calculate hand orientation (angle of palm)
        middle_mcp = normalized_landmarks[9]
        wrist = normalized_landmarks[0]
        palm_vector = np.array([middle_mcp['x'] - wrist['x'], middle_mcp['y'] - wrist['y']])
        
        # Angle with vertical axis
        vertical = np.array([0, -1])  # Pointing up
        dot_product = np.dot(palm_vector, vertical)
        norms = np.linalg.norm(palm_vector) * np.linalg.norm(vertical)
        
        if norms > 0:
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles['palm_orientation'] = math.degrees(math.acos(cos_angle))
        else:
            angles['palm_orientation'] = 0.0
        
        return angles
    
    def extract_distances(self, normalized_landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate distances between key landmarks.
        
        Args:
            normalized_landmarks: Normalized landmark coordinates
            
        Returns:
            Dictionary of calculated distances
        """
        distances = {}
        
        # Distance between thumb tip and index tip
        thumb_tip = normalized_landmarks[4]
        index_tip = normalized_landmarks[8]
        distances['thumb_index_distance'] = math.sqrt(
            (thumb_tip['x'] - index_tip['x']) ** 2 + 
            (thumb_tip['y'] - index_tip['y']) ** 2
        )
        
        # Distance between index and middle finger tips
        middle_tip = normalized_landmarks[12]
        distances['index_middle_distance'] = math.sqrt(
            (index_tip['x'] - middle_tip['x']) ** 2 + 
            (index_tip['y'] - middle_tip['y']) ** 2
        )
        
        # Distance from wrist to each fingertip
        wrist = normalized_landmarks[0]
        for i, finger_name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
            tip_idx = self.finger_tips[i]
            tip = normalized_landmarks[tip_idx]
            distances[f'wrist_{finger_name}_distance'] = math.sqrt(
                (tip['x'] - wrist['x']) ** 2 + 
                (tip['y'] - wrist['y']) ** 2
            )
        
        return distances
    
    def create_gesture_description(self, hand_landmarks: Dict[str, Any]) -> str:
        """
        Create a textual description of the gesture for OpenAI API.
        
        Args:
            hand_landmarks: Hand landmark data from MediaPipe
            
        Returns:
            String description of the gesture
        """
        normalized_landmarks = self.normalize_landmarks(hand_landmarks)
        finger_states = self.extract_finger_states(normalized_landmarks)
        angles = self.calculate_angles(normalized_landmarks)
        distances = self.extract_distances(normalized_landmarks)
        
        # Create description
        description_parts = []
        
        # Hand label
        description_parts.append(f"Hand: {hand_landmarks['label']}")
        
        # Finger states
        extended_fingers = [name for name, extended in finger_states.items() if extended]
        closed_fingers = [name for name, extended in finger_states.items() if not extended]
        
        if extended_fingers:
            description_parts.append(f"Extended fingers: {', '.join(extended_fingers)}")
        if closed_fingers:
            description_parts.append(f"Closed fingers: {', '.join(closed_fingers)}")
        
        # Key measurements
        description_parts.append(f"Thumb-index angle: {angles['thumb_index_angle']:.1f} degrees")
        description_parts.append(f"Thumb-index distance: {distances['thumb_index_distance']:.3f}")
        description_parts.append(f"Palm orientation: {angles['palm_orientation']:.1f} degrees")
        
        # Special gesture patterns
        if all(not extended for extended in finger_states.values()):
            description_parts.append("Pattern: Closed fist")
        elif all(extended for extended in finger_states.values()):
            description_parts.append("Pattern: Open hand")
        elif finger_states['index'] and not any(finger_states[f] for f in ['middle', 'ring', 'pinky']):
            description_parts.append("Pattern: Pointing gesture")
        elif finger_states['thumb'] and finger_states['index'] and distances['thumb_index_distance'] < 0.1:
            description_parts.append("Pattern: Pinch gesture")
        
        return "; ".join(description_parts)
    
    def extract_features_vector(self, hand_landmarks: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical feature vector for machine learning models.
        
        Args:
            hand_landmarks: Hand landmark data from MediaPipe
            
        Returns:
            NumPy array of features
        """
        normalized_landmarks = self.normalize_landmarks(hand_landmarks)
        finger_states = self.extract_finger_states(normalized_landmarks)
        angles = self.calculate_angles(normalized_landmarks)
        distances = self.extract_distances(normalized_landmarks)
        
        # Create feature vector
        features = []
        
        # Finger states (5 features)
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            features.append(1.0 if finger_states[finger] else 0.0)
        
        # Angles (2 features)
        features.extend([
            angles['thumb_index_angle'] / 180.0,  # Normalize to 0-1
            angles['palm_orientation'] / 180.0
        ])
        
        # Distances (7 features)
        features.extend([
            distances['thumb_index_distance'],
            distances['index_middle_distance'],
            distances['wrist_thumb_distance'],
            distances['wrist_index_distance'],
            distances['wrist_middle_distance'],
            distances['wrist_ring_distance'],
            distances['wrist_pinky_distance']
        ])
        
        return np.array(features)
