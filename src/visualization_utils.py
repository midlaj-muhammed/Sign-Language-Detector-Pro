"""
Visualization utilities for enhanced result display
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple
import pandas as pd


class HandLandmarkVisualizer:
    """
    Enhanced visualization for hand landmarks and gesture analysis.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # MediaPipe hand landmark connections
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        # Color scheme for different parts
        self.colors = {
            'thumb': (255, 0, 0),      # Red
            'index': (0, 255, 0),      # Green
            'middle': (0, 0, 255),     # Blue
            'ring': (255, 255, 0),     # Yellow
            'pinky': (255, 0, 255),    # Magenta
            'palm': (0, 255, 255),     # Cyan
            'wrist': (128, 128, 128)   # Gray
        }
        
        # Finger landmark ranges
        self.finger_ranges = {
            'thumb': range(1, 5),
            'index': range(5, 9),
            'middle': range(9, 13),
            'ring': range(13, 17),
            'pinky': range(17, 21),
            'wrist': [0]
        }
    
    def draw_enhanced_landmarks(self, image: np.ndarray, 
                              hand_landmarks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw enhanced hand landmarks with color coding and connections.
        
        Args:
            image: Input image
            hand_landmarks: List of hand landmark data
            
        Returns:
            Image with enhanced landmarks drawn
        """
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        for hand_data in hand_landmarks:
            landmarks = hand_data['landmarks']
            hand_label = hand_data['label']
            
            # Convert normalized coordinates to pixel coordinates
            landmark_points = []
            for landmark in landmarks:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                landmark_points.append((x, y))
            
            # Draw connections
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                
                # Determine color based on finger
                color = self._get_connection_color(start_idx, end_idx)
                cv2.line(annotated_image, start_point, end_point, color, 2)
            
            # Draw landmark points
            for i, point in enumerate(landmark_points):
                color = self._get_landmark_color(i)
                cv2.circle(annotated_image, point, 4, color, -1)
                cv2.circle(annotated_image, point, 6, (255, 255, 255), 1)
            
            # Add hand label
            if landmark_points:
                label_pos = (landmark_points[0][0] - 50, landmark_points[0][1] - 20)
                cv2.putText(annotated_image, f"{hand_label} Hand", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, f"{hand_label} Hand", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return annotated_image
    
    def _get_landmark_color(self, landmark_idx: int) -> Tuple[int, int, int]:
        """Get color for a specific landmark."""
        for finger, indices in self.finger_ranges.items():
            if landmark_idx in indices:
                return self.colors[finger]
        return (128, 128, 128)  # Default gray
    
    def _get_connection_color(self, start_idx: int, end_idx: int) -> Tuple[int, int, int]:
        """Get color for a connection between landmarks."""
        # Use the color of the finger that both landmarks belong to
        for finger, indices in self.finger_ranges.items():
            if start_idx in indices and end_idx in indices:
                return self.colors[finger]
        return self.colors['palm']  # Default to palm color
    
    def create_3d_hand_plot(self, hand_landmarks: Dict[str, Any]) -> go.Figure:
        """
        Create a 3D visualization of hand landmarks.
        
        Args:
            hand_landmarks: Hand landmark data
            
        Returns:
            Plotly 3D figure
        """
        landmarks = hand_landmarks['landmarks']
        
        # Extract coordinates
        x_coords = [landmark['x'] for landmark in landmarks]
        y_coords = [-landmark['y'] for landmark in landmarks]  # Flip Y for proper orientation
        z_coords = [landmark['z'] for landmark in landmarks]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add landmark points
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=8,
                color=z_coords,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Depth")
            ),
            text=[f"Landmark {i}" for i in range(len(landmarks))],
            name="Hand Landmarks"
        ))
        
        # Add connections
        for connection in self.hand_connections:
            start_idx, end_idx = connection
            fig.add_trace(go.Scatter3d(
                x=[x_coords[start_idx], x_coords[end_idx]],
                y=[y_coords[start_idx], y_coords[end_idx]],
                z=[z_coords[start_idx], z_coords[end_idx]],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.6)', width=3),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title=f"3D Hand Landmarks - {hand_landmarks['label']} Hand",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z (Depth)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=600,
            height=500
        )
        
        return fig
    
    def create_gesture_feature_radar(self, gesture_features: Dict[str, float]) -> go.Figure:
        """
        Create a radar chart for gesture features.
        
        Args:
            gesture_features: Dictionary of gesture features
            
        Returns:
            Plotly radar chart figure
        """
        # Normalize features for radar chart
        features = ['Thumb Ext.', 'Index Ext.', 'Middle Ext.', 'Ring Ext.', 'Pinky Ext.',
                   'Thumb-Index Angle', 'Palm Orientation', 'Hand Openness']
        
        # Extract and normalize values
        values = [
            gesture_features.get('thumb_extended', 0),
            gesture_features.get('index_extended', 0),
            gesture_features.get('middle_extended', 0),
            gesture_features.get('ring_extended', 0),
            gesture_features.get('pinky_extended', 0),
            gesture_features.get('thumb_index_angle', 0) / 180,  # Normalize angle
            gesture_features.get('palm_orientation', 0) / 180,   # Normalize angle
            gesture_features.get('hand_openness', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features,
            fill='toself',
            name='Gesture Features',
            line_color='rgb(46, 134, 171)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Gesture Feature Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence: float, title: str = "Confidence") -> go.Figure:
        """
        Create a gauge chart for confidence scores.
        
        Args:
            confidence: Confidence value (0-1)
            title: Title for the gauge
            
        Returns:
            Plotly gauge figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig


def create_comparison_view(original_image: np.ndarray, 
                          annotated_image: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison view of original and annotated images.
    
    Args:
        original_image: Original input image
        annotated_image: Image with landmarks drawn
        
    Returns:
        Combined comparison image
    """
    # Ensure both images have the same height
    height = max(original_image.shape[0], annotated_image.shape[0])
    
    # Resize images to same height if needed
    if original_image.shape[0] != height:
        aspect_ratio = original_image.shape[1] / original_image.shape[0]
        new_width = int(height * aspect_ratio)
        original_image = cv2.resize(original_image, (new_width, height))
    
    if annotated_image.shape[0] != height:
        aspect_ratio = annotated_image.shape[1] / annotated_image.shape[0]
        new_width = int(height * aspect_ratio)
        annotated_image = cv2.resize(annotated_image, (new_width, height))
    
    # Create comparison image
    comparison = np.hstack([original_image, annotated_image])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Detected", (original_image.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return comparison


def create_processing_timeline(frame_detections: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline visualization for video processing results.
    
    Args:
        frame_detections: List of frame detection results
        
    Returns:
        Plotly timeline figure
    """
    if not frame_detections:
        return go.Figure()
    
    # Prepare data
    timestamps = [frame['timestamp'] for frame in frame_detections]
    hands_detected = [frame['hands_detected'] for frame in frame_detections]
    frame_numbers = [frame['frame_number'] for frame in frame_detections]
    
    # Create timeline plot
    fig = go.Figure()
    
    # Add hands detected over time
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=hands_detected,
        mode='markers+lines',
        name='Hands Detected',
        marker=dict(
            size=8,
            color=hands_detected,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Hands")
        ),
        text=[f"Frame {fn}" for fn in frame_numbers],
        hovertemplate="<b>Frame %{text}</b><br>" +
                     "Time: %{x:.1f}s<br>" +
                     "Hands: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title="Hand Detection Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Number of Hands Detected",
        hovermode='closest'
    )
    
    return fig
