"""
File Input Handler for Sign Language Detection
Processes video and image files for gesture analysis
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from PIL import Image
import time

from .hand_detector import HandDetector
from .gesture_extractor import GestureExtractor
from .openai_classifier import SignLanguageClassifier
from .gemini_classifier import GeminiSignLanguageClassifier
from .prediction_logger import PredictionLogger
from .visualization_utils import HandLandmarkVisualizer, create_comparison_view


class FileHandler:
    """
    Handles file input (images and videos) for sign language detection.
    """
    
    def __init__(self, 
                 frame_skip: int = 5,
                 max_frames: int = 100):
        """
        Initialize the FileHandler.
        
        Args:
            frame_skip: Number of frames to skip between processing (for videos)
            max_frames: Maximum number of frames to process from a video
        """
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        
        # Initialize components
        self.hand_detector = HandDetector(static_image_mode=True)
        self.gesture_extractor = GestureExtractor()
        self.classifier = None
        self.visualizer = HandLandmarkVisualizer()
        self.logger = PredictionLogger(debug=True)
        
        # Supported file formats
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    def initialize_classifier(self, api_key: Optional[str] = None, use_gemini: bool = True) -> bool:
        """
        Initialize the AI classifier (Gemini or OpenAI).

        Args:
            api_key: API key (Gemini or OpenAI)
            use_gemini: Whether to use Gemini instead of OpenAI (default: True)

        Returns:
            True if classifier initialized successfully, False otherwise
        """
        # Check environment variable for preference
        use_gemini_env = os.getenv('USE_GEMINI', 'True').lower() == 'true'
        use_gemini = use_gemini and use_gemini_env

        if use_gemini:
            try:
                self.classifier = GeminiSignLanguageClassifier(api_key=api_key)
                print("✅ Gemini AI classifier initialized for file processing")
                return True
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini classifier: {e}")
                print("🔄 Falling back to OpenAI classifier...")

                # Fallback to OpenAI
                try:
                    self.classifier = SignLanguageClassifier(api_key=api_key)
                    print("✅ OpenAI classifier initialized as fallback")
                    return True
                except Exception as e2:
                    print(f"❌ OpenAI classifier also failed: {e2}")
                    print("🔧 Will use pattern-based fallback only")
                    return False
        else:
            try:
                self.classifier = SignLanguageClassifier(api_key=api_key)
                print("✅ OpenAI classifier initialized for file processing")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI classifier: {e}")
                print("🔧 Will use pattern-based fallback only")
                return False
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if the file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file format is supported, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_image_formats or file_ext in self.supported_video_formats
    
    def get_file_type(self, file_path: str) -> str:
        """
        Determine if file is image or video.
        
        Args:
            file_path: Path to the file
            
        Returns:
            'image', 'video', or 'unknown'
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_image_formats:
            return 'image'
        elif file_ext in self.supported_video_formats:
            return 'video'
        else:
            return 'unknown'
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image file for gesture detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing processing results
        """
        if not os.path.exists(image_path):
            return {'success': False, 'error': 'File not found'}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Detect hands
            annotated_image, hand_landmarks = self.hand_detector.detect_hands(image)

            print(f"\n=== Hand Detection Debug ===")
            print(f"Processing image: {os.path.basename(image_path)}")
            print(f"Image shape: {image.shape}")
            print(f"Hands detected: {len(hand_landmarks) if hand_landmarks else 0}")
            if hand_landmarks:
                for i, hand in enumerate(hand_landmarks):
                    print(f"Hand {i+1}: {hand['label']}, confidence: {hand['confidence']:.3f}")
            print("=== End Hand Detection Debug ===\n")

            # Create enhanced visualization
            enhanced_image = self.visualizer.draw_enhanced_landmarks(image, hand_landmarks) if hand_landmarks else annotated_image

            # Create comparison view
            comparison_image = create_comparison_view(image, enhanced_image)

            # Process gestures
            detections = []
            if hand_landmarks:
                for hand_data in hand_landmarks:
                    gesture_description = self.gesture_extractor.create_gesture_description(hand_data)

                    detection = {
                        'hand_label': hand_data['label'],
                        'gesture_description': gesture_description,
                        'confidence': hand_data['confidence'],
                        'bounding_box': self.hand_detector.get_bounding_box(
                            hand_data, image.shape[1], image.shape[0]
                        ),
                        'landmarks_3d': hand_data['landmarks']  # Store for 3D visualization
                    }

                    # Classify gesture if classifier available
                    if self.classifier:
                        print(f"\n=== File Handler Debug ===")
                        print(f"Processing hand: {hand_data['label']}")
                        print(f"Gesture description: {gesture_description}")

                        classification = self.classifier.classify_gesture(gesture_description)
                        detection['classification'] = classification

                        print(f"Classification result: {classification}")
                        print("=== End File Handler Debug ===\n")

                    detections.append(detection)
            
            return {
                'success': True,
                'file_path': image_path,
                'file_type': 'image',
                'image_shape': image.shape,
                'hands_detected': len(hand_landmarks) if hand_landmarks else 0,
                'detections': detections,
                'annotated_image': annotated_image,
                'enhanced_image': enhanced_image,
                'comparison_image': comparison_image,
                'original_image': image
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def process_video(self, video_path: str, 
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process a video file for gesture detection.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results
        """
        if not os.path.exists(video_path):
            return {'success': False, 'error': 'File not found'}
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open video file'}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Process frames
            frame_detections = []
            frame_count = 0
            processed_frames = 0
            
            while cap.isOpened() and processed_frames < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip setting
                if frame_count % (self.frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                timestamp = frame_count / fps if fps > 0 else frame_count
                frame_result = self._process_video_frame(frame, timestamp, frame_count)
                
                if frame_result['hands_detected'] > 0:
                    frame_detections.append(frame_result)
                
                processed_frames += 1
                frame_count += 1
                
                # Progress callback
                if progress_callback:
                    progress = min(processed_frames / self.max_frames, frame_count / total_frames)
                    progress_callback(progress)
            
            cap.release()
            
            # Analyze sequence if detections found
            sequence_analysis = None
            if frame_detections and self.classifier:
                sequence_analysis = self._analyze_video_sequence(frame_detections)
            
            return {
                'success': True,
                'file_path': video_path,
                'file_type': 'video',
                'video_properties': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration,
                    'processed_frames': processed_frames
                },
                'frame_detections': frame_detections,
                'sequence_analysis': sequence_analysis,
                'total_hands_detected': sum(f['hands_detected'] for f in frame_detections)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_video_frame(self, frame: np.ndarray, 
                           timestamp: float, frame_number: int) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame as numpy array
            timestamp: Timestamp in seconds
            frame_number: Frame number
            
        Returns:
            Dictionary containing frame processing results
        """
        # Detect hands
        annotated_frame, hand_landmarks = self.hand_detector.detect_hands(frame)
        
        # Process gestures
        detections = []
        if hand_landmarks:
            for hand_data in hand_landmarks:
                gesture_description = self.gesture_extractor.create_gesture_description(hand_data)
                
                detection = {
                    'hand_label': hand_data['label'],
                    'gesture_description': gesture_description,
                    'confidence': hand_data['confidence']
                }
                
                # Classify gesture if classifier available
                if self.classifier:
                    classification = self.classifier.classify_gesture(gesture_description)
                    detection['classification'] = classification
                
                detections.append(detection)
        
        return {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'hands_detected': len(hand_landmarks) if hand_landmarks else 0,
            'detections': detections
        }
    
    def _analyze_video_sequence(self, frame_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sequence of video frame detections.
        
        Args:
            frame_detections: List of frame detection results
            
        Returns:
            Dictionary containing sequence analysis
        """
        if not self.classifier:
            return {'error': 'Classifier not initialized'}
        
        try:
            # Extract gesture descriptions from frames with detections
            gesture_descriptions = []
            for frame_data in frame_detections:
                for detection in frame_data['detections']:
                    if detection.get('classification', {}).get('success', False):
                        gesture_descriptions.append(detection['gesture_description'])
            
            if len(gesture_descriptions) < 2:
                return {'error': 'Not enough gestures for sequence analysis'}
            
            # Classify sequence
            sequence_result = self.classifier.classify_sequence(gesture_descriptions)
            
            # Add timing information
            sequence_result['start_time'] = frame_detections[0]['timestamp']
            sequence_result['end_time'] = frame_detections[-1]['timestamp']
            sequence_result['duration'] = sequence_result['end_time'] - sequence_result['start_time']
            sequence_result['gesture_count'] = len(gesture_descriptions)
            
            return sequence_result
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_thumbnail(self, file_path: str, size: Tuple[int, int] = (150, 150)) -> Optional[np.ndarray]:
        """
        Create a thumbnail for the given file.

        Args:
            file_path: Path to the file
            size: Thumbnail size (width, height)

        Returns:
            Thumbnail image or None if failed
        """
        try:
            file_type = self.get_file_type(file_path)

            if file_type == 'image':
                image = cv2.imread(file_path)
                if image is not None:
                    thumbnail = cv2.resize(image, size)
                    return thumbnail

            elif file_type == 'video':
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        thumbnail = cv2.resize(frame, size)
                        cap.release()
                        return thumbnail
                    cap.release()

        except Exception as e:
            print(f"Error creating thumbnail for {file_path}: {e}")

        return None

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        metadata = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'file_type': self.get_file_type(file_path),
            'supported': self.is_supported_file(file_path)
        }

        try:
            file_type = metadata['file_type']

            if file_type == 'image':
                image = cv2.imread(file_path)
                if image is not None:
                    metadata.update({
                        'width': image.shape[1],
                        'height': image.shape[0],
                        'channels': image.shape[2] if len(image.shape) > 2 else 1
                    })

            elif file_type == 'video':
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    metadata.update({
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                    })
                    cap.release()

        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def batch_process_files(self, file_paths: List[str],
                          progress_callback: Optional[callable] = None,
                          detailed_progress: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Enhanced batch processing with detailed progress tracking.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for overall progress updates
            detailed_progress: Optional callback for detailed progress updates

        Returns:
            List of processing results for each file
        """
        results = []
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            # Update detailed progress
            if detailed_progress:
                detailed_progress(f"Processing {os.path.basename(file_path)}...", i, total_files)

            if not self.is_supported_file(file_path):
                results.append({
                    'success': False,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'error': 'Unsupported file format'
                })
                continue

            try:
                file_type = self.get_file_type(file_path)

                if file_type == 'image':
                    result = self.process_image(file_path)
                elif file_type == 'video':
                    result = self.process_video(file_path, progress_callback=None)  # Disable nested progress
                else:
                    result = {
                        'success': False,
                        'file_path': file_path,
                        'filename': os.path.basename(file_path),
                        'error': 'Unknown file type'
                    }

                # Add metadata
                if result.get('success'):
                    metadata = self.get_file_metadata(file_path)
                    result.update(metadata)

                results.append(result)

            except Exception as e:
                results.append({
                    'success': False,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'error': str(e)
                })

            # Update overall progress
            if progress_callback:
                progress_callback((i + 1) / total_files)

        return results
    
    def save_annotated_image(self, annotated_image: np.ndarray, 
                           output_path: str) -> bool:
        """
        Save annotated image to file.
        
        Args:
            annotated_image: Annotated image array
            output_path: Path to save the image
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            cv2.imwrite(output_path, annotated_image)
            return True
        except Exception as e:
            print(f"Error saving annotated image: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.hand_detector:
            self.hand_detector.cleanup()
