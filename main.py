#!/usr/bin/env python3
"""
Sign Language Detector - Main Application
Real-time and file-based sign language gesture detection and translation
"""

import argparse
import os
import sys
import cv2
import time
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from src.file_handler import FileHandler
from src.output_handler import OutputHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SignLanguageDetectorApp:
    """
    Main application class for Sign Language Detector.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.file_handler = None
        self.output_handler = None
        self.api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in the .env file or as an environment variable.")
    

    
    def run_file_mode(self, input_path: str, 
                     output_dir: Optional[str] = None,
                     enable_speech: bool = True):
        """
        Run the application in file processing mode.
        
        Args:
            input_path: Path to input file or directory
            output_dir: Directory to save output files
            enable_speech: Whether to enable text-to-speech
        """
        print(f"Processing file(s): {input_path}")
        
        # Initialize components
        self.file_handler = FileHandler()
        self.output_handler = OutputHandler(enable_speech=enable_speech)
        
        # Initialize classifier
        if self.api_key:
            if not self.file_handler.initialize_classifier(self.api_key):
                print("Failed to initialize OpenAI classifier")
                return
        else:
            print("Running without OpenAI classifier (no API key provided)")
        
        # Process files
        if os.path.isfile(input_path):
            # Single file
            self._process_single_file(input_path, output_dir)
        elif os.path.isdir(input_path):
            # Directory
            self._process_directory(input_path, output_dir)
        else:
            print(f"Error: {input_path} is not a valid file or directory")
    
    def _process_single_file(self, file_path: str, output_dir: Optional[str]):
        """Process a single file."""
        if not self.file_handler.is_supported_file(file_path):
            print(f"Error: Unsupported file format: {file_path}")
            return
        
        file_type = self.file_handler.get_file_type(file_path)
        print(f"Processing {file_type}: {file_path}")
        
        # Process file
        if file_type == 'image':
            result = self.file_handler.process_image(file_path)
        else:  # video
            result = self.file_handler.process_video(
                file_path, 
                progress_callback=self._progress_callback
            )
        
        # Display results
        self._display_file_results(result, output_dir)
    
    def _process_directory(self, dir_path: str, output_dir: Optional[str]):
        """Process all supported files in a directory."""
        supported_files = []
        
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) and self.file_handler.is_supported_file(file_path):
                supported_files.append(file_path)
        
        if not supported_files:
            print(f"No supported files found in {dir_path}")
            return
        
        print(f"Found {len(supported_files)} supported files")
        
        # Process files in batch
        results = self.file_handler.batch_process_files(
            supported_files,
            progress_callback=self._progress_callback
        )
        
        # Display results
        for result in results:
            self._display_file_results(result, output_dir)
    
    def _display_file_results(self, result: dict, output_dir: Optional[str]):
        """Display results from file processing."""
        if not result['success']:
            print(f"Error processing {result.get('file_path', 'unknown')}: {result.get('error', 'unknown error')}")
            return
        
        file_path = result['file_path']
        file_type = result['file_type']
        
        print(f"\nResults for {file_path}:")
        print(f"File type: {file_type}")
        
        if file_type == 'image':
            hands_detected = result['hands_detected']
            print(f"Hands detected: {hands_detected}")
            
            for i, detection in enumerate(result['detections']):
                print(f"  Hand {i+1}: {detection['hand_label']}")
                if 'classification' in detection:
                    self.output_handler.display_detection(detection, speak=False)
            
            # Save annotated image if output directory specified
            if output_dir and 'annotated_image' in result:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_annotated{ext}")
                self.file_handler.save_annotated_image(result['annotated_image'], output_path)
                print(f"  Saved annotated image: {output_path}")
        
        else:  # video
            total_hands = result['total_hands_detected']
            processed_frames = result['video_properties']['processed_frames']
            print(f"Total hands detected: {total_hands} in {processed_frames} frames")
            
            # Display sequence analysis if available
            if result.get('sequence_analysis'):
                self.output_handler.display_sequence(result['sequence_analysis'], speak=False)
    

    
    def _progress_callback(self, progress: float):
        """Callback for progress updates."""
        print(f"\rProgress: {progress:.1%}", end='', flush=True)
        if progress >= 1.0:
            print()  # New line when complete
    
    def _cleanup(self):
        """Clean up resources."""
        if self.file_handler:
            self.file_handler.cleanup()

        if self.output_handler:
            self.output_handler.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sign Language Detector - File Processing Mode')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', type=str,
                       help='Output directory for processed files')
    parser.add_argument('--no-speech', action='store_true',
                       help='Disable text-to-speech')

    args = parser.parse_args()

    # Create and run application
    app = SignLanguageDetectorApp()

    enable_speech = not args.no_speech

    app.run_file_mode(
        input_path=args.input,
        output_dir=args.output,
        enable_speech=enable_speech
    )


if __name__ == '__main__':
    main()
