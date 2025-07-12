"""
Export utilities for sign language detection results
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import cv2
import numpy as np
from PIL import Image
import io
import base64


class ResultExporter:
    """
    Export sign language detection results in various formats.
    """
    
    def __init__(self):
        """Initialize the exporter."""
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for PDF reports."""
        custom_styles = {}
        
        # Title style
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Heading style
        custom_styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        # Subheading style
        custom_styles['CustomSubheading'] = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        )
        
        return custom_styles
    
    def export_to_json(self, results: List[Dict[str, Any]], 
                      output_path: str, 
                      include_metadata: bool = True) -> bool:
        """
        Export results to JSON format.
        
        Args:
            results: List of processing results
            output_path: Output file path
            include_metadata: Whether to include metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_files': len(results),
                'successful_files': sum(1 for r in results if r.get('success', False)),
                'results': []
            }
            
            for result in results:
                # Clean result for JSON serialization
                clean_result = self._clean_result_for_export(result)
                
                if not include_metadata:
                    # Remove large data like images
                    clean_result.pop('annotated_image', None)
                    clean_result.pop('enhanced_image', None)
                    clean_result.pop('comparison_image', None)
                    clean_result.pop('original_image', None)
                
                export_data['results'].append(clean_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def export_to_csv(self, results: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Export results to CSV format.
        
        Args:
            results: List of processing results
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            csv_data = []
            
            for result in results:
                if not result.get('success'):
                    csv_data.append({
                        'filename': result.get('filename', ''),
                        'file_type': result.get('file_type', ''),
                        'success': False,
                        'error': result.get('error', ''),
                        'hands_detected': 0,
                        'hand_label': '',
                        'confidence': 0,
                        'letter': '',
                        'word': '',
                        'ai_confidence': 0
                    })
                    continue
                
                if result.get('detections'):
                    for detection in result['detections']:
                        row = {
                            'filename': result.get('filename', ''),
                            'file_type': result.get('file_type', ''),
                            'success': True,
                            'error': '',
                            'hands_detected': result.get('hands_detected', 0),
                            'hand_label': detection.get('hand_label', ''),
                            'confidence': detection.get('confidence', 0),
                            'gesture_description': detection.get('gesture_description', '')
                        }
                        
                        # Add classification data if available
                        if 'classification' in detection and detection['classification'].get('success'):
                            classification = detection['classification']
                            row.update({
                                'letter': classification.get('letter', ''),
                                'word': classification.get('word', ''),
                                'ai_confidence': classification.get('confidence', 0)
                            })
                        else:
                            row.update({
                                'letter': '',
                                'word': '',
                                'ai_confidence': 0
                            })
                        
                        csv_data.append(row)
                else:
                    # No detections
                    csv_data.append({
                        'filename': result.get('filename', ''),
                        'file_type': result.get('file_type', ''),
                        'success': True,
                        'error': '',
                        'hands_detected': 0,
                        'hand_label': '',
                        'confidence': 0,
                        'letter': '',
                        'word': '',
                        'ai_confidence': 0
                    })
            
            # Write to CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(output_path, index=False)
                return True
            
            return False
        
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_pdf(self, results: List[Dict[str, Any]], 
                     output_path: str,
                     include_images: bool = True) -> bool:
        """
        Export results to PDF report.
        
        Args:
            results: List of processing results
            output_path: Output file path
            include_images: Whether to include images in the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title
            title = Paragraph("Sign Language Detection Report", self.custom_styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Summary
            successful_files = sum(1 for r in results if r.get('success', False))
            total_hands = sum(r.get('hands_detected', 0) for r in results if r.get('success', False))
            
            summary_text = f"""
            <b>Processing Summary</b><br/>
            Total Files: {len(results)}<br/>
            Successful: {successful_files}<br/>
            Total Hands Detected: {total_hands}<br/>
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            summary = Paragraph(summary_text, self.styles['Normal'])
            story.append(summary)
            story.append(Spacer(1, 20))
            
            # Results for each file
            for i, result in enumerate(results):
                # File header
                filename = result.get('filename', f'File {i+1}')
                header = Paragraph(f"File: {filename}", self.custom_styles['CustomHeading'])
                story.append(header)
                
                if not result.get('success'):
                    error_text = f"<font color='red'>Error: {result.get('error', 'Unknown error')}</font>"
                    error_para = Paragraph(error_text, self.styles['Normal'])
                    story.append(error_para)
                    story.append(Spacer(1, 10))
                    continue
                
                # File info
                file_info = [
                    ['Property', 'Value'],
                    ['File Type', result.get('file_type', 'Unknown')],
                    ['File Size', f"{result.get('file_size', 0) / 1024:.1f} KB"],
                    ['Hands Detected', str(result.get('hands_detected', 0))]
                ]
                
                if result.get('file_type') == 'video':
                    video_props = result.get('video_properties', {})
                    file_info.extend([
                        ['Duration', f"{video_props.get('duration', 0):.1f}s"],
                        ['FPS', f"{video_props.get('fps', 0):.1f}"],
                        ['Total Frames', str(video_props.get('total_frames', 0))]
                    ])
                
                info_table = Table(file_info)
                info_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(info_table)
                story.append(Spacer(1, 15))
                
                # Detection details
                if result.get('detections'):
                    detections_header = Paragraph("Detection Details", self.custom_styles['CustomSubheading'])
                    story.append(detections_header)
                    
                    for j, detection in enumerate(result['detections']):
                        detection_text = f"""
                        <b>Hand {j+1}: {detection.get('hand_label', 'Unknown')}</b><br/>
                        Confidence: {detection.get('confidence', 0):.1%}<br/>
                        """
                        
                        if 'classification' in detection and detection['classification'].get('success'):
                            classification = detection['classification']
                            if classification.get('letter'):
                                detection_text += f"Letter: <b>{classification['letter']}</b><br/>"
                            if classification.get('word'):
                                detection_text += f"Word: <b>{classification['word']}</b><br/>"
                            if classification.get('confidence'):
                                detection_text += f"AI Confidence: {classification['confidence']:.1%}<br/>"
                        
                        detection_para = Paragraph(detection_text, self.styles['Normal'])
                        story.append(detection_para)
                        story.append(Spacer(1, 10))
                
                story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            return True
        
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False
    
    def _clean_result_for_export(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean result dictionary for export by converting numpy arrays to lists.
        
        Args:
            result: Result dictionary
            
        Returns:
            Cleaned result dictionary
        """
        clean_result = {}
        
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to base64 encoded strings for images
                if key in ['annotated_image', 'enhanced_image', 'comparison_image', 'original_image']:
                    try:
                        # Convert to PIL Image and then to base64
                        if len(value.shape) == 3:
                            # Convert BGR to RGB for proper color representation
                            value_rgb = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(value_rgb)
                        else:
                            pil_image = Image.fromarray(value)
                        
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='PNG')
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        clean_result[key] = f"data:image/png;base64,{img_str}"
                    except:
                        clean_result[key] = None
                else:
                    clean_result[key] = value.tolist()
            elif isinstance(value, (list, dict)):
                clean_result[key] = value
            else:
                clean_result[key] = value
        
        return clean_result
    
    def create_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary report of the processing results.
        
        Args:
            results: List of processing results
            
        Returns:
            Summary report dictionary
        """
        summary = {
            'total_files': len(results),
            'successful_files': 0,
            'failed_files': 0,
            'total_hands_detected': 0,
            'file_types': {},
            'detected_letters': {},
            'detected_words': {},
            'average_confidence': 0,
            'processing_errors': []
        }
        
        confidences = []
        
        for result in results:
            if result.get('success'):
                summary['successful_files'] += 1
                summary['total_hands_detected'] += result.get('hands_detected', 0)
                
                # File type statistics
                file_type = result.get('file_type', 'unknown')
                summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1
                
                # Process detections
                for detection in result.get('detections', []):
                    if 'confidence' in detection:
                        confidences.append(detection['confidence'])
                    
                    if 'classification' in detection and detection['classification'].get('success'):
                        classification = detection['classification']
                        
                        if classification.get('letter'):
                            letter = classification['letter']
                            summary['detected_letters'][letter] = summary['detected_letters'].get(letter, 0) + 1
                        
                        if classification.get('word'):
                            word = classification['word']
                            summary['detected_words'][word] = summary['detected_words'].get(word, 0) + 1
            else:
                summary['failed_files'] += 1
                summary['processing_errors'].append({
                    'filename': result.get('filename', 'unknown'),
                    'error': result.get('error', 'unknown error')
                })
        
        # Calculate average confidence
        if confidences:
            summary['average_confidence'] = sum(confidences) / len(confidences)
        
        return summary
