"""
Enhanced Streamlit GUI for Sign Language Detector
Modern, Professional File Processing Interface
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
import threading
from PIL import Image
import tempfile
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import base64
from io import BytesIO
import json

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from src.file_handler import FileHandler
from src.output_handler import OutputHandler
from src.hand_detector import HandDetector
from src.gesture_extractor import GestureExtractor
from src.openai_classifier import SignLanguageClassifier
from src.visualization_utils import HandLandmarkVisualizer, create_processing_timeline
from src.export_utils import ResultExporter


# Page configuration
st.set_page_config(
    page_title="Sign Language Detector Pro",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive CSS for optimal text visibility and professional design
st.markdown("""
<style>
    /* Enhanced theme colors with WCAG AA compliant contrast ratios */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --text-light: #FFFFFF;
        --text-dark: #1A1A1A;
        --text-medium: #495057;
        --text-muted: #6C757D;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --error-color: #E74C3C;
        --info-color: #17A2B8;
        --border-color: #E1E5E9;
        --card-background: #FFFFFF;
        --sidebar-background: #F8F9FA;
        --hover-background: #E9ECEF;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global text color improvements - Foundation */
    .stApp {
        color: var(--text-dark) !important;
        background-color: var(--background-color) !important;
    }

    /* All headings - Comprehensive coverage */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }

    /* All paragraph text */
    p {
        color: var(--text-color) !important;
    }

    /* All span elements */
    span {
        color: var(--text-dark) !important;
    }

    /* All div text content */
    div {
        color: var(--text-dark) !important;
    }

    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: var(--text-light);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text-light) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
        color: var(--text-light) !important;
    }

    /* File upload area with improved text visibility */
    .upload-area {
        border: 3px dashed var(--primary-color);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: var(--card-background);
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: var(--text-dark) !important;
    }

    .upload-area h3 {
        color: var(--text-dark) !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .upload-area p {
        color: var(--text-color) !important;
        margin: 0.5rem 0;
    }

    .upload-area:hover {
        border-color: var(--accent-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* Result cards with improved text contrast */
    .result-card {
        background: var(--card-background);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary-color);
        transition: all 0.3s ease;
        color: var(--text-dark) !important;
    }

    .result-card h3 {
        color: var(--text-dark) !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-card p {
        color: var(--text-color) !important;
    }

    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }

    /* Metrics styling with improved text visibility */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: var(--text-light) !important;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: var(--text-light) !important;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        color: var(--text-light) !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 10px;
    }

    /* Comprehensive Button styling - All states covered */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: var(--text-light) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
        color: var(--text-light) !important;
        background: linear-gradient(135deg, #3A9BC1, #B8457A) !important;
    }

    .stButton > button:focus {
        color: var(--text-light) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
        outline: 2px solid var(--accent-color) !important;
        outline-offset: 2px !important;
    }

    .stButton > button:active {
        color: var(--text-light) !important;
        transform: translateY(0px) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    }

    /* Download button specific styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color), #2ECC71) !important;
        color: var(--text-light) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
    }

    .stDownloadButton > button:hover {
        color: var(--text-light) !important;
        background: linear-gradient(135deg, #2ECC71, #27AE60) !important;
    }

    /* Comprehensive Sidebar styling - All elements covered */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, .css-1y4p8pa {
        background: var(--sidebar-background) !important;
        color: var(--text-dark) !important;
    }

    /* Sidebar text - All variations */
    .css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown, .css-17eq0hr .stMarkdown {
        color: var(--text-dark) !important;
    }

    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }

    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3, .css-1lcbmhc h4, .css-1lcbmhc h5, .css-1lcbmhc h6 {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }

    /* Sidebar labels and text */
    .css-1d391kg label, .css-1lcbmhc label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }

    .css-1d391kg p, .css-1lcbmhc p {
        color: var(--text-color) !important;
    }

    .css-1d391kg span, .css-1lcbmhc span {
        color: var(--text-dark) !important;
    }

    /* Sidebar widget labels */
    .css-1d391kg .stSelectbox label, .css-1d391kg .stSlider label, .css-1d391kg .stCheckbox label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }

    /* Success/Error messages with proper contrast */
    .success-message {
        background: var(--success-color) !important;
        color: var(--text-light) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }

    .error-message {
        background: var(--error-color) !important;
        color: var(--text-light) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }

    /* Streamlit native message styling improvements */
    .stAlert {
        color: var(--text-dark) !important;
    }

    .stSuccess {
        background-color: rgba(39, 174, 96, 0.1) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--success-color) !important;
    }

    .stError {
        background-color: rgba(231, 76, 60, 0.1) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--error-color) !important;
    }

    .stWarning {
        background-color: rgba(243, 156, 18, 0.1) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--warning-color) !important;
    }

    .stInfo {
        background-color: rgba(46, 134, 171, 0.1) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--primary-color) !important;
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Comprehensive Form and Input styling - All form elements */
    .stTextInput > div > div > input {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.7 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.2) !important;
        color: var(--text-dark) !important;
    }

    /* Text area styling */
    .stTextArea > div > div > textarea {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.7 !important;
    }

    /* Select box styling */
    .stSelectbox > div > div > div {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div > div > div {
        color: var(--text-dark) !important;
    }

    /* Multi-select styling */
    .stMultiSelect > div > div > div {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
    }

    /* Number input styling */
    .stNumberInput > div > div > input {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        color: var(--text-dark) !important;
    }

    .stSlider > div > div > div > div {
        color: var(--text-dark) !important;
    }

    /* Checkbox and radio styling */
    .stCheckbox > label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }

    .stRadio > label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }

    /* Form labels - comprehensive coverage */
    label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Comprehensive Tab styling - All states and variations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 2px solid var(--border-color) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        margin-bottom: -2px !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--hover-background) !important;
        color: var(--text-dark) !important;
        border-color: var(--primary-color) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: var(--text-light) !important;
        border-color: var(--primary-color) !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2) !important;
    }

    /* Tab content styling */
    .stTabs [data-baseweb="tab-panel"] {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        padding: 1.5rem !important;
        border-radius: 0 8px 8px 8px !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
    }

    /* Comprehensive Expander styling */
    .streamlit-expanderHeader {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: var(--hover-background) !important;
        color: var(--text-dark) !important;
    }

    .streamlit-expanderContent {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1.5rem !important;
    }

    /* Comprehensive Metric styling - All metric components */
    .metric-container {
        background-color: var(--card-background) !important;
        color: var(--text-dark) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Streamlit native metrics */
    .css-1xarl3l {
        color: var(--text-dark) !important;
    }

    .css-1xarl3l > div {
        color: var(--text-dark) !important;
    }

    /* Metric values and labels */
    [data-testid="metric-container"] {
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    [data-testid="metric-container"] > div {
        color: var(--text-dark) !important;
    }

    [data-testid="metric-container"] label {
        color: var(--text-medium) !important;
        font-weight: 500 !important;
    }

    /* Progress indicators and loading text */
    .stProgress > div > div > div {
        color: var(--text-dark) !important;
    }

    .stSpinner > div {
        color: var(--text-dark) !important;
    }

    /* File uploader styling */
    .stFileUploader > div > div > div {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
    }

    .stFileUploader > div > div > div:hover {
        border-color: var(--primary-color) !important;
    }

    .stFileUploader label {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }

    /* Data frame and table styling */
    .stDataFrame {
        color: var(--text-dark) !important;
    }

    .stDataFrame table {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
    }

    .stDataFrame th {
        color: var(--text-dark) !important;
        background-color: var(--hover-background) !important;
        font-weight: 600 !important;
    }

    .stDataFrame td {
        color: var(--text-dark) !important;
    }

    /* Code blocks and preformatted text */
    .stCode {
        color: var(--text-dark) !important;
        background-color: var(--hover-background) !important;
    }

    code {
        color: var(--text-dark) !important;
        background-color: var(--hover-background) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }

    pre {
        color: var(--text-dark) !important;
        background-color: var(--hover-background) !important;
    }

    /* JSON and data display */
    .stJson {
        color: var(--text-dark) !important;
        background-color: var(--card-background) !important;
    }

    /* Caption and help text */
    .caption {
        color: var(--text-muted) !important;
        font-size: 0.9rem !important;
    }

    .help {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
    }

    /* Tooltip styling */
    .stTooltipIcon {
        color: var(--text-medium) !important;
    }

    /* Link styling */
    a {
        color: var(--primary-color) !important;
        text-decoration: none !important;
    }

    a:hover {
        color: var(--secondary-color) !important;
        text-decoration: underline !important;
    }

    /* Status indicators */
    .status-success {
        color: var(--success-color) !important;
        font-weight: 600 !important;
    }

    .status-error {
        color: var(--error-color) !important;
        font-weight: 600 !important;
    }

    .status-warning {
        color: var(--warning-color) !important;
        font-weight: 600 !important;
    }

    .status-info {
        color: var(--info-color) !important;
        font-weight: 600 !important;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
            color: var(--text-light) !important;
        }
        .main-header p {
            font-size: 1rem !important;
            color: var(--text-light) !important;
        }
        .upload-area {
            padding: 2rem !important;
        }

        /* Mobile text adjustments */
        h1, h2, h3, h4, h5, h6 {
            font-size: calc(1rem + 0.5vw) !important;
        }

        p, span, div {
            font-size: 0.9rem !important;
        }

        label {
            font-size: 0.9rem !important;
        }
    }

    /* High contrast mode support */
    @media (prefers-contrast: high) {
        :root {
            --text-dark: #000000;
            --text-light: #FFFFFF;
            --border-color: #000000;
        }
    }

    /* Dark mode support (if needed) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1E1E1E !important;
        }

        :root {
            --background-color: #1E1E1E;
            --card-background: #2D2D2D;
            --text-dark: #FFFFFF;
            --text-color: #E0E0E0;
            --border-color: #404040;
            --hover-background: #404040;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'file_handler' not in st.session_state:
    st.session_state.file_handler = None
if 'output_handler' not in st.session_state:
    st.session_state.output_handler = None
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'transcript' not in st.session_state:
    st.session_state.transcript = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = HandLandmarkVisualizer()
if 'exporter' not in st.session_state:
    st.session_state.exporter = ResultExporter()


def initialize_components():
    """Initialize the application components."""
    if st.session_state.file_handler is None:
        st.session_state.file_handler = FileHandler()

    if st.session_state.output_handler is None:
        st.session_state.output_handler = OutputHandler(
            enable_speech=False,  # Disable speech in web interface
            save_transcript=False  # Handle transcript in session state
        )

def create_header():
    """Create the main header with modern styling."""
    st.markdown("""
    <div class="main-header">
        <h1>🤟 Sign Language Detector Pro</h1>
        <p>Advanced AI-Powered Gesture Recognition & Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_file_upload_area():
    """Create an enhanced file upload area with drag-and-drop styling."""
    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #2C3E50 !important; font-weight: 600; margin-bottom: 1rem;">📁 Upload Your Files</h3>
        <p style="color: #2C3E50 !important; font-size: 1.1rem; margin-bottom: 0.5rem;">Drag and drop your images or videos here, or click to browse</p>
        <p style="color: #666666 !important; font-size: 0.9rem; margin: 0;"><small>Supported formats: JPG, PNG, BMP, MP4, AVI, MOV, MKV</small></p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_dashboard(results: List[Dict[str, Any]]):
    """Create a metrics dashboard showing processing statistics."""
    if not results:
        return

    # Calculate metrics
    total_files = len(results)
    successful_files = sum(1 for r in results if r.get('success', False))
    total_hands = sum(r.get('hands_detected', 0) for r in results if r.get('success', False))
    avg_confidence = 0

    if successful_files > 0:
        confidences = []
        for result in results:
            if result.get('success') and result.get('detections'):
                for detection in result['detections']:
                    if 'confidence' in detection:
                        confidences.append(detection['confidence'])
        avg_confidence = np.mean(confidences) if confidences else 0

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FFFFFF !important; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{total_files}</div>
            <div class="metric-label" style="color: #FFFFFF !important; font-size: 1rem; opacity: 0.9;">Files Processed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FFFFFF !important; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{successful_files}</div>
            <div class="metric-label" style="color: #FFFFFF !important; font-size: 1rem; opacity: 0.9;">Successful</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FFFFFF !important; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{total_hands}</div>
            <div class="metric-label" style="color: #FFFFFF !important; font-size: 1rem; opacity: 0.9;">Hands Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FFFFFF !important; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{avg_confidence:.1%}</div>
            <div class="metric-label" style="color: #FFFFFF !important; font-size: 1rem; opacity: 0.9;">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)

def create_confidence_chart(results: List[Dict[str, Any]], chart_key: str = "confidence_chart"):
    """Create a confidence score visualization."""
    confidences = []
    file_names = []

    for result in results:
        if result.get('success') and result.get('detections'):
            for i, detection in enumerate(result['detections']):
                if 'confidence' in detection:
                    confidences.append(detection['confidence'])
                    file_name = os.path.basename(result.get('file_path', 'Unknown'))
                    file_names.append(f"{file_name} - Hand {i+1}")

    if confidences:
        df = pd.DataFrame({
            'File': file_names,
            'Confidence': confidences
        })

        fig = px.bar(df, x='File', y='Confidence',
                    title='Hand Detection Confidence Scores',
                    color='Confidence',
                    color_continuous_scale='Viridis')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

def create_gesture_analysis_chart(results: List[Dict[str, Any]], chart_key: str = "gesture_analysis_chart"):
    """Create gesture analysis visualization."""
    gesture_data = []

    for result in results:
        if result.get('success') and result.get('detections'):
            for detection in result['detections']:
                if 'classification' in detection and detection['classification'].get('success'):
                    classification = detection['classification']
                    gesture_data.append({
                        'File': os.path.basename(result.get('file_path', 'Unknown')),
                        'Hand': detection.get('hand_label', 'Unknown'),
                        'Letter': classification.get('letter', 'N/A'),
                        'Word': classification.get('word', 'N/A'),
                        'Confidence': classification.get('confidence', 0)
                    })

    if gesture_data:
        df = pd.DataFrame(gesture_data)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Letters Detected', 'Classification Confidence'),
            specs=[[{"type": "pie"}, {"type": "histogram"}]]
        )

        # Letter distribution pie chart
        letter_counts = df['Letter'].value_counts()
        fig.add_trace(
            go.Pie(labels=letter_counts.index, values=letter_counts.values, name="Letters"),
            row=1, col=1
        )

        # Confidence histogram
        fig.add_trace(
            go.Histogram(x=df['Confidence'], name="Confidence", nbinsx=10),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


def setup_ai_api():
    """Setup AI API key with automatic Gemini configuration."""
    st.sidebar.markdown("### 🔑 AI API Configuration")

    # Use Gemini by default
    default_gemini_key = "AIzaSyDd2BfvfgnVQFkGufpuD76QOsaPM3hWgxo"

    # AI provider selection
    ai_provider = st.sidebar.selectbox(
        "AI Provider",
        ["Google Gemini (Recommended)", "OpenAI GPT"],
        index=0,
        help="Choose your AI provider for sign language classification"
    )

    use_gemini = "Gemini" in ai_provider

    # Check if user wants to use a custom API key
    use_custom_key = st.sidebar.checkbox("Use Custom API Key", value=False)

    if use_custom_key:
        if use_gemini:
            api_key = st.sidebar.text_input(
                "Custom Gemini API Key",
                type="password",
                help="Enter your custom Google Gemini API key",
                placeholder="AIza..."
            )
            env_key = 'GEMINI_API_KEY'
        else:
            api_key = st.sidebar.text_input(
                "Custom OpenAI API Key",
                type="password",
                help="Enter your custom OpenAI API key",
                placeholder="sk-..."
            )
            env_key = 'OPENAI_API_KEY'

        if api_key:
            os.environ[env_key] = api_key
            st.sidebar.success(f"✅ Custom {ai_provider.split()[0]} API key configured")
            return api_key, use_gemini
        else:
            st.sidebar.warning("⚠️ Please enter your custom API key")
            return None, use_gemini
    else:
        # Use default keys
        if use_gemini:
            os.environ['GEMINI_API_KEY'] = default_gemini_key
            os.environ['USE_GEMINI'] = 'True'
            st.sidebar.success("✅ Gemini API configured automatically")
            st.sidebar.info("🚀 Using Google Gemini for fast, accurate predictions")
            return default_gemini_key, True
        else:
            # OpenAI fallback (will likely fail due to quota)
            st.sidebar.warning("⚠️ OpenAI quota may be exceeded")
            st.sidebar.info("💡 Recommend using Gemini for better reliability")
            return None, False

def create_settings_panel():
    """Create an advanced settings panel."""
    st.sidebar.markdown("### ⚙️ Processing Settings")

    # Detection confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimum confidence for hand detection"
    )

    # Maximum hands to detect
    max_hands = st.sidebar.selectbox(
        "Maximum Hands to Detect",
        options=[1, 2, 3, 4],
        index=1,
        help="Maximum number of hands to detect per image"
    )

    # Video frame sampling
    frame_skip = st.sidebar.slider(
        "Video Frame Sampling",
        min_value=1,
        max_value=30,
        value=5,
        help="Process every Nth frame in videos (higher = faster processing)"
    )

    # Export options
    st.sidebar.markdown("### 📊 Export Options")
    export_format = st.sidebar.selectbox(
        "Export Format",
        options=["JSON", "CSV", "PDF Report"],
        help="Choose format for exporting results"
    )

    return {
        'confidence_threshold': confidence_threshold,
        'max_hands': max_hands,
        'frame_skip': frame_skip,
        'export_format': export_format
    }

def process_uploaded_files(uploaded_files: List, api_key: str, settings: Dict[str, Any], use_gemini: bool = True):
    """Process multiple uploaded files with progress tracking."""
    if not uploaded_files:
        return []

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize file handler with settings
    file_handler = FileHandler(
        frame_skip=settings['frame_skip'],
        max_frames=100
    )

    if api_key:
        file_handler.initialize_classifier(api_key, use_gemini=use_gemini)

    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Determine file type and process
            file_type = file_handler.get_file_type(tmp_path)

            if file_type == 'image':
                result = file_handler.process_image(tmp_path)
            elif file_type == 'video':
                result = file_handler.process_video(tmp_path)
            else:
                result = {'success': False, 'error': 'Unsupported file format'}

            # Add filename to result
            result['filename'] = uploaded_file.name
            result['file_size'] = len(uploaded_file.getvalue())
            results.append(result)

        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'filename': uploaded_file.name,
                'file_size': len(uploaded_file.getvalue())
            })

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    progress_bar.empty()
    status_text.empty()

    return results


def create_image_with_landmarks(image_array: np.ndarray, hand_landmarks: List[Dict[str, Any]]) -> Image.Image:
    """Create an image with hand landmarks overlaid."""
    # Convert to PIL Image for display
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # BGR to RGB conversion
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array

    return Image.fromarray(image_rgb)

def display_image_results(result: Dict[str, Any]):
    """Display results for image processing with enhanced UI."""
    if not result['success']:
        st.error(f"❌ Error processing {result.get('filename', 'file')}: {result.get('error', 'Unknown error')}")
        return

    filename = result.get('filename', 'Unknown')
    file_size = result.get('file_size', 0)

    # Create result card
    st.markdown(f"""
    <div class="result-card">
        <h3 style="color: #2C3E50 !important; font-weight: 600; margin-bottom: 1rem;">📸 {filename}</h3>
        <p style="color: #2C3E50 !important;"><strong>File Size:</strong> {file_size / 1024:.1f} KB | <strong>Hands Detected:</strong> {result['hands_detected']}</p>
    </div>
    """, unsafe_allow_html=True)

    if result['hands_detected'] > 0:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🖼️ Processed Images")

            # Create tabs for different views
            img_tab1, img_tab2, img_tab3 = st.tabs(["🔍 Enhanced", "📊 Comparison", "🎯 3D View"])

            with img_tab1:
                if 'enhanced_image' in result:
                    enhanced_img = create_image_with_landmarks(result['enhanced_image'], [])
                    st.image(enhanced_img, caption="Enhanced Hand Landmarks", use_container_width=True)
                elif 'annotated_image' in result:
                    annotated_img = create_image_with_landmarks(result['annotated_image'], [])
                    st.image(annotated_img, caption="Hand Landmarks Detected", use_container_width=True)

            with img_tab2:
                if 'comparison_image' in result:
                    comparison_img = create_image_with_landmarks(result['comparison_image'], [])
                    st.image(comparison_img, caption="Before vs After Comparison", use_container_width=True)

            with img_tab3:
                # 3D visualization for first detected hand
                if result['detections'] and 'landmarks_3d' in result['detections'][0]:
                    hand_data = {
                        'label': result['detections'][0]['hand_label'],
                        'landmarks': result['detections'][0]['landmarks_3d']
                    }

                    visualizer = st.session_state.visualizer
                    fig_3d = visualizer.create_3d_hand_plot(hand_data)
                    st.plotly_chart(fig_3d, use_container_width=True, key="3d_hand_plot")
                else:
                    st.info("3D visualization requires hand landmark data")

        with col2:
            st.subheader("🔍 Detection Details")

            for i, detection in enumerate(result['detections']):
                with st.expander(f"✋ Hand {i+1}: {detection['hand_label']}", expanded=True):
                    # Confidence meter
                    confidence = detection['confidence']
                    st.metric("Detection Confidence", f"{confidence:.1%}")

                    # Progress bar for confidence
                    st.progress(confidence)

                    # Gesture description
                    st.text_area(
                        "Gesture Description",
                        detection['gesture_description'],
                        height=100,
                        disabled=True
                    )

                    # Classification results
                    if 'classification' in detection and detection['classification']['success']:
                        classification = detection['classification']

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if classification.get('letter'):
                                st.success(f"🔤 **Letter:** {classification['letter']}")
                        with col_b:
                            if classification.get('word'):
                                st.success(f"📝 **Word:** {classification['word']}")

                        if classification.get('confidence'):
                            st.info(f"🎯 **AI Confidence:** {classification['confidence']:.1%}")

def display_video_results(result: Dict[str, Any]):
    """Display results for video processing with enhanced UI."""
    if not result['success']:
        st.error(f"❌ Error processing {result.get('filename', 'file')}: {result.get('error', 'Unknown error')}")
        return

    filename = result.get('filename', 'Unknown')
    file_size = result.get('file_size', 0)
    video_props = result['video_properties']

    # Create result card
    st.markdown(f"""
    <div class="result-card">
        <h3 style="color: #2C3E50 !important; font-weight: 600; margin-bottom: 1rem;">🎥 {filename}</h3>
        <p style="color: #2C3E50 !important;"><strong>File Size:</strong> {file_size / (1024*1024):.1f} MB |
           <strong>Duration:</strong> {video_props['duration']:.1f}s |
           <strong>Total Hands:</strong> {result['total_hands_detected']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Video metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Frames", video_props['total_frames'])
    with col2:
        st.metric("Processed Frames", video_props['processed_frames'])
    with col3:
        st.metric("FPS", f"{video_props['fps']:.1f}")
    with col4:
        st.metric("Hands Found", result['total_hands_detected'])

    # Frame-by-frame analysis
    if result['frame_detections']:
        st.subheader("📊 Frame-by-Frame Analysis")

        # Enhanced timeline visualization
        timeline_fig = create_processing_timeline(result['frame_detections'])
        st.plotly_chart(timeline_fig, use_container_width=True, key="video_timeline")

        # Additional analysis charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Confidence over time
            confidence_data = []
            for frame in result['frame_detections']:
                for detection in frame['detections']:
                    if 'confidence' in detection:
                        confidence_data.append({
                            'Timestamp': frame['timestamp'],
                            'Confidence': detection['confidence'],
                            'Hand': detection['hand_label']
                        })

            if confidence_data:
                conf_df = pd.DataFrame(confidence_data)
                fig_conf = px.scatter(conf_df, x='Timestamp', y='Confidence',
                                    color='Hand', title='Detection Confidence Over Time')
                st.plotly_chart(fig_conf, use_container_width=True, key="confidence_over_time")

        with col_chart2:
            # Hand distribution
            hand_counts = {}
            for frame in result['frame_detections']:
                for detection in frame['detections']:
                    hand_label = detection.get('hand_label', 'Unknown')
                    hand_counts[hand_label] = hand_counts.get(hand_label, 0) + 1

            if hand_counts:
                fig_pie = px.pie(values=list(hand_counts.values()),
                               names=list(hand_counts.keys()),
                               title='Hand Distribution')
                st.plotly_chart(fig_pie, use_container_width=True, key="hand_distribution")

        # Detailed frame results
        st.subheader("🔍 Detailed Frame Results")

        # Show first 10 frames with detections
        frames_to_show = [f for f in result['frame_detections'] if f['hands_detected'] > 0][:10]

        for frame_data in frames_to_show:
            with st.expander(f"⏱️ Frame {frame_data['frame_number']} (t={frame_data['timestamp']:.1f}s)"):
                for i, detection in enumerate(frame_data['detections']):
                    st.write(f"**✋ {detection['hand_label']} Hand {i+1}**")

                    if 'classification' in detection and detection['classification']['success']:
                        classification = detection['classification']

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if classification.get('letter'):
                                st.info(f"Letter: **{classification['letter']}**")
                        with col_b:
                            if classification.get('word'):
                                st.info(f"Word: **{classification['word']}**")
                        with col_c:
                            if classification.get('confidence'):
                                st.info(f"Confidence: **{classification['confidence']:.1%}**")

    # Sequence analysis
    if result.get('sequence_analysis') and result['sequence_analysis'].get('success'):
        st.subheader("🔗 Sequence Analysis")
        sequence = result['sequence_analysis']

        col1, col2 = st.columns(2)
        with col1:
            if sequence.get('word'):
                st.success(f"🎯 **Detected Word:** {sequence['word']}")
            if sequence.get('sentence'):
                st.success(f"📝 **Detected Sentence:** {sequence['sentence']}")

        with col2:
            if sequence.get('individual_letters'):
                letters_str = ' → '.join(sequence['individual_letters'])
                st.info(f"🔤 **Letter Sequence:** {letters_str}")

            if sequence.get('confidence'):
                st.metric("Sequence Confidence", f"{sequence['confidence']:.1%}")

def export_results(results: List[Dict[str, Any]], format_type: str):
    """Enhanced export functionality with multiple formats."""
    if not results:
        st.warning("No results to export")
        return

    exporter = st.session_state.exporter
    timestamp = int(time.time())

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export JSON", use_container_width=True):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                if exporter.export_to_json(results, tmp_file.name, include_metadata=True):
                    with open(tmp_file.name, 'r') as f:
                        json_data = f.read()

                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"sign_language_results_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    os.unlink(tmp_file.name)
                else:
                    st.error("Failed to export JSON")

    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                if exporter.export_to_csv(results, tmp_file.name):
                    with open(tmp_file.name, 'r') as f:
                        csv_data = f.read()

                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"sign_language_results_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    os.unlink(tmp_file.name)
                else:
                    st.error("Failed to export CSV")

    with col3:
        if st.button("📋 Export PDF Report", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if exporter.export_to_pdf(results, tmp_file.name, include_images=False):
                    with open(tmp_file.name, 'rb') as f:
                        pdf_data = f.read()

                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=f"sign_language_report_{timestamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    os.unlink(tmp_file.name)
                else:
                    st.error("Failed to export PDF")

    # Summary report
    if st.button("📈 Generate Summary Report", use_container_width=True):
        summary = exporter.create_summary_report(results)

        st.markdown("### 📊 Processing Summary")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Files", summary['total_files'])
        with col_b:
            st.metric("Successful", summary['successful_files'])
        with col_c:
            st.metric("Failed", summary['failed_files'])
        with col_d:
            st.metric("Hands Detected", summary['total_hands_detected'])

        if summary['detected_letters']:
            st.markdown("#### 🔤 Most Common Letters")
            letters_df = pd.DataFrame(list(summary['detected_letters'].items()),
                                    columns=['Letter', 'Count'])
            letters_df = letters_df.sort_values('Count', ascending=False)

            fig = px.bar(letters_df.head(10), x='Letter', y='Count',
                        title='Top 10 Detected Letters')
            st.plotly_chart(fig, use_container_width=True, key="top_letters_chart")

        if summary['detected_words']:
            st.markdown("#### 📝 Most Common Words")
            words_df = pd.DataFrame(list(summary['detected_words'].items()),
                                  columns=['Word', 'Count'])
            words_df = words_df.sort_values('Count', ascending=False)

            fig = px.bar(words_df.head(10), x='Word', y='Count',
                        title='Top 10 Detected Words')
            st.plotly_chart(fig, use_container_width=True, key="top_words_chart")


def get_single_prediction(result: Dict[str, Any]) -> str:
    """
    Extract a single, clear prediction from the result.

    Args:
        result: Processing result dictionary

    Returns:
        Single prediction string (letter, word, or "No prediction")
    """
    if not result.get('success') or not result.get('detections'):
        return "No prediction"

    # Collect all predictions from all detected hands
    letters = []
    words = []

    for detection in result['detections']:
        if 'classification' in detection and detection['classification'].get('success'):
            classification = detection['classification']

            # Get letter prediction
            if classification.get('letter') and classification['letter'] != 'N/A':
                letters.append(classification['letter'])

            # Get word prediction
            if classification.get('word') and classification['word'] != 'N/A':
                words.append(classification['word'])

    # Priority: Word > Letter > No prediction
    if words:
        # Return the most confident word or the first word if multiple
        return words[0].upper()
    elif letters:
        # Return the most confident letter or the first letter if multiple
        return letters[0].upper()
    else:
        return "No prediction"

def display_single_prediction_card(result: Dict[str, Any]):
    """Display a single, clear prediction card for the result."""
    filename = os.path.basename(result.get('file_path', 'Unknown'))
    prediction = get_single_prediction(result)

    # Determine card color based on prediction
    if prediction == "No prediction":
        card_color = "#E74C3C"  # Red for no prediction
        icon = "❌"
        confidence_text = ""
    else:
        card_color = "#27AE60"  # Green for successful prediction
        icon = "✅"

        # Get confidence if available
        confidence = 0.0
        for detection in result.get('detections', []):
            if 'classification' in detection and detection['classification'].get('success'):
                conf = detection['classification'].get('confidence', 0)
                if conf > confidence:
                    confidence = conf

        confidence_text = f" (Confidence: {confidence:.1%})" if confidence > 0 else ""

    # Display the prediction card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {card_color}, {card_color}dd);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    ">
        <h2 style="color: white !important; margin-bottom: 1rem; font-size: 2.5rem;">
            {icon} {prediction}
        </h2>
        <p style="color: white !important; font-size: 1.2rem; margin: 0;">
            📁 {filename}{confidence_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_results(results: List[Dict[str, Any]]):
    """Display processing results with enhanced UI."""
    if not results:
        st.info("No results to display")
        return

    # Display Single Predictions First (Most Important)
    st.markdown("## 🎯 **SIGN LANGUAGE PREDICTIONS**")

    # Create a summary table of all predictions
    prediction_data = []
    for result in results:
        filename = os.path.basename(result.get('file_path', 'Unknown'))
        prediction = get_single_prediction(result)

        # Get confidence
        confidence = 0.0
        for detection in result.get('detections', []):
            if 'classification' in detection and detection['classification'].get('success'):
                conf = detection['classification'].get('confidence', 0)
                if conf > confidence:
                    confidence = conf

        prediction_data.append({
            'File': filename,
            'Prediction': prediction,
            'Confidence': f"{confidence:.1%}" if confidence > 0 else "N/A"
        })

    if prediction_data:
        # Display as a clean table
        import pandas as pd
        df = pd.DataFrame(prediction_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Individual Prediction Cards")

    # Show single prediction cards for each file
    for result in results:
        display_single_prediction_card(result)

    # Add separator
    st.markdown("---")

    # Create metrics dashboard
    create_metrics_dashboard(results)

    # Create visualizations
    col1, col2 = st.columns(2)
    with col1:
        create_confidence_chart(results, "main_confidence_chart")
    with col2:
        create_gesture_analysis_chart(results, "main_gesture_analysis_chart")

    # Display individual results
    st.markdown("## 📋 Detailed Analysis")

    for result in results:
        if result.get('file_type') == 'image':
            display_image_results(result)
        elif result.get('file_type') == 'video':
            display_video_results(result)
        else:
            st.error(f"❌ Failed to process {result.get('filename', 'unknown file')}: {result.get('error', 'Unknown error')}")


def display_quick_summary(results: List[Dict[str, Any]]):
    """Display a quick summary of predictions at the top."""
    if not results:
        return

    predictions = []
    for result in results:
        filename = os.path.basename(result.get('file_path', 'Unknown'))
        prediction = get_single_prediction(result)
        if prediction != "No prediction":
            predictions.append(f"**{filename}** → **{prediction}**")

    if predictions:
        st.success("🎯 **Quick Results:** " + " | ".join(predictions))
    else:
        st.warning("⚠️ No clear predictions found in uploaded files")

def main():
    """Enhanced Streamlit application with modern UI."""
    # Create header
    create_header()

    # Show quick summary if results exist
    if st.session_state.processing_results:
        display_quick_summary(st.session_state.processing_results)

    # Initialize components
    initialize_components()

    # Sidebar configuration
    st.sidebar.markdown("# 🎛️ Control Panel")

    # AI API setup
    api_key, use_gemini = setup_ai_api()

    # Settings panel
    settings = create_settings_panel()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 File Processing", "📊 Analytics", "ℹ️ About"])

    with tab1:
        st.markdown("## 📁 File Processing")

        # Enhanced file upload area
        create_file_upload_area()

        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=True,
            help="Upload multiple images or videos for batch processing"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")

            # Show file details
            with st.expander("📋 File Details", expanded=True):
                for file in uploaded_files:
                    file_size = len(file.getvalue())
                    st.write(f"• **{file.name}** ({file_size / 1024:.1f} KB)")

            # Process button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                    if not api_key:
                        st.error("❌ Please provide an OpenAI API key to analyze gestures")
                    else:
                        with st.spinner("🔄 Processing files..."):
                            results = process_uploaded_files(uploaded_files, api_key, settings, use_gemini)
                            st.session_state.processing_results = results

                            if results:
                                st.success(f"✅ Processing complete! {len(results)} files processed.")
                                display_results(results)

                                # Export options
                                st.markdown("### 📤 Export Results")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    export_results(results, settings['export_format'])
                                with col_b:
                                    if st.button("🗑️ Clear Results"):
                                        st.session_state.processing_results = []
                                        st.experimental_rerun()

        # Display previous results if available
        elif st.session_state.processing_results:
            st.markdown("### 📊 Previous Results")
            display_results(st.session_state.processing_results)

            # Export options
            st.markdown("### 📤 Export Results")
            col_a, col_b = st.columns(2)
            with col_a:
                export_results(st.session_state.processing_results, settings['export_format'])
            with col_b:
                if st.button("🗑️ Clear Results"):
                    st.session_state.processing_results = []
                    st.experimental_rerun()

    with tab2:
        st.markdown("## 📊 Analytics Dashboard")

        if st.session_state.processing_results:
            results = st.session_state.processing_results

            # Overall statistics
            st.markdown("### 📈 Overall Statistics")
            create_metrics_dashboard(results)

            # Detailed charts
            st.markdown("### 📊 Detailed Analysis")
            col1, col2 = st.columns(2)

            with col1:
                create_confidence_chart(results, "analytics_confidence_chart")

            with col2:
                create_gesture_analysis_chart(results, "analytics_gesture_analysis_chart")

            # File processing timeline
            st.markdown("### ⏱️ Processing Timeline")
            if results:
                timeline_data = []
                for i, result in enumerate(results):
                    timeline_data.append({
                        'File': result.get('filename', f'File {i+1}'),
                        'Success': result.get('success', False),
                        'Hands': result.get('hands_detected', 0) if result.get('success') else 0,
                        'Size (KB)': result.get('file_size', 0) / 1024
                    })

                df = pd.DataFrame(timeline_data)

                fig = px.scatter(df, x='Size (KB)', y='Hands',
                               color='Success', size='Hands',
                               hover_data=['File'],
                               title='File Size vs Hands Detected')
                st.plotly_chart(fig, use_container_width=True, key="file_size_scatter")
        else:
            st.info("📊 No data available. Process some files to see analytics.")

    with tab3:
        st.markdown("## ℹ️ About Sign Language Detector Pro")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 Features
            - **Advanced File Processing**: Batch analysis of images and videos
            - **AI-Powered Classification**: OpenAI API integration for accurate gesture recognition
            - **Interactive Analytics**: Real-time charts and metrics
            - **Multiple Export Formats**: JSON, CSV, and PDF reports
            - **Professional UI**: Modern, responsive design
            - **Comprehensive Analysis**: Hand landmarks, gesture features, and confidence scores

            ### 🔧 How It Works
            1. **Upload Files**: Drag and drop or select multiple files
            2. **Hand Detection**: MediaPipe detects 21 hand landmarks
            3. **Feature Extraction**: Advanced gesture analysis
            4. **AI Classification**: OpenAI interprets gestures
            5. **Results Display**: Interactive charts and detailed analysis
            """)

        with col2:
            st.markdown("""
            ### 📋 Supported Formats
            **Images:**
            - JPG, JPEG, PNG, BMP

            **Videos:**
            - MP4, AVI, MOV, MKV

            ### ⚙️ System Requirements
            - Python 3.8+
            - OpenAI API key
            - Modern web browser

            ### 🚀 Performance
            - Batch processing support
            - Optimized video frame sampling
            - Real-time progress tracking
            - Memory-efficient processing
            """)

        # System information
        st.markdown("### 💻 System Information")
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.info(f"**Python:** {sys.version.split()[0]}")
            st.info(f"**OpenCV:** {cv2.__version__}")

        with info_col2:
            st.info(f"**Streamlit:** {st.__version__}")
            api_status = "✅ Configured" if api_key else "❌ Not configured"
            st.info(f"**OpenAI API:** {api_status}")

    # Enhanced footer with improved text visibility
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; color: #FFFFFF !important; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h4 style="color: #FFFFFF !important; margin-bottom: 1rem; font-weight: 600;">🤟 Sign Language Detector Pro</h4>
        <p style="color: #FFFFFF !important; margin-bottom: 0.5rem; font-size: 1.1rem;">Empowering communication through AI-powered gesture recognition</p>
        <p style="color: #FFFFFF !important; margin: 0; opacity: 0.9;"><small>Built with ❤️ using MediaPipe, OpenAI, and Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
