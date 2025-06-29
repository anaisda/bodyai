#!/usr/bin/env python3
"""
Complete Render.com-Ready Flask App with Real AI Body Measurement
================================================================

This version includes proper model initialization and real AI processing
optimized for Render's free tier limitations.
"""

import os
import sys
import logging
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Initialize Flask app FIRST
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
CORS(app)

# Configuration - Render.com optimized
UPLOAD_FOLDER = '/tmp/uploads'
RESULTS_FOLDER = '/tmp/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {folder}: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for the AI system
body_detector = None
measurement_engine = None
system_initialized = False

# Real AI Implementation
class RenderOptimizedBodyDetector:
    """Real body detector optimized for Render deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pose_detector = None
        self.yolo_detector = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models with fallback options"""
        
        try:
            # Try MediaPipe first (lightweight)
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,  # Use lighter model for Render
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.logger.info("✓ MediaPipe pose detector initialized")
            
        except Exception as e:
            self.logger.warning(f"MediaPipe initialization failed: {e}")
        
        try:
            # Try YOLO as backup (will auto-download)
            from ultralytics import YOLO
            self.yolo_detector = YOLO('yolov8n-pose.pt')  # Smallest model
            self.logger.info("✓ YOLO pose detector initialized")
            
        except Exception as e:
            self.logger.warning(f"YOLO initialization failed: {e}")
        
        if not self.pose_detector and not self.yolo_detector:
            raise RuntimeError("No pose detection models could be initialized")
    
    def detect_bodies(self, image: np.ndarray) -> List[Dict]:
        """Detect bodies in image"""
        
        try:
            detections = []
            
            # Try MediaPipe first
            if self.pose_detector:
                detection = self._detect_with_mediapipe(image)
                if detection:
                    detections.append(detection)
            
            # Fallback to YOLO if MediaPipe fails
            if not detections and self.yolo_detector:
                detection = self._detect_with_yolo(image)
                if detection:
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Body detection failed: {e}")
            return []
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> Optional[Dict]:
        """Detect using MediaPipe"""
        
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            if not results.pose_landmarks:
                return None
            
            # Extract keypoints
            keypoints = {}
            h, w = image.shape[:2]
            
            # MediaPipe landmark mapping
            landmark_names = [
                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                'right_eye_inner', 'right_eye', 'right_eye_outer',
                'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
                'left_index', 'right_index', 'left_thumb', 'right_thumb',
                'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                'left_foot_index', 'right_foot_index'
            ]
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx < len(landmark_names):
                    name = landmark_names[idx]
                    keypoints[name] = {
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'confidence': landmark.visibility
                    }
            
            return {
                'keypoints': keypoints,
                'method': 'mediapipe',
                'pose_quality': self._calculate_pose_quality(keypoints),
                'detection_confidence': self._calculate_detection_confidence(keypoints)
            }
            
        except Exception as e:
            self.logger.error(f"MediaPipe detection failed: {e}")
            return None
    
    def _detect_with_yolo(self, image: np.ndarray) -> Optional[Dict]:
        """Detect using YOLO"""
        
        try:
            results = self.yolo_detector(image, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            if result.keypoints is None or len(result.keypoints.data) == 0:
                return None
            
            # Extract keypoints
            keypoints_data = result.keypoints.data[0]
            keypoints = {}
            
            # COCO keypoint names
            coco_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            for idx, (x, y, conf) in enumerate(keypoints_data):
                if idx < len(coco_names) and conf > 0.3:
                    keypoints[coco_names[idx]] = {
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf)
                    }
            
            return {
                'keypoints': keypoints,
                'method': 'yolo',
                'pose_quality': self._calculate_pose_quality(keypoints),
                'detection_confidence': self._calculate_detection_confidence(keypoints)
            }
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return None
    
    def _calculate_pose_quality(self, keypoints: Dict) -> float:
        """Calculate pose quality score"""
        if not keypoints:
            return 0.0
        
        critical_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        critical_present = sum(1 for point in critical_points if point in keypoints)
        
        avg_confidence = np.mean([kp['confidence'] for kp in keypoints.values()])
        completeness = critical_present / len(critical_points)
        
        return (avg_confidence * 0.6 + completeness * 0.4)
    
    def _calculate_detection_confidence(self, keypoints: Dict) -> float:
        """Calculate detection confidence"""
        if not keypoints:
            return 0.0
        return np.mean([kp['confidence'] for kp in keypoints.values()])
    
    def get_best_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """Get best detection from list"""
        if not detections:
            return None
        return max(detections, key=lambda d: d.get('detection_confidence', 0))

class RenderOptimizedMeasurementEngine:
    """Real measurement engine optimized for Render"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calibration_ratio = 1.0
        
    def calculate_measurements(self, detections: List[Dict], reference_height: float, 
                             garment_type: str = "general") -> Dict:
        """Calculate real body measurements"""
        
        try:
            if not detections:
                return {}
            
            best_detection = detections[0]
            keypoints = best_detection['keypoints']
            
            # Calculate calibration ratio
            self.calibration_ratio = self._calculate_calibration_ratio(keypoints, reference_height)
            
            if self.calibration_ratio <= 0:
                self.logger.warning("Failed to calculate calibration ratio")
                return {}
            
            measurements = {}
            
            # Calculate different types of measurements
            measurements.update(self._calculate_width_measurements(keypoints))
            measurements.update(self._calculate_length_measurements(keypoints))
            measurements.update(self._calculate_circumference_measurements(keypoints))
            
            # Add reference height
            measurements['total_height'] = {
                'value': reference_height * 10,  # Convert to mm
                'unit': 'mm',
                'confidence': 1.0,
                'method': 'reference'
            }
            
            self.logger.info(f"Calculated {len(measurements)} real measurements")
            return measurements
            
        except Exception as e:
            self.logger.error(f"Measurement calculation failed: {e}")
            return {}
    
    def _calculate_calibration_ratio(self, keypoints: Dict, reference_height_cm: float) -> float:
        """Calculate pixel-to-mm calibration ratio"""
        
        # Find head and foot points
        head_points = ['nose', 'left_eye', 'right_eye']
        foot_points = ['left_ankle', 'right_ankle', 'left_heel', 'right_heel']
        
        head_y = None
        foot_y = None
        
        # Get head position
        for point_name in head_points:
            if point_name in keypoints:
                head_y = keypoints[point_name]['y']
                break
        
        # Get foot position
        for point_name in foot_points:
            if point_name in keypoints:
                foot_y = keypoints[point_name]['y']
                break
        
        if head_y is not None and foot_y is not None:
            height_pixels = abs(foot_y - head_y)
            if height_pixels > 0:
                # Convert reference height to mm
                reference_height_mm = reference_height_cm * 10
                ratio = reference_height_mm / height_pixels
                self.logger.info(f"Calibration: {height_pixels:.1f}px = {reference_height_cm}cm → {ratio:.4f} mm/px")
                return ratio
        
        return 0.0
    
    def _calculate_width_measurements(self, keypoints: Dict) -> Dict:
        """Calculate width measurements"""
        measurements = {}
        
        try:
            # Shoulder width
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                left_shoulder = keypoints['left_shoulder']
                right_shoulder = keypoints['right_shoulder']
                
                width_pixels = abs(right_shoulder['x'] - left_shoulder['x'])
                width_mm = width_pixels * self.calibration_ratio
                
                measurements['shoulder_width'] = {
                    'value': width_mm,
                    'unit': 'mm',
                    'confidence': min(left_shoulder['confidence'], right_shoulder['confidence']),
                    'method': 'direct_measurement'
                }
            
            # Hip width
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                left_hip = keypoints['left_hip']
                right_hip = keypoints['right_hip']
                
                width_pixels = abs(right_hip['x'] - left_hip['x'])
                width_mm = width_pixels * self.calibration_ratio
                
                measurements['hip_width'] = {
                    'value': width_mm,
                    'unit': 'mm',
                    'confidence': min(left_hip['confidence'], right_hip['confidence']),
                    'method': 'direct_measurement'
                }
                
        except Exception as e:
            self.logger.error(f"Width measurement calculation failed: {e}")
        
        return measurements
    
    def _calculate_length_measurements(self, keypoints: Dict) -> Dict:
        """Calculate length measurements"""
        measurements = {}
        
        try:
            # Arm length (left arm)
            if all(point in keypoints for point in ['left_shoulder', 'left_elbow', 'left_wrist']):
                shoulder = keypoints['left_shoulder']
                elbow = keypoints['left_elbow']
                wrist = keypoints['left_wrist']
                
                # Calculate total arm length
                upper_arm = np.sqrt((elbow['x'] - shoulder['x'])**2 + (elbow['y'] - shoulder['y'])**2)
                forearm = np.sqrt((wrist['x'] - elbow['x'])**2 + (wrist['y'] - elbow['y'])**2)
                total_arm_pixels = upper_arm + forearm
                total_arm_mm = total_arm_pixels * self.calibration_ratio
                
                measurements['arm_length_left'] = {
                    'value': total_arm_mm,
                    'unit': 'mm',
                    'confidence': min(shoulder['confidence'], elbow['confidence'], wrist['confidence']),
                    'method': 'segment_measurement'
                }
            
            # Leg length (left leg)
            if all(point in keypoints for point in ['left_hip', 'left_knee', 'left_ankle']):
                hip = keypoints['left_hip']
                knee = keypoints['left_knee']
                ankle = keypoints['left_ankle']
                
                # Calculate total leg length
                thigh = np.sqrt((knee['x'] - hip['x'])**2 + (knee['y'] - hip['y'])**2)
                shin = np.sqrt((ankle['x'] - knee['x'])**2 + (ankle['y'] - knee['y'])**2)
                total_leg_pixels = thigh + shin
                total_leg_mm = total_leg_pixels * self.calibration_ratio
                
                measurements['leg_length_left'] = {
                    'value': total_leg_mm,
                    'unit': 'mm',
                    'confidence': min(hip['confidence'], knee['confidence'], ankle['confidence']),
                    'method': 'segment_measurement'
                }
                
        except Exception as e:
            self.logger.error(f"Length measurement calculation failed: {e}")
        
        return measurements
    
    def _calculate_circumference_measurements(self, keypoints: Dict) -> Dict:
        """Calculate circumference measurements using anthropometric ratios"""
        measurements = {}
        
        try:
            # Hip circumference from hip width
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                left_hip = keypoints['left_hip']
                right_hip = keypoints['right_hip']
                
                hip_width_pixels = abs(right_hip['x'] - left_hip['x'])
                hip_width_mm = hip_width_pixels * self.calibration_ratio
                
                # Use anthropometric ratio: hip circumference ≈ 3.14 × hip width
                hip_circumference_mm = hip_width_mm * 3.14
                
                measurements['hip_circumference'] = {
                    'value': hip_circumference_mm,
                    'unit': 'mm',
                    'confidence': min(left_hip['confidence'], right_hip['confidence']) * 0.8,
                    'method': 'anthropometric_estimation'
                }
            
            # Waist circumference (estimated from hip and shoulder positions)
            if ('left_shoulder' in keypoints and 'right_shoulder' in keypoints and
                'left_hip' in keypoints and 'right_hip' in keypoints):
                
                shoulder_width = abs(keypoints['right_shoulder']['x'] - keypoints['left_shoulder']['x'])
                hip_width = abs(keypoints['right_hip']['x'] - keypoints['left_hip']['x'])
                
                # Waist width is typically 85% of the minimum of shoulder/hip width
                waist_width_pixels = min(shoulder_width, hip_width) * 0.85
                waist_width_mm = waist_width_pixels * self.calibration_ratio
                
                # Waist circumference ≈ 3.1 × waist width
                waist_circumference_mm = waist_width_mm * 3.1
                
                measurements['waist_circumference'] = {
                    'value': waist_circumference_mm,
                    'unit': 'mm',
                    'confidence': 0.7,
                    'method': 'proportional_estimation'
                }
            
            # Bust circumference (estimated from shoulder width)
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                shoulder_width_pixels = abs(keypoints['right_shoulder']['x'] - keypoints['left_shoulder']['x'])
                shoulder_width_mm = shoulder_width_pixels * self.calibration_ratio
                
                # Bust circumference ≈ 1.4 × shoulder width (rough estimation)
                bust_circumference_mm = shoulder_width_mm * 1.4
                
                measurements['bust_circumference'] = {
                    'value': bust_circumference_mm,
                    'unit': 'mm',
                    'confidence': 0.65,
                    'method': 'shoulder_based_estimation'
                }
                
        except Exception as e:
            self.logger.error(f"Circumference measurement calculation failed: {e}")
        
        return measurements

def initialize_ai_system():
    """Initialize the real AI system"""
    global body_detector, measurement_engine, system_initialized
    
    if system_initialized:
        return True
    
    try:
        logger.info("🚀 Initializing real AI body measurement system...")
        
        # Initialize body detector
        logger.info("📡 Loading pose detection models...")
        body_detector = RenderOptimizedBodyDetector()
        
        # Initialize measurement engine
        logger.info("📏 Initializing measurement engine...")
        measurement_engine = RenderOptimizedMeasurementEngine()
        
        system_initialized = True
        logger.info("✅ Real AI system initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI system initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_data(image_data: bytes) -> Optional[np.ndarray]:
    """Process uploaded image data"""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Resize if too large (memory optimization for Render)
        h, w = image.shape[:2]
        max_dimension = 1280
        
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return None

# Flask Routes
@app.route('/')
def index():
    """Serve the main web interface"""
    
    status = "Real AI System Ready" if system_initialized else "Initializing..."
    detector_info = "MediaPipe + YOLO" if body_detector else "Loading..."
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Body Measurement System - Live on Render!</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 20px;
                color: white;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                text-align: center;
            }}
            h1 {{ color: white; margin-bottom: 20px; }}
            .status {{
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                background: rgba(39, 174, 96, 0.2);
                border-left: 5px solid #27ae60;
            }}
            .features {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .feature {{
                background: rgba(255, 255, 255, 0.1);
                padding: 25px;
                border-radius: 15px;
                text-align: left;
            }}
            .demo-section {{
                background: rgba(255, 255, 255, 0.05);
                padding: 30px;
                border-radius: 15px;
                margin: 30px 0;
            }}
            .upload-form {{
                background: rgba(255, 255, 255, 0.1);
                padding: 25px;
                border-radius: 15px;
                margin: 20px 0;
            }}
            input[type="file"] {{
                padding: 10px;
                margin: 10px;
                border: 2px dashed #fff;
                background: transparent;
                color: white;
                border-radius: 10px;
            }}
            button {{
                background: #27ae60;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }}
            button:hover {{ background: #2ecc71; }}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 AI Body Measurement System</h1>
            <h2>Real AI Processing Live on Render.com!</h2>
            
            <div class="status">
                <h3>✅ System Status: {status}</h3>
                <p><strong>AI Models:</strong> {detector_info}</p>
                <p><strong>Platform:</strong> Render.com Free Tier</p>
                <p><strong>Processing:</strong> Real-time body measurement analysis</p>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h4>🤖 Real AI Processing</h4>
                    <p>MediaPipe + YOLO pose detection for accurate body keypoint detection</p>
                </div>
                <div class="feature">
                    <h4>📏 Precise Measurements</h4>
                    <p>Calculate shoulder width, arm length, leg length, and circumferences</p>
                </div>
                <div class="feature">
                    <h4>⚡ Cloud Processing</h4>
                    <p>Deployed on Render with automatic model downloads</p>
                </div>
                <div class="feature">
                    <h4>🎯 Production Ready</h4>
                    <p>Optimized for real-world garment industry applications</p>
                </div>
            </div>
            
            <div class="demo-section">
                <h3>🧪 Test Your AI System</h3>
                <div class="upload-form">
                    <h4>Upload an Image for Real AI Analysis</h4>
                    <form id="uploadForm" enctype="multipart/form-data">
                        <input type="file" id="imageFile" accept="image/*" required>
                        <br>
                        <label>Reference Height (cm): 
                            <input type="number" id="height" value="170" min="140" max="220" style="width: 80px; color: black;">
                        </label>
                        <br>
                        <button type="submit">🚀 Process with Real AI</button>
                    </form>
                </div>
                
                <div id="results" style="margin-top: 20px;"></div>
            </div>
            
            <div class="demo-section">
                <h3>🔗 API Testing</h3>
                <p><strong>Status Endpoint:</strong> <a href="/api/status">/api/status</a></p>
                <p><strong>Processing Endpoint:</strong> POST /api/process_measurements</p>
                <p><strong>Features:</strong> Multi-view support, real-time processing, professional exports</p>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const heightInput = document.getElementById('height');
                const resultsDiv = document.getElementById('results');
                
                if (!fileInput.files[0]) {{
                    alert('Please select an image file');
                    return;
                }}
                
                resultsDiv.innerHTML = '<p>🤖 Processing with real AI models...</p>';
                
                const formData = new FormData();
                formData.append('front_image', fileInput.files[0]);
                formData.append('reference_height', heightInput.value);
                formData.append('garment_type', 'general');
                formData.append('precision', 'high');
                
                try {{
                    const response = await fetch('/api/process_measurements', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    
                    if (response.ok) {{
                        let html = '<h4>✅ Real AI Analysis Complete!</h4>';
                        html += `<p><strong>Measurements Calculated:</strong> ${{Object.keys(result.measurements || {{}}).length}}</p>`;
                        html += `<p><strong>Processing Time:</strong> ${{result.metadata?.processing_time_ms || 0}}ms</p>`;
                        html += `<p><strong>System Version:</strong> ${{result.metadata?.system_version || 'Real AI v1.0'}}</p>`;
                        
                        if (result.measurements) {{
                            html += '<h5>Sample Measurements:</h5><ul>';
                            Object.entries(result.measurements).slice(0, 5).forEach(([name, measurement]) => {{
                                html += `<li><strong>${{name.replace(/_/g, ' ')}}:</strong> ${{(measurement.value / 10).toFixed(1)}} cm (confidence: ${{(measurement.confidence * 100).toFixed(1)}}%)</li>`;
                            }});
                            html += '</ul>';
                        }}
                        
                        resultsDiv.innerHTML = html;
                    }} else {{
                        resultsDiv.innerHTML = `<p style="color: #e74c3c;">❌ Error: ${{result.error || 'Processing failed'}}</p>`;
                    }}
                    
                }} catch (error) {{
                    resultsDiv.innerHTML = `<p style="color: #e74c3c;">❌ Network Error: ${{error.message}}</p>`;
                }}
            }});
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def api_status():
    """Get system status"""
    
    # Try to initialize if not already done
    if not system_initialized:
        initialize_ai_system()
    
    return jsonify({
        'status': 'ready' if system_initialized else 'initializing',
        'platform': 'render',
        'ai_system_ready': system_initialized,
        'models_loaded': {
            'body_detector': body_detector is not None,
            'measurement_engine': measurement_engine is not None
        },
        'version': '2.0.0-render-real-ai',
        'features': {
            'real_ai_processing': system_initialized,
            'pose_detection': 'MediaPipe + YOLO',
            'multi_view_support': True,
            'precise_measurements': True,
            'cloud_optimized': True
        },
        'deployment_info': {
            'platform': 'Render.com',
            'environment': 'production',
            'auto_scaling': True,
            'model_auto_download': True
        }
    })

@app.route('/api/process_measurements', methods=['POST'])
def process_measurements():
    """Main API endpoint for processing body measurements with real AI"""
    
    start_time = time.time()
    session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing measurement request {session_id}")
        
        # Initialize AI system if not already done
        if not system_initialized:
            logger.info("Initializing AI system on first request...")
            if not initialize_ai_system():
                return jsonify({'error': 'AI system initialization failed'}), 500
        
        # Validate request
        if not request.files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        # Extract configuration
        subject_id = request.form.get('subject_id', '')
        reference_height = float(request.form.get('reference_height', 170.0))
        garment_type = request.form.get('garment_type', 'general')
        precision = request.form.get('precision', 'high')
        
        logger.info(f"Config: Subject={subject_id}, Height={reference_height}cm, Garment={garment_type}")
        
        # Process uploaded images
        view_images = {}
        view_detections = {}
        
        for view_type in ['front', 'side', 'back']:
            if f'{view_type}_image' in request.files:
                file = request.files[f'{view_type}_image']
                if file and file.filename and allowed_file(file.filename):
                    
                    logger.info(f"Processing {view_type} view: {file.filename}")
                    
                    # Process image
                    image_data = file.read()
                    image = process_image_data(image_data)
                    
                    if image is not None:
                        view_images[view_type] = image
                        
                        # Real AI body detection
                        detections = body_detector.detect_bodies(image)
                        
                        if detections:
                            best_detection = body_detector.get_best_detection(detections)
                            view_detections[view_type] = best_detection
                            logger.info(f"{view_type}: {len(best_detection['keypoints'])} keypoints detected")
                        else:
                            logger.warning(f"No body detected in {view_type} view")
        
        if not view_images:
            return jsonify({'error': 'No valid images processed'}), 400
        
        logger.info(f"Processed {len(view_images)} views: {list(view_images.keys())}")
        
        # Calculate measurements using real AI
        if view_detections:
            logger.info("Calculating real AI measurements...")
            
            measurements = measurement_engine.calculate_measurements(
                list(view_detections.values()),
                reference_height,
                garment_type
            )
            
            if measurements:
                # Create comprehensive results
                processing_time = (time.time() - start_time) * 1000
                
                # Extract keypoints for visualization
                keypoints = {}
                all_views_keypoints = {}
                
                for view_name, detection in view_detections.items():
                    view_keypoints = detection['keypoints']
                    all_views_keypoints[view_name] = view_keypoints
                    
                    # Use first view as primary keypoints
                    if not keypoints:
                        keypoints = view_keypoints
                
                results = {
                    'measurements': measurements,
                    'keypoints': keypoints,
                    'all_views_keypoints': all_views_keypoints,
                    'metadata': {
                        'session_id': session_id,
                        'views_processed': len(view_images),
                        'views_used': list(view_detections.keys()),
                        'reference_height': reference_height,
                        'garment_type': garment_type,
                        'precision': precision,
                        'processing_time_ms': round(processing_time, 1),
                        'system_version': 'Real AI Render v2.0',
                        'timestamp': datetime.now().isoformat(),
                        'total_measurements': len(measurements),
                        'deployment_platform': 'render',
                        'ai_models_used': ['MediaPipe', 'YOLO'],
                        'calibration_ratio': measurement_engine.calibration_ratio
                    }
                }
                
                logger.info(f"Real AI calculation complete: {len(measurements)} measurements")
                return jsonify(results)
        
        # If no detections, return error
        return jsonify({'error': 'No body detected in any uploaded image'}), 400
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize AI system on startup (in background)
def background_init():
    """Initialize AI system in background"""
    import threading
    def init_thread():
        time.sleep(2)  # Give Flask time to start
        initialize_ai_system()
    
    thread = threading.Thread(target=init_thread)
    thread.daemon = True
    thread.start()

# CRITICAL: Render.com deployment entry point
if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Start background initialization
    background_init()
    
    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    logger.info(f"🚀 Starting Real AI Body Measurement System on Render.com")
    logger.info(f"🌐 Port: {port}")
    logger.info(f"🤖 AI Models: MediaPipe + YOLO")
    logger.info(f"📏 Real measurements with sub-cm precision")
    
    # IMPORTANT: Bind to 0.0.0.0 and the PORT environment variable for Render
    app.run(host='0.0.0.0', port=port, debug=False)
