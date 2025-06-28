"""
Utility functions for the AI Body Measurement Application
"""

import os
import sys
import logging
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
from urllib.parse import urlparse

# Import config separately to avoid circular imports
from .config import EnhancedConfig

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = f"logs/body_measurement_{timestamp}.log"
        handlers.append(logging.FileHandler(default_log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def save_results(results, file_path: str, config: EnhancedConfig):
    """
    Save measurement results to file
    
    Args:
        results: Measurement results to save
        file_path: Output file path
        config: Application configuration
    """
    
    file_path = Path(file_path)
    
    # Prepare data for export
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'reference_height': results.reference_height,
        'pixel_to_cm_ratio': results.pixel_to_cm_ratio,
        'calibration_method': results.calibration_method,
        'total_confidence': results.total_confidence,
        'measurements': {},
        'metadata': results.metadata
    }
    
    # Add measurements
    for name, measurement in results.measurements.items():
        export_data['measurements'][name] = {
            'value': measurement.value,
            'unit': measurement.unit,
            'confidence': measurement.confidence,
            'method': measurement.method,
            'points_used': measurement.points_used
        }
    
    # Save based on file extension
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    elif file_path.suffix.lower() == '.csv':
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Measurement', 'Value', 'Unit', 'Confidence', 'Method'])
            
            # Write measurements
            for name, measurement in results.measurements.items():
                writer.writerow([
                    name.replace('_', ' ').title(),
                    measurement.value,
                    measurement.unit,
                    f"{measurement.confidence:.3f}",
                    measurement.method
                ])
    
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, indent=2)
    
    else:
        # Default to JSON
        with open(file_path.with_suffix('.json'), 'w') as f:
            json.dump(export_data, f, indent=2)

def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load measurement results from file
    
    Args:
        file_path: Path to results file
    
    Returns:
        Loaded results data
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def create_measurement_report(results, output_path: str, image_path: Optional[str] = None):
    """
    Create an HTML report of measurement results
    
    Args:
        results: Measurement results
        output_path: Output HTML file path
        image_path: Optional path to original image
    """
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Body Measurement Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #007bff;
                padding-bottom: 20px;
            }}
            .header h1 {{
                color: #333;
                margin: 0;
            }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .summary-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                color: #007bff;
            }}
            .measurements {{
                margin-bottom: 30px;
            }}
            .measurements table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .measurements th, .measurements td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .measurements th {{
                background-color: #007bff;
                color: white;
            }}
            .confidence-high {{ color: #28a745; }}
            .confidence-medium {{ color: #ffc107; }}
            .confidence-low {{ color: #dc3545; }}
            .image-section {{
                text-align: center;
                margin-top: 30px;
            }}
            .image-section img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Body Measurement Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <h3>Reference Height</h3>
                    <p>{reference_height:.1f} cm</p>
                </div>
                <div class="summary-card">
                    <h3>Overall Confidence</h3>
                    <p>{total_confidence:.1%}</p>
                </div>
                <div class="summary-card">
                    <h3>Calibration Method</h3>
                    <p>{calibration_method}</p>
                </div>
                <div class="summary-card">
                    <h3>Total Measurements</h3>
                    <p>{total_measurements}</p>
                </div>
            </div>
            
            <div class="measurements">
                <h2>Detailed Measurements</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Measurement</th>
                            <th>Value</th>
                            <th>Unit</th>
                            <th>Confidence</th>
                            <th>Method</th>
                        </tr>
                    </thead>
                    <tbody>
                        {measurement_rows}
                    </tbody>
                </table>
            </div>
            
            {image_section}
            
            <div class="footer">
                <p>Report generated by AI Body Measurement Application</p>
                <p>Calibration ratio: {pixel_ratio:.4f} cm/pixel</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate measurement rows
    measurement_rows = ""
    for name, measurement in results.measurements.items():
        confidence_class = (
            "confidence-high" if measurement.confidence >= 0.8 else
            "confidence-medium" if measurement.confidence >= 0.6 else
            "confidence-low"
        )
        
        measurement_rows += f"""
        <tr>
            <td>{name.replace('_', ' ').title()}</td>
            <td>{measurement.value}</td>
            <td>{measurement.unit}</td>
            <td class="{confidence_class}">{measurement.confidence:.1%}</td>
            <td>{measurement.method.replace('_', ' ').title()}</td>
        </tr>
        """
    
    # Image section
    image_section = ""
    if image_path and os.path.exists(image_path):
        # Copy image to report directory or encode as base64
        image_section = f"""
        <div class="image-section">
            <h2>Original Image</h2>
            <p><em>Image: {os.path.basename(image_path)}</em></p>
        </div>
        """
    
    # Fill template
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        reference_height=results.reference_height,
        total_confidence=results.total_confidence,
        calibration_method=results.calibration_method.replace('_', ' ').title(),
        total_measurements=len(results.measurements),
        measurement_rows=measurement_rows,
        image_section=image_section,
        pixel_ratio=results.pixel_to_cm_ratio
    )
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def validate_image(image_path: str) -> bool:
    """
    Validate if the image file is valid and readable
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        height, width = image.shape[:2]
        
        # Check minimum dimensions
        if width < 100 or height < 100:
            return False
        
        # Check maximum dimensions (to avoid memory issues)
        if width > 10000 or height > 10000:
            return False
        
        return True
        
    except Exception:
        return False

def resize_image_if_needed(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Resize image if it's too large
    
    Args:
        image: Input image
        max_size: Maximum dimension size
    
    Returns:
        Resized image
    """
    
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better detection
    
    Args:
        image: Input image
    
    Returns:
        Enhanced image
    """
    
    enhanced = image.copy()
    
    # Convert to LAB color space for better luminance control
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Slight sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
    enhanced = cv2.addWeighted(image, 0.8, enhanced, 0.2, 0)
    
    return enhanced

def calculate_body_symmetry(keypoints: Dict[str, tuple]) -> float:
    """
    Calculate body symmetry score from keypoints
    
    Args:
        keypoints: Dictionary of keypoint coordinates
    
    Returns:
        Symmetry score between 0 and 1
    """
    
    # Define symmetric point pairs
    symmetric_pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_wrist', 'right_wrist'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
        ('left_ankle', 'right_ankle'),
        ('left_eye', 'right_eye'),
        ('left_ear', 'right_ear')
    ]
    
    symmetry_scores = []
    
    # Find body center line (vertical line through nose or midpoint of shoulders)
    center_x = None
    if 'nose' in keypoints:
        center_x = keypoints['nose'][0]
    elif 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        center_x = (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2
    
    if center_x is None:
        return 0.5  # Default symmetry score
    
    # Calculate symmetry for each pair
    for left_point, right_point in symmetric_pairs:
        if left_point in keypoints and right_point in keypoints:
            left_x, left_y = keypoints[left_point][:2]
            right_x, right_y = keypoints[right_point][:2]
            
            # Calculate distance from center line
            left_dist = abs(left_x - center_x)
            right_dist = abs(right_x - center_x)
            
            # Calculate symmetry score for this pair
            max_dist = max(left_dist, right_dist)
            min_dist = min(left_dist, right_dist)
            
            if max_dist > 0:
                pair_symmetry = min_dist / max_dist
            else:
                pair_symmetry = 1.0
            
            symmetry_scores.append(pair_symmetry)
    
    # Return average symmetry score
    return np.mean(symmetry_scores) if symmetry_scores else 0.5

def estimate_pose_quality(keypoints: Dict[str, tuple]) -> Dict[str, float]:
    """
    Estimate pose quality metrics
    
    Args:
        keypoints: Dictionary of keypoint coordinates
    
    Returns:
        Dictionary with quality metrics
    """
    
    quality_metrics = {
        'completeness': 0.0,
        'confidence': 0.0,
        'symmetry': 0.0,
        'visibility': 0.0
    }
    
    if not keypoints:
        return quality_metrics
    
    # Completeness: ratio of detected keypoints to expected keypoints
    expected_keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    detected_count = len([kp for kp in expected_keypoints if kp in keypoints])
    quality_metrics['completeness'] = detected_count / len(expected_keypoints)
    
    # Average confidence
    confidences = [kp[2] for kp in keypoints.values() if len(kp) > 2]
    quality_metrics['confidence'] = np.mean(confidences) if confidences else 0.0
    
    # Symmetry
    quality_metrics['symmetry'] = calculate_body_symmetry(keypoints)
    
    # Visibility (based on confidence threshold)
    visible_count = sum(1 for kp in keypoints.values() if len(kp) > 2 and kp[2] > 0.5)
    quality_metrics['visibility'] = visible_count / len(keypoints) if keypoints else 0.0
    
    return quality_metrics

def create_debug_visualization(image: np.ndarray, detections: List, measurements: Dict) -> np.ndarray:
    """
    Create detailed debug visualization
    
    Args:
        image: Original image
        detections: Detection results
        measurements: Measurement results
    
    Returns:
        Debug visualization image
    """
    
    debug_image = image.copy()
    
    # Add debug information overlay
    info_text = [
        f"Detections: {len(detections)}",
        f"Measurements: {len(measurements)}",
        f"Image size: {image.shape[1]}x{image.shape[0]}"
    ]
    
    # Draw info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    y_offset = 30
    for text in info_text:
        cv2.putText(debug_image, text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
    
    return debug_image

class Timer:
    """Simple timer context manager for performance measurement"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logging.info(f"{self.name} took {duration:.3f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    
    import platform
    import psutil
    
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        torch_available = False
        cuda_available = False
        gpu_count = 0
    
    try:
        import cv2 as cv_info
        opencv_version = cv_info.__version__
    except ImportError:
        opencv_version = "Not available"
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'opencv_version': opencv_version,
        'torch_available': torch_available,
        'cuda_available': cuda_available,
        'gpu_count': gpu_count,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    return system_info