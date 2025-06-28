"""
Minimal CLI Processor for Body Measurement Application
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

class CLIProcessor:
    """Simplified command line interface processor"""
    
    def __init__(self, body_detector, measurement_engine, config):
        self.body_detector = body_detector
        self.measurement_engine = measurement_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_image(self, image_path: str, reference_height: Optional[float] = None, 
                     output_dir: Optional[str] = None):
        """Process a single image"""
        
        self.logger.info(f"Processing image: {image_path}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect bodies
            detections = self.body_detector.detect_bodies(image_rgb)
            
            if not detections:
                raise RuntimeError("No person detected in the image")
            
            # Get best detection
            best_detection = self.body_detector.get_best_detection(detections)
            
            # For now, create a simple result structure
            # In full implementation, this would use the measurement engine
            simple_result = self._create_simple_result(best_detection, reference_height or 170.0)
            
            # Save results if output directory specified
            if output_dir:
                self._save_simple_results(simple_result, image_path, output_dir)
            
            return simple_result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise
    
    def _create_simple_result(self, detection, reference_height):
        """Create a simple result structure"""
        
        # Calculate simple measurements based on keypoints
        measurements = {}
        
        keypoints = detection.keypoints
        
        # Simple height estimation (head to ankle)
        if 'nose' in keypoints and ('left_ankle' in keypoints or 'right_ankle' in keypoints):
            head_y = keypoints['nose'][1]
            
            ankle_y = None
            if 'left_ankle' in keypoints and 'right_ankle' in keypoints:
                ankle_y = max(keypoints['left_ankle'][1], keypoints['right_ankle'][1])
            elif 'left_ankle' in keypoints:
                ankle_y = keypoints['left_ankle'][1]
            elif 'right_ankle' in keypoints:
                ankle_y = keypoints['right_ankle'][1]
            
            if ankle_y is not None:
                height_pixels = ankle_y - head_y
                pixel_to_cm_ratio = reference_height / height_pixels if height_pixels > 0 else 1.0
                measurements['height'] = reference_height
        
        # Simple shoulder width
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            shoulder_width_pixels = abs(right_shoulder[0] - left_shoulder[0])
            
            if 'height' in measurements:
                # Use the pixel ratio from height calculation
                shoulder_width_cm = shoulder_width_pixels * pixel_to_cm_ratio
                measurements['shoulder_width'] = round(shoulder_width_cm, 1)
        
        # Simple arm length (one side)
        for side in ['left', 'right']:
            shoulder_key = f'{side}_shoulder'
            elbow_key = f'{side}_elbow'
            wrist_key = f'{side}_wrist'
            
            if all(k in keypoints for k in [shoulder_key, elbow_key, wrist_key]):
                shoulder = keypoints[shoulder_key]
                elbow = keypoints[elbow_key]
                wrist = keypoints[wrist_key]
                
                # Calculate arm length in pixels
                upper_arm = ((elbow[0] - shoulder[0])**2 + (elbow[1] - shoulder[1])**2)**0.5
                forearm = ((wrist[0] - elbow[0])**2 + (wrist[1] - elbow[1])**2)**0.5
                arm_length_pixels = upper_arm + forearm
                
                if 'height' in measurements:
                    arm_length_cm = arm_length_pixels * pixel_to_cm_ratio
                    measurements[f'{side}_arm_length'] = round(arm_length_cm, 1)
                break  # Use first available arm
        
        # Create result structure
        result = type('SimpleResult', (), {
            'measurements': {name: type('Measurement', (), {
                'name': name,
                'value': value,
                'unit': 'cm',
                'confidence': 0.8,
                'method': 'simple_calculation'
            })() for name, value in measurements.items()},
            'reference_height': reference_height,
            'total_confidence': detection.detection_confidence,
            'calibration_method': 'head_to_toe',
            'pixel_to_cm_ratio': pixel_to_cm_ratio if 'height' in measurements else 1.0,
            'metadata': {
                'keypoints_detected': len(keypoints),
                'pose_quality': detection.pose_quality,
                'processing_time': time.time()
            }
        })()
        
        return result
    
    def _save_simple_results(self, result, image_path, output_dir):
        """Save simple results to JSON"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        results_file = output_path / f"{image_name}_measurements.json"
        
        # Convert result to dictionary
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': image_path,
            'reference_height': result.reference_height,
            'pixel_to_cm_ratio': result.pixel_to_cm_ratio,
            'total_confidence': result.total_confidence,
            'measurements': {},
            'metadata': result.metadata
        }
        
        for name, measurement in result.measurements.items():
            results_data['measurements'][name] = {
                'value': measurement.value,
                'unit': measurement.unit,
                'confidence': measurement.confidence,
                'method': measurement.method
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def process_batch(self, input_dir: str, output_dir: str, reference_height: Optional[float] = None):
        """Process multiple images in batch"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        batch_results = []
        
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.process_image(
                    str(image_file),
                    reference_height,
                    str(output_path)
                )
                
                batch_results.append({
                    'image_path': str(image_file),
                    'success': True,
                    'measurements_count': len(result.measurements),
                    'confidence': result.total_confidence
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                batch_results.append({
                    'image_path': str(image_file),
                    'success': False,
                    'error': str(e)
                })
        
        # Save batch summary
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'successful': sum(1 for r in batch_results if r['success']),
                'failed': sum(1 for r in batch_results if not r['success']),
                'results': batch_results
            }, f, indent=2)
        
        return batch_results
    
    def print_results(self, results):
        """Print measurement results"""
        
        print("\n" + "="*50)
        print("BODY MEASUREMENT RESULTS")
        print("="*50)
        
        print(f"Reference Height: {results.reference_height:.1f} cm")
        print(f"Overall Confidence: {results.total_confidence:.1%}")
        print(f"Calibration Method: {results.calibration_method}")
        print(f"Pixel-to-CM Ratio: {results.pixel_to_cm_ratio:.4f}")
        
        print("\nMeasurements:")
        print("-" * 30)
        
        for name, measurement in results.measurements.items():
            print(f"{name.replace('_', ' ').title():20s}: {measurement.value:6.1f} {measurement.unit} "
                  f"(conf: {measurement.confidence:.1%})")
        
        print("\nMetadata:")
        print("-" * 30)
        print(f"Keypoints Detected: {results.metadata['keypoints_detected']}")
        print(f"Pose Quality: {results.metadata['pose_quality']}")
        
        print("="*50)
    
    def print_batch_summary(self, batch_results):
        """Print batch processing summary"""
        
        total = len(batch_results)
        successful = sum(1 for r in batch_results if r['success'])
        failed = total - successful
        
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        
        print(f"Total Images: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        if failed > 0:
            print("\nFailed Images:")
            print("-" * 30)
            for result in batch_results:
                if not result['success']:
                    image_name = Path(result['image_path']).name
                    print(f"  {image_name}: {result['error']}")
        
        if successful > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in batch_results if r['success']) / successful
            print(f"\nAverage Confidence: {avg_confidence:.1%}")
        
        print("="*50)
    
    def interactive_mode(self):
        """Interactive command-line mode"""
        
        print("\n" + "="*50)
        print("AI Body Measurement - Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  process <image_path> - Process a single image")
        print("  batch <input_dir> <output_dir> - Process multiple images")
        print("  help - Show this help")
        print("  quit - Exit")
        print("="*50)
        
        while True:
            try:
                command = input("\n>>> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif command.lower() == 'help':
                    print("Available commands:")
                    print("  process <image_path>")
                    print("  batch <input_dir> <output_dir>")
                    print("  quit")
                
                elif command.startswith('process '):
                    image_path = command[8:].strip().strip('"\'')
                    if Path(image_path).exists():
                        try:
                            result = self.process_image(image_path)
                            self.print_results(result)
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        print(f"Image not found: {image_path}")
                
                elif command.startswith('batch '):
                    parts = command[6:].strip().split()
                    if len(parts) >= 2:
                        input_dir, output_dir = parts[0], parts[1]
                        if Path(input_dir).exists():
                            try:
                                results = self.process_batch(input_dir, output_dir)
                                self.print_batch_summary(results)
                            except Exception as e:
                                print(f"Error: {e}")
                        else:
                            print(f"Input directory not found: {input_dir}")
                    else:
                        print("Usage: batch <input_dir> <output_dir>")
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")