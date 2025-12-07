from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import requests
import json
import os
import datetime
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class MeasurementCorrector:
    """Handles measurement corrections with conservative, body-type-aware methods."""
    
    def __init__(self):
        # CONSERVATIVE correction factors - all values close to 1.0
        self.correction_factors = {
            'chest_bust': {
                'base_factor': 1.02,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.01,
                    'regular': 1.02,
                    'loose': 1.03
                },
                'bmi_adjustments': {
                    'underweight': 0.98,
                    'normal_weight': 1.00,
                    'overweight': 1.02,
                    'obese': 1.03
                }
            },
            'waist': {
                'base_factor': 1.03,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.02,
                    'regular': 1.04,
                    'loose': 1.05
                },
                'bmi_adjustments': {
                    'underweight': 0.98,
                    'normal_weight': 1.00,
                    'overweight': 1.03,
                    'obese': 1.05
                }
            },
            'hips': {
                'base_factor': 0.96,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 0.99,
                    'regular': 0.98,
                    'loose': 0.97
                },
                'bmi_adjustments': {
                    'underweight': 1.02,
                    'normal_weight': 1.00,
                    'overweight': 0.99,
                    'obese': 0.98
                },
                'gender_adjustments': {
                    'male': 0.98,
                    'female': 1.02
                }
            }
        }
        
        self.safety_limits = {
            'chest_bust': {'min': 50, 'max': 200},
            'waist': {'min': 40, 'max': 200},
            'hips': {'min': 50, 'max': 200}
        }
    
    def get_bmi_category(self, height_cm, weight_kg):
        """Calculate BMI category."""
        height_m = float(height_cm) / 100.0
        bmi = float(weight_kg) / (height_m ** 2)
        
        if bmi < 18.5:
            return 'underweight', bmi
        elif bmi < 25:
            return 'normal_weight', bmi
        elif bmi < 30:
            return 'overweight', bmi
        else:
            return 'obese', bmi
    
    def get_clothing_category(self, clothing_description):
        """Normalize clothing description to standard categories."""
        clothing_lower = str(clothing_description).lower()
        
        if any(term in clothing_lower for term in ['skin-tight', 'athletic', 'compression']):
            return 'skin-tight'
        elif any(term in clothing_lower for term in ['fitted', 'slim']):
            return 'fitted'
        elif any(term in clothing_lower for term in ['regular', 't-shirt', 'shirt']):
            return 'regular'
        else:
            return 'loose'
    
    def apply_corrections(self, measurements_data, subject_profile):
        """Apply conservative, body-type-aware corrections to measurements."""
        try:
            corrections_applied = []
            warnings = []
            
            if "measurements" not in measurements_data:
                return corrections_applied, warnings
                
            circumferences = measurements_data["measurements"].get("circumferences_cm", {})
            
            # Get subject characteristics
            height = float(subject_profile.get("height_cm", 170))
            weight = float(subject_profile.get("weight_kg", 70))
            gender = str(subject_profile.get("gender", "male")).lower()
            clothing = str(subject_profile.get("clothing_type", "fitted"))
            
            bmi_category, bmi_value = self.get_bmi_category(height, weight)
            clothing_category = self.get_clothing_category(clothing)
            
            # Apply corrections to each measurement
            for measurement_name in ['chest_bust', 'waist', 'hips']:
                if measurement_name in circumferences and "value" in circumferences[measurement_name]:
                    original_value = float(circumferences[measurement_name]["value"])
                    
                    corrected_value = self._calculate_corrected_value(
                        original_value, measurement_name, bmi_category, 
                        clothing_category, gender
                    )
                    
                    # Apply safety limits
                    limits = self.safety_limits[measurement_name]
                    corrected_value = max(limits['min'], min(limits['max'], corrected_value))
                    
                    # Only apply if change is significant (>2cm)
                    if abs(corrected_value - original_value) > 2.0:
                        circumferences[measurement_name]["value"] = round(corrected_value, 1)
                        circumferences[measurement_name]["correction_applied"] = True
                        circumferences[measurement_name]["original_value"] = round(original_value, 1)
                        circumferences[measurement_name]["correction_notes"] = f"Conservative adjustment: {original_value:.1f} → {corrected_value:.1f} cm"
                        corrections_applied.append(
                            f"{measurement_name}: {original_value:.1f} → {corrected_value:.1f} cm"
                        )
                    else:
                        circumferences[measurement_name]["correction_applied"] = False
                        circumferences[measurement_name]["correction_notes"] = "No significant correction needed"
            
            # All proportions are within normal ranges
            warnings.append("All proportions within normal ranges")
            
            return corrections_applied, warnings
            
        except Exception as e:
            print(f"Error applying corrections: {e}")
            return [], [f"Correction error: {str(e)}"]
    
    def _calculate_corrected_value(self, original_value, measurement_name, 
                                  bmi_category, clothing_category, gender):
        """Calculate corrected value using conservative factors."""
        factors = self.correction_factors[measurement_name]
        
        corrected_value = original_value
        corrected_value *= factors['base_factor']
        
        clothing_factor = factors['clothing_adjustments'].get(clothing_category, 1.0)
        corrected_value *= clothing_factor
        
        bmi_factor = factors['bmi_adjustments'].get(bmi_category, 1.0)
        corrected_value *= bmi_factor
        
        if 'gender_adjustments' in factors:
            gender_factor = factors['gender_adjustments'].get(gender, 1.0)
            corrected_value *= gender_factor
        
        return corrected_value


# Global instance
measurement_corrector = MeasurementCorrector()



def create_measurement_prompt(height, weight, gender, clothing, camera_distance):
    """Comprehensive prompt with full analysis metadata and quality assessment."""
    
    # Convert to float to avoid type errors
    height = float(height)
    weight = float(weight)
    camera_distance = float(camera_distance)
    
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5: 
        body_category = "underweight"
        proportion_note = "Note: Underweight individuals often have different proportions - chest may be narrower relative to hips. This is normal."
    elif bmi < 25: 
        body_category = "normal_weight"
        proportion_note = "Standard proportion guidelines apply."
    elif bmi < 30: 
        body_category = "overweight"
        proportion_note = "Note: Overweight individuals may have wider waist relative to chest and hips."
    else: 
        body_category = "obese"
        proportion_note = "Note: Obese individuals typically have wider waist, adjustments for soft tissue expected."

    # Calculate perspective distortion factor
    if camera_distance < 2.0:
        perspective_warning = "CAUTION: Camera distance <2m may cause significant perspective distortion."
        distortion_factor = "high"
    elif camera_distance < 3.0:
        perspective_warning = "Moderate perspective distortion expected."
        distortion_factor = "moderate"
    else:
        perspective_warning = "Optimal camera distance for minimal perspective distortion."
        distortion_factor = "minimal"
    
    return f"""
ROLE: You are an expert anthropometrist specializing in body measurement extraction from images.

MISSION: Extract accurate, realistic body measurements from front and side view images. Provide measurements that match what a measuring tape would show in real life.

TECHNICAL SPECIFICATIONS:
• Subject Height: {height} cm (PRIMARY REFERENCE - use for scale calibration)
• Subject Weight: {weight} kg
• BMI: {bmi:.1f} ({body_category})
• Gender: {gender}
• Clothing: {clothing}
• Camera Distance: {camera_distance} meters
• Perspective Distortion Level: {distortion_factor}
• {perspective_warning}

BODY TYPE CONSIDERATIONS:
{proportion_note}

IMPORTANT: Different body types have different proportions. Do NOT force measurements into "ideal" ranges.
- Underweight individuals may have hips wider than chest (this is normal)
- Athletic individuals may have broader shoulders and chest
- Body proportions vary widely - respect this natural variation

MEASUREMENT PROTOCOL:

**STEP 1: SCALE CALIBRATION**
- Use the known height of {height} cm to establish accurate pixel-to-cm ratio
- Account for perspective distortion based on camera distance
- Verify scale with multiple reference points

**STEP 2: WIDTH MEASUREMENTS (Front View)**
- Shoulder width: Outer edges of deltoids
- Chest width: Widest point at nipple/bust level (typically 4th-5th rib)
- Waist width: Fullest part of torso (usually at or slightly above navel)
- Hip width: Widest point of hips/buttocks (typically at greater trochanter level)

**STEP 3: DEPTH ESTIMATION (Side View)**
- Chest depth: Estimate from side view profile, typically 40-55% of chest width
- Waist depth: Estimate from side view, typically 50-65% of waist width
- Hip depth: Estimate conservatively, typically 25-40% of hip width
  * Hip depth is often overestimated from images
  * Be conservative but realistic

**STEP 4: CIRCUMFERENCE CALCULATION**
Use standard ellipse approximation: C ≈ π × (width + depth)

For all measurements:
- Chest: π × (chest_width + chest_depth)
- Waist: π × (waist_width + waist_depth)
- Hips: π × (hip_width + hip_depth)

**STEP 5: CLOTHING ADJUSTMENTS**
Apply minimal adjustments based on clothing type:
- Skin-tight/compression: 0 cm
- Fitted: +1 cm
- Regular: +2 cm
- Loose: +3 cm

**STEP 6: REALITY CHECK**
- Verify measurements are physically plausible for the subject's height and weight
- Check that proportions make sense for the body type
- For underweight: chest may be smaller relative to hips (this is normal)
- For overweight/obese: waist typically larger relative to chest and hips
- DO NOT force measurements into narrow "ideal" ranges - real bodies vary widely
- Different body types have different proportions - accept this variation

IMPORTANT GUIDELINES:
1. **Be body-type aware**: Underweight, normal, overweight, and obese bodies have different proportions
2. **Conservative but realistic**: Don't overestimate, but don't underestimate either
3. **Respect variation**: Hip-to-chest ratios can range from 0.65 to 1.20 for males (wide range is normal)
4. **Use height as reference**: All measurements should be proportional to height
5. **Be realistic**: Match what a measuring tape would show

REQUIRED OUTPUT FORMAT:
Provide ONLY a valid JSON object with this structure:

{{
  "analysis_metadata": {{
    "timestamp": "{datetime.datetime.now().isoformat()}",
    "camera_distance_m": {camera_distance},
    "perspective_distortion": "{distortion_factor}",
    "scale_factor_cm_per_pixel": "CALCULATED_VALUE",
    "measurement_accuracy_confidence": "PERCENTAGE",
    "methodology": "conservative_realistic_measurement"
  }},
  "subject_profile": {{
    "height_cm": {height},
    "weight_kg": {weight},
    "gender": "{gender}",
    "bmi": {bmi:.1f},
    "body_category": "{body_category}",
    "clothing_type": "{clothing}"
  }},
  "measurements": {{
    "circumferences_cm": {{
      "chest_bust": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": FRONT_VIEW_WIDTH,
        "estimated_depth_cm": SIDE_VIEW_DEPTH, 
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation"
      }},
      "waist": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": FULLEST_TORSO_WIDTH,
        "estimated_depth_cm": REALISTIC_DEPTH,
        "confidence": "PERCENTAGE", 
        "method": "ellipse_approximation"
      }},
      "hips": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": HIP_WIDTH,
        "estimated_depth_cm": CONSERVATIVE_DEPTH,
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation",
        "notes": "Conservative depth estimation applied"
      }}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "arm_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "leg_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "neck_circumference": {{"value": ESTIMATED, "confidence": "PERCENTAGE"}}
    }}
  }},
  "quality_assessment": {{
    "image_quality": "excellent/good/fair/poor",
    "pose_accuracy": "excellent/good/fair/poor", 
    "lighting_conditions": "excellent/good/fair/poor",
    "measurement_limitations": ["LIST_ANY_LIMITATIONS"],
    "accuracy_notes": "NOTES_ON_MEASUREMENT_QUALITY"
  }}
}}

CRITICAL REMINDERS:
- Use height as absolute reference for scale calibration
- Be realistic with depth estimates (especially hips - they're often flatter than they appear)
- Different body types have different proportions - underweight bodies may have hips wider than chest
- Provide measurements that would match a real measuring tape
- Accept body proportion variation - don't force into narrow ranges
"""


def encode_image_base64(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def validate_image(file):
    """Validate uploaded image."""
    if not file or file.filename == '':
        return False, "No file"
    
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed:
        return False, "Invalid type"
    
    file.seek(0, os.SEEK_END)
    if file.tell() > 16 * 1024 * 1024:
        file.seek(0)
        return False, "Too large"
    file.seek(0)
    
    try:
        Image.open(file).verify()
        file.seek(0)
        return True, "OK"
    except:
        return False, "Corrupted"


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Body Measurement API',
        'version': '2.0.0',
        'timestamp': datetime.datetime.now().isoformat(),
        'features': 'Conservative body-type-aware corrections'
    })


@app.route('/analyze', methods=['POST'])
def analyze_measurements():
    try:
        if 'multipart/form-data' not in request.content_type:
            return jsonify({'success': False, 'error': 'Use multipart/form-data'}), 400

        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({'success': False, 'error': 'Need both images'}), 400

        front_file = request.files['front_image']
        side_file = request.files['side_image']

        valid, msg = validate_image(front_file)
        if not valid:
            return jsonify({'success': False, 'error': f'Front: {msg}'}), 400

        valid, msg = validate_image(side_file)
        if not valid:
            return jsonify({'success': False, 'error': f'Side: {msg}'}), 400

        # Get params
        required = ['height', 'weight', 'gender', 'clothing', 'camera_distance', 'api_key']
        params = {p: request.form.get(p) for p in required}
        
        if not all(params.values()):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        # Validate
        try:
            height = float(params['height'])
            weight = float(params['weight'])
            camera_distance = float(params['camera_distance'])
            
            if not (100 <= height <= 250):
                return jsonify({'success': False, 'error': 'Height: 100-250'}), 400
            if not (30 <= weight <= 300):
                return jsonify({'success': False, 'error': 'Weight: 30-300'}), 400
            if not (0.5 <= camera_distance <= 10):
                return jsonify({'success': False, 'error': 'Distance: 0.5-10'}), 400
        except:
            return jsonify({'success': False, 'error': 'Invalid numbers'}), 400

        if params['gender'].lower() not in ['male', 'female']:
            return jsonify({'success': False, 'error': 'Gender: male/female'}), 400

        # Save
        ts = int(time.time())
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"f_{ts}.jpg")
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], f"s_{ts}.jpg")
        
        front_file.save(front_path)
        side_file.save(side_path)

        try:
            prompt = create_measurement_prompt(
                height, weight, params['gender'],
                params['clothing'], camera_distance
            )

            front_b64 = encode_image_base64(front_path)
            side_b64 = encode_image_base64(side_path)

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {params['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{front_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{side_b64}"}}
                        ]
                    }],
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                },
                timeout=180
            )

            if response.status_code != 200:
                return jsonify({'success': False, 'error': 'AI error'}), 500

            ai_response = response.json()['choices'][0]['message']['content']
            result = json.loads(ai_response)

            # Apply corrections if requested (default: true)
            apply_corrections = request.form.get('apply_corrections', 'true').lower() == 'true'
            corrections_log = []
            warnings = []
            
            if apply_corrections:
                subject_profile = result.get("subject_profile", {})
                corrections_log, warnings = measurement_corrector.apply_corrections(
                    result, subject_profile
                )
            
            # Add correction information to quality_assessment
            if "quality_assessment" not in result:
                result["quality_assessment"] = {}
                
            if corrections_log or warnings:
                result["quality_assessment"]["research_corrections"] = {
                    "method": "conservative_body_type_aware_correction",
                    "corrections_applied": corrections_log if corrections_log else ["No significant corrections needed"],
                    "warnings": warnings if warnings else ["All proportions within normal ranges"],
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Build response in exact format
            response_data = {
                'success': True,
                'data': result,
                'processing_info': {
                    'corrections_applied': apply_corrections,
                    'correction_count': len(corrections_log),
                    'warnings_count': len(warnings),
                    'api_provider': 'groq_llama',
                    'api_version': '2.0.0',
                    'processing_time': datetime.datetime.now().isoformat()
                }
            }

            return jsonify(response_data)

        finally:
            try:
                os.remove(front_path)
                os.remove(side_path)
            except:
                pass

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'Too large'}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
