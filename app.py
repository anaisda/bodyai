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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class AdvancedPromptEngine:
    """Creates precise, calibrated prompts that generate accurate measurements."""

    @staticmethod
    def create_expert_measurement_prompt(height, weight, gender, clothing_desc, camera_distance):
        height_m = float(height) / 100
        bmi = float(weight) / (height_m ** 2)

        if bmi < 18.5: 
            body_category = "underweight"
            depth_guidance = """
DEPTH ESTIMATION FOR UNDERWEIGHT BODY:
- Chest depth: 30-40% of chest width (less muscle/tissue)
- Waist depth: 25-35% of waist width (flat abdomen)
- Hip depth: 25-35% of hip width (minimal soft tissue)
"""
        elif 18.5 <= bmi < 25: 
            body_category = "normal_weight"
            depth_guidance = """
DEPTH ESTIMATION FOR NORMAL WEIGHT BODY:
- Chest depth: 35-45% of chest width
- Waist depth: 30-40% of waist width
- Hip depth: 30-40% of hip width
"""
        elif 25 <= bmi < 30: 
            body_category = "overweight"
            depth_guidance = """
DEPTH ESTIMATION FOR OVERWEIGHT BODY:
- Chest depth: 40-50% of chest width
- Waist depth: 35-50% of waist width (more tissue)
- Hip depth: 35-45% of hip width
"""
        else: 
            body_category = "obese"
            depth_guidance = """
DEPTH ESTIMATION FOR OBESE BODY:
- Chest depth: 45-55% of chest width
- Waist depth: 45-60% of waist width (significant tissue)
- Hip depth: 40-50% of hip width
"""

        return f"""You are a professional anthropometrist. Extract ACCURATE body measurements from these images.

SUBJECT SPECIFICATIONS:
Height: {height} cm ← PRIMARY SCALE REFERENCE (use this to calibrate everything)
Weight: {weight} kg
Gender: {gender}
BMI: {bmi:.1f} ({body_category})
Clothing: {clothing_desc}
Camera Distance: {camera_distance} meters

{depth_guidance}

CRITICAL MEASUREMENT LOCATIONS:

1. CHEST/BUST:
   - Location: At nipple line (4th intercostal space)
   - Width measurement: Widest point of chest/bust
   - This is NOT the shoulders, NOT the upper chest

2. WAIST:
   ⚠️ CRITICAL: Measure at NATURAL WAIST (narrowest point)
   - Location: Halfway between bottom of ribs and top of hip bones
   - This is typically 2-3 inches ABOVE the navel
   - DO NOT measure at belly button
   - DO NOT measure at fullest part of stomach
   - Natural waist is where the body naturally bends when leaning sideways

3. HIPS:
   - Location: At the widest point of buttocks
   - Typically at the level of greater trochanter
   - Width measurement includes hip bones and buttocks

MEASUREMENT PROTOCOL:

STEP 1: ESTABLISH SCALE
- Use the known height of {height} cm
- Calculate pixels per cm ratio
- Validate with multiple body landmarks

STEP 2: MEASURE WIDTHS (Front View)
- Chest width at nipple line
- Waist width at natural waist (narrowest point)
- Hip width at widest point

STEP 3: ESTIMATE DEPTHS (Side View)
Use the depth percentages above for {body_category} body type.
Be CONSERVATIVE - bodies are typically flatter than they appear in photos.

STEP 4: CALCULATE CIRCUMFERENCES
Formula: Circumference = π × (width + depth)

Apply this for each measurement.

STEP 5: VALIDATE PROPORTIONS
Sanity checks based on height {height} cm:
- Chest should be roughly {float(height) * 0.60:.0f}-{float(height) * 0.65:.0f} cm
- Waist should be roughly {float(height) * 0.55:.0f}-{float(height) * 0.62:.0f} cm  
- Hips should be roughly {float(height) * 0.55:.0f}-{float(height) * 0.65:.0f} cm

Expected proportions for {gender}:
- Waist should be 80-95% of chest (NOT larger than chest)
- Hips should be 85-110% of chest for males, 90-115% for females

COMMON ERRORS TO AVOID:
❌ Measuring waist at belly button → Use natural waist
❌ Overestimating depth → Be conservative with depth
❌ Confusing shoulders with chest → Chest is at nipple line
❌ Making waist larger than chest → Waist is always narrower

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis_metadata": {{
    "timestamp": "{datetime.datetime.now().isoformat()}",
    "camera_distance_m": {camera_distance},
    "scale_factor_cm_per_pixel": "CALCULATED_VALUE",
    "measurement_accuracy_confidence": "PERCENTAGE",
    "methodology": "calibrated_anatomical_landmarks"
  }},
  "subject_profile": {{
    "height_cm": {height},
    "weight_kg": {weight},
    "gender": "{gender}",
    "bmi": {bmi:.1f},
    "body_category": "{body_category}",
    "clothing_type": "{clothing_desc}"
  }},
  "measurements": {{
    "circumferences_cm": {{
      "chest_bust": {{
        "value": MEASURED_CIRCUMFERENCE,
        "visible_width_cm": MEASURED_WIDTH,
        "estimated_depth_cm": ESTIMATED_DEPTH,
        "depth_to_width_ratio": CALCULATED_RATIO,
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation",
        "landmark": "nipple_line_4th_intercostal"
      }},
      "waist": {{
        "value": MEASURED_CIRCUMFERENCE,
        "visible_width_cm": MEASURED_WIDTH,
        "estimated_depth_cm": ESTIMATED_DEPTH,
        "depth_to_width_ratio": CALCULATED_RATIO,
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation",
        "landmark": "natural_waist_narrowest_point"
      }},
      "hips": {{
        "value": MEASURED_CIRCUMFERENCE,
        "visible_width_cm": MEASURED_WIDTH,
        "estimated_depth_cm": ESTIMATED_DEPTH,
        "depth_to_width_ratio": CALCULATED_RATIO,
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation",
        "landmark": "greater_trochanter_widest_point"
      }}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "arm_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "leg_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "neck_circumference": {{"value": ESTIMATED, "confidence": "PERCENTAGE"}}
    }},
    "validation": {{
      "waist_to_chest_ratio": CALCULATED,
      "hip_to_chest_ratio": CALCULATED,
      "measurements_within_expected_range": true/false,
      "proportion_check_passed": true/false
    }}
  }},
  "quality_assessment": {{
    "image_quality": "excellent/good/fair/poor",
    "pose_accuracy": "excellent/good/fair/poor",
    "lighting_conditions": "excellent/good/fair/poor",
    "measurement_limitations": ["LIST_ANY_ISSUES"],
    "accuracy_notes": "ASSESSMENT"
  }}
}}

FINAL CHECKLIST BEFORE RESPONDING:
✓ Used height ({height} cm) for scale calibration
✓ Measured waist at NATURAL WAIST (narrowest point, NOT belly)
✓ Used appropriate depth percentages for {body_category} body
✓ Waist is smaller than chest
✓ All measurements are proportional to height
✓ Depth estimates are conservative and realistic

Provide ONLY the JSON object, no additional text."""


def encode_image_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def validate_image(file):
    """Validate uploaded image file."""
    if not file or file.filename == '':
        return False, "No file provided"
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return False, "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP"
    
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > 16 * 1024 * 1024:
        return False, "File too large. Maximum size: 16MB"
    
    try:
        Image.open(file).verify()
        file.seek(0)
        return True, "Valid image"
    except Exception:
        return False, "Invalid or corrupted image file"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Body Measurement API',
        'version': '4.0.0',
        'timestamp': datetime.datetime.now().isoformat(),
        'features': 'Intelligent prompt-based accuracy - no post-corrections needed'
    })


@app.route('/analyze', methods=['POST'])
def analyze_measurements():
    """Main API endpoint for body measurement analysis."""
    try:
        if 'multipart/form-data' not in request.content_type:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be multipart/form-data',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400

        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both front_image and side_image are required',
                'error_code': 'MISSING_IMAGES'
            }), 400

        front_file = request.files['front_image']
        side_file = request.files['side_image']

        is_valid, message = validate_image(front_file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Front image validation failed: {message}',
                'error_code': 'INVALID_FRONT_IMAGE'
            }), 400

        is_valid, message = validate_image(side_file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Side image validation failed: {message}',
                'error_code': 'INVALID_SIDE_IMAGE'
            }), 400

        required_params = ['height', 'weight', 'gender', 'clothing', 'camera_distance', 'api_key']
        params = {}
        
        for param in required_params:
            value = request.form.get(param)
            if not value:
                return jsonify({
                    'success': False,
                    'error': f'Missing required parameter: {param}',
                    'error_code': 'MISSING_PARAMETER'
                }), 400
            params[param] = value

        try:
            height = float(params['height'])
            weight = float(params['weight'])
            camera_distance = float(params['camera_distance'])
            
            if not (100 <= height <= 250):
                return jsonify({
                    'success': False,
                    'error': 'Height must be between 100-250 cm',
                    'error_code': 'INVALID_HEIGHT'
                }), 400
                
            if not (30 <= weight <= 300):
                return jsonify({
                    'success': False,
                    'error': 'Weight must be between 30-300 kg',
                    'error_code': 'INVALID_WEIGHT'
                }), 400
                
            if not (0.5 <= camera_distance <= 10):
                return jsonify({
                    'success': False,
                    'error': 'Camera distance must be between 0.5-10 meters',
                    'error_code': 'INVALID_CAMERA_DISTANCE'
                }), 400
                
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric values for height, weight, or camera_distance',
                'error_code': 'INVALID_NUMERIC_VALUES'
            }), 400

        if params['gender'].lower() not in ['male', 'female']:
            return jsonify({
                'success': False,
                'error': 'Gender must be either "male" or "female"',
                'error_code': 'INVALID_GENDER'
            }), 400

        timestamp = int(time.time())
        front_filename = secure_filename(f"front_{timestamp}_{front_file.filename}")
        side_filename = secure_filename(f"side_{timestamp}_{side_file.filename}")
        
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
        
        front_file.save(front_path)
        side_file.save(side_path)

        try:
            # Create intelligent prompt
            prompt = AdvancedPromptEngine.create_expert_measurement_prompt(
                params['height'],
                params['weight'],
                params['gender'],
                params['clothing'],
                params['camera_distance']
            )

            front_b64 = encode_image_base64(front_path)
            side_b64 = encode_image_base64(side_path)

            headers = {
                "Authorization": f"Bearer {params['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{front_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{side_b64}"}}
                    ]
                }],
                "max_tokens": 8192,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'AI API error: {response.text}',
                    'error_code': 'AI_API_ERROR',
                    'status_code': response.status_code
                }), 500

            try:
                result_json = response.json()['choices'][0]['message']['content']
                measurements = json.loads(result_json)
            except (KeyError, json.JSONDecodeError) as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to parse AI response: {str(e)}',
                    'error_code': 'AI_RESPONSE_PARSE_ERROR'
                }), 500

            # Success response - no corrections needed!
            return jsonify({
                'success': True,
                'data': measurements,
                'processing_info': {
                    'method': 'intelligent_prompt_engineering',
                    'corrections_applied': False,
                    'api_provider': 'groq_llama',
                    'api_version': '4.0.0',
                    'processing_time': datetime.datetime.now().isoformat(),
                    'note': 'Accurate measurements from optimized AI prompt - no post-processing corrections needed'
                }
            })

        finally:
            try:
                os.remove(front_path)
                os.remove(side_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_SERVER_ERROR'
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB per file.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'error_code': 'METHOD_NOT_ALLOWED'
    }), 405


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
