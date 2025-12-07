""" meta-llama/llama-4-maverick-17b-128e-instruct
Body Measurement API - Production Version
Using exact prompt structure from user
"""

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


def create_measurement_prompt(height, weight, gender, clothing, camera_distance):
    """User's exact prompt - algorithmic and precise."""
    
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5: 
        body_category = "underweight"
    elif bmi < 25: 
        body_category = "normal_weight"
    elif bmi < 30: 
        body_category = "overweight"
    else: 
        body_category = "obese"
    
    return f"""Extract body measurements using this EXACT algorithm and return results in JSON format.

SUBJECT: {gender}, {height} cm tall, {weight} kg

STEP 1: CALIBRATION
─────────────────────
1.1. Locate the TOP of the head in front view
1.2. Locate the BOTTOM of the feet in front view  
1.3. Measure pixel distance from head to feet
1.4. Calculate: pixels_per_cm = total_pixels / {height}
1.5. Verify: measure shoulder-to-hip distance and check if proportional

STEP 2: LOCATE MEASUREMENT POINTS
──────────────────────────────────
Use these EXACT anatomical landmarks:

2.1. CHEST/BUST LEVEL:
   - Find the nipples in front view (or equivalent horizontal line for females)
   - Mark this horizontal line across the torso
   - This is your chest measurement line

2.2. WAIST LEVEL:
   - Find the belly button (navel) in front view
   - Mark this horizontal line across the torso
   - This is your waist measurement line
   
2.3. HIP LEVEL (HIGH HIP - for clothing):
Note that Hip depth at hip bone level is MUCH SMALLER than at buttocks . Find backmost body edge AT THE SAME HEIGHT
 Do NOT include buttocks below - they are below this level
   - Find the top of the hip bones (iliac crest) in side view
   - This is below the belly button
   - Mark this horizontal line
   - This is your hip measurement line (NOT the widest buttocks point)

STEP 3: MEASURE WIDTHS (Front View)
────────────────────────────────────
For each measurement line:

3.1. CHEST WIDTH:
   - At chest level line, find leftmost body edge
   - Find rightmost body edge
   - Measure pixel distance between them
   - Convert: chest_width_cm = pixels × pixels_per_cm

3.2. WAIST WIDTH:
   - At waist level line (belly button), find leftmost body edge
   - Find rightmost body edge  
   - Measure pixel distance between them
   - Convert: waist_width_cm = pixels × pixels_per_cm

3.3. HIP WIDTH:
   - At hip level line, find leftmost body edge
   - Find rightmost body edge
   - Measure pixel distance between them
   - Convert: hip_width_cm = pixels × pixels_per_cm

STEP 4: MEASURE DEPTHS (Side View)
───────────────────────────────────
For each measurement line:

4.1. CHEST DEPTH:
   - At chest level line, find frontmost body edge (chest)
   - Find backmost body edge (back)
   - Measure pixel distance between them
   - Convert: chest_depth_cm = pixels × pixels_per_cm

4.2. WAIST DEPTH:
   - At waist level line (belly button), find frontmost body edge (belly)
   - Find backmost body edge (lower back)
   - Measure pixel distance between them
   - Convert: waist_depth_cm = pixels × pixels_per_cm

4.3. HIP DEPTH:
   - At hip level line (hip bones, NOT buttocks), find frontmost body edge
   - Find backmost body edge
   - Measure pixel distance between them
   - Convert: hip_depth_cm = pixels × pixels_per_cm

STEP 5: CALCULATE CIRCUMFERENCES
─────────────────────────────────
Use ellipse formula: C = π × (width + depth)

5.1. chest_circumference = 3.14159 × (chest_width_cm + chest_depth_cm)
5.2. waist_circumference = 3.14159 × (waist_width_cm + waist_depth_cm)  
5.3. hip_circumference = 3.14159 × (hip_width_cm + hip_depth_cm)


STEP 7: LINEAR MEASUREMENTS
────────────────────────────
7.1. Shoulder width: distance between left and right shoulder points (front view)
7.2. Arm length: shoulder to wrist (side view)  
7.3. Leg length: hip to ankle (side or front view)
7.4. Neck: estimate circumference at base of neck

RETURN FORMAT (exact structure):
{{
  "measurements": {{
    "circumferences_cm": {{
      "chest_bust": {{
        "value": CALCULATED_FROM_STEP_5.1,
        "visible_width_cm": "FROM_STEP_3.1",
        "estimated_depth_cm": "FROM_STEP_4.1",
        "confidence": "90%"
      }},
      "waist": {{
        "value": CALCULATED_FROM_STEP_5.2,
        "visible_width_cm": "FROM_STEP_3.2",
        "estimated_depth_cm": "FROM_STEP_4.2",
        "confidence": "90%"
      }},
      "hips": {{
        "value": CALCULATED_FROM_STEP_5.3,
        "visible_width_cm": "FROM_STEP_3.3",
        "estimated_depth_cm": "FROM_STEP_4.3",
        "confidence": "85%"
      }}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": FROM_STEP_7.1, "confidence": "85%"}},
      "arm_length": {{"value": FROM_STEP_7.2, "confidence": "80%"}},
      "leg_length": {{"value": FROM_STEP_7.3, "confidence": "85%"}},
      "neck_circumference": {{"value": FROM_STEP_7.4, "confidence": "75%"}}
    }}
  }},
  "subject_profile": {{
    "height_cm": {height},
    "weight_kg": {weight},
    "gender": "{gender}",
    "bmi": {bmi:.1f},
    "body_category": "{body_category}"
  }},
  "calibration": {{
    "pixels_per_cm": FROM_STEP_1.4,
    "chest_level_pixel": PIXEL_Y_COORDINATE,
    "waist_level_pixel": PIXEL_Y_COORDINATE,
    "hip_level_pixel": PIXEL_Y_COORDINATE
  }}
}}

CRITICAL: Follow every step in exact order. Use the same measurement points every time for consistency."""


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
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
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
                params['height'], params['weight'], params['gender'],
                params['clothing'], params['camera_distance']
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

            result = json.loads(response.json()['choices'][0]['message']['content'])

            return jsonify({
                'success': True,
                'data': result,
                'processing_info': {
                    'corrections_applied': False,
                    'api_provider': 'groq_llama',
                    'processing_time': datetime.datetime.now().isoformat()
                }
            })

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
