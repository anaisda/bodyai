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
    """Clean, simple instructions - let AI do its job."""
    
    return f"""Measure this person's body from the images.

REFERENCE:
Height: {height} cm - use this to calibrate your measurements

WHAT TO MEASURE:

1. CHEST/BUST
   - Location: At nipple level
   - Measure width from front view
   - Measure depth from side view
   - Calculate circumference

2. WAIST
   - Location: At narrowest point of torso (between ribs and hips)
   - NOT at belly button
   - Measure width from front view
   - Measure depth from side view
   - Calculate circumference

3. HIPS
   - Location: At widest part of buttocks
   - Measure width from front view
   - Measure depth from side view
   - Calculate circumference

4. SHOULDER WIDTH (front view)

5. ARM LENGTH (shoulder to wrist)

6. LEG LENGTH (hip to ankle)

RETURN JSON:
{{
  "measurements": {{
    "circumferences_cm": {{
      "chest_bust": {{"value": NUMBER, "confidence": "XX%"}},
      "waist": {{"value": NUMBER, "confidence": "XX%"}},
      "hips": {{"value": NUMBER, "confidence": "XX%"}}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": NUMBER, "confidence": "XX%"}},
      "arm_length": {{"value": NUMBER, "confidence": "XX%"}},
      "leg_length": {{"value": NUMBER, "confidence": "XX%"}},
      "neck_circumference": {{"value": NUMBER, "confidence": "XX%"}}
    }}
  }},
  "subject_profile": {{
    "height_cm": {height},
    "weight_kg": {weight},
    "gender": "{gender}",
    "clothing_type": "{clothing}"
  }}
}}"""


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def validate_image(file):
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
        'version': '6.0.0',
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
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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
                'version': '6.0.0'
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
