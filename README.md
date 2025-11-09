# Body Measurement API

AI-powered body measurement extraction from images using computer vision and LLM analysis.

## Features

-  Extract body measurements from front and side view images
-  Conservative, body-type-aware corrections
-  Supports all body types (underweight, normal, overweight, obese)
-  Validates input with safety limits
-  Transparent correction logging

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python body_measurement_api_v2_complete.py
```

### API Usage

```bash
curl -X POST "http://localhost:5000/analyze" \
  -F "front_image=@front.jpg" \
  -F "side_image=@side.jpg" \
  -F "height=175" \
  -F "weight=70" \
  -F "gender=male" \
  -F "clothing=fitted shirt" \
  -F "camera_distance=2.5" \
  -F "api_key=YOUR_GROQ_API_KEY"
```

## API Endpoints

- **GET** `/health` - Health check
- **POST** `/analyze` - Extract body measurements

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| front_image | File | Yes | Front view image (JPG, PNG, etc.) |
| side_image | File | Yes | Side view image |
| height | String | Yes | Height in cm (100-250) |
| weight | String | Yes | Weight in kg (30-300) |
| gender | String | Yes | "male" or "female" |
| clothing | String | Yes | Clothing description |
| camera_distance | String | Yes | Distance in meters (0.5-10) |
| api_key | String | Yes | Groq API key |
| apply_corrections | String | No | "true" or "false" (default: "true") |

## Response Example

```json
{
  "success": true,
  "data": {
    "measurements": {
      "circumferences_cm": {
        "chest_bust": {"value": 95.2, "confidence": "88%"},
        "waist": {"value": 82.1, "confidence": "85%"},
        "hips": {"value": 89.7, "confidence": "82%"}
      }
    }
  }
}
```

## Requirements

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com))

## Image Guidelines

- **Resolution**: Minimum 800x600px
- **Format**: JPG, PNG, GIF, BMP, WEBP
- **Lighting**: Even, natural lighting
- **Pose**: Standing straight, arms slightly away from body
- **Camera**: 2.5-3.5m distance, chest level

## Deployment

### Render.com

1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn body_measurement_api_v2_complete:app`
4. Deploy

## Version

**2.0.0** - Conservative body-type-aware corrections

## Improvements in v2.0

-  Fixed aggressive hip measurement corrections
-  Added body-type awareness (underweight, normal, overweight, obese)
-  Soft validation (warnings instead of forced corrections)
-  Safety limits to prevent impossible measurements
-  Transparent correction logging

## License

MIT

## Author

**Anais Daoud**  
Master's Student in AI, ENSIA Algeria  
Specializing in AI integration and computer vision

---

For detailed documentation, see [IMPROVEMENTS.md](IMPROVEMENTS.md) and [DEPLOYMENT.md](DEPLOYMENT.md)
