# BeautyEvaluator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/MediaPipe-v0.10+-green" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-red" alt="OpenCV">
  <img src="https://img.shields.io/badge/Flask-2.3+-orange" alt="Flask">
</p>

An open-source computer vision project to evaluate beauty based on the Marquardt Mask, facial symmetry, and body proportions.

## Features

- **Facial Landmark Detection** - Using MediaPipe Face Mesh (468 points)
- **Facial Symmetry Analysis** - Measures left-right facial asymmetry
- **Golden Ratio Harmony** - Calculates Marquardt beauty mask proportions
- **Detailed Feature Analysis** - Eyes, nose, lips, eyebrows
- **Body Proportions** - Shoulder-to-hip ratio using pose estimation
- **Marquardt Mask Overlay** - Visualize golden ratio grid on faces
- **Real-time Webcam Mode** - Live beauty scoring
- **REST API Server** - Run as a web service
- **Batch Processing** - Analyze multiple images at once
- **JSON Export** - Programmatic results output

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- opencv-python
- mediapipe
- numpy
- flask

## Usage

### Basic Analysis
```bash
python main.py path/to/photo.jpg
```

### With Overlay
```bash
python main.py path/to/photo.jpg --overlay
```

### Export Results to JSON
```bash
python main.py photo.jpg --json results.json
```

### Webcam Realtime Mode
```bash
python main.py --webcam
```

### Start REST API Server
```bash
python api.py
# Server runs at http://localhost:5000
```

### Batch Processing
```bash
python batch.py ./photos/ --output results.json
```

## API Endpoints

When running `python api.py`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze image (multipart file) |
| `/analyze-url` | POST | Analyze image from URL |
| `/health` | GET | API health check |

### API Examples

```bash
# Analyze local file
curl -X POST -F "image=@selfie.jpg" http://localhost:5000/analyze

# Analyze from URL
curl -X POST -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}' \
  http://localhost:5000/analyze-url
```

## Output Example

```
📷 Analyzing: selfie.jpg
✅ Face detected
⚖️  Symmetry Score: 87.5/100
🥇 Golden Ratio Harmony: 82.3/100
👁️  Eye Symmetry: 89.2/100
👃 Nose Proportion Score: 78.5/100
👄 Lip Symmetry: 91.0/100
🩹 Eyebrow Symmetry: 85.4/100
💪 Shoulder/Hip Ratio: 1.62

✨ Overall Beauty Score: 84.2/100
💾 Saved analysis to: selfie_analyzed.jpg

📊 Analysis Complete!
```

## Scoring Breakdown

| Metric | Weight | Description |
|--------|--------|-------------|
| Symmetry | 35% | Left-right facial balance |
| Golden Ratio | 35% | Marquardt mask proportions |
| Feature Analysis | 30% | Eyes, nose, lips, eyebrows |

## The Science

### Marquardt Beauty Mask
Based on Dr. Stephen Marquardt's research, the golden ratio (φ ≈ 1.618) defines aesthetically pleasing facial proportions.

### Symmetry
Facial symmetry is a strong predictor of perceived attractiveness. This analyzer measures deviation between left and right facial halves.

### Body Proportions
The ideal shoulder-to-hip ratio (≈1.618 for women, ≈1.2 for men) contributes to overall attractiveness assessment.

## Project Structure

```
BeautyEvaluator/
├── main.py           # CLI analyzer
├── api.py            # REST API server
├── batch.py          # Batch processing
├── requirements.txt  # Dependencies
├── README.md         # This file
└── photos/           # Sample images (optional)
```

## License

MIT License - Feel free to use and modify!

## Author

Developed by ALZROA