# BeautyEvaluator

An open-source computer vision project to evaluate beauty based on the Marquardt Mask, facial symmetry, and body proportions.

![Beauty Analysis](https://img.shields.io/badge/python-3.8+-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.10+-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red)

## Features

- **Facial Landmark Detection** - Using MediaPipe Face Mesh (468 points)
- **Facial Symmetry Analysis** - Measures left-right facial asymmetry
- **Golden Ratio Harmony** - Calculates Marquardt beauty mask proportions
- **Body Proportions** - Shoulder-to-hip ratio using pose estimation
- **Marquardt Mask Overlay** - Visualize golden ratio grid on faces

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- opencv-python
- mediapipe
- numpy

## Usage

### Basic Analysis
```bash
python main.py path/to/photo.jpg
```

### With Overlay
```bash
python main.py path/to/photo.jpg --overlay
```

## Output Example

```
📷 Analyzing: selfie.jpg
✅ Face detected
⚖️  Symmetry Score: 87.5/100
🥇 Golden Ratio Harmony: 82.3/100
💪 Shoulder/Hip Ratio: 1.62

✨ Overall Beauty Score: 84.2/100
💾 Saved analysis to: selfie_analyzed.jpg

📊 Analysis Complete!
```

## Scoring Breakdown

| Metric | Weight | Description |
|--------|--------|-------------|
| Symmetry | 40% | Left-right facial balance |
| Golden Ratio | 40% | Marquardt mask proportions |
| Body Proportions | 20% | Shoulder-to-hip ratio |

## The Science

### Marquardt Beauty Mask
Based on Dr. Stephen Marquardt's research, the golden ratio (φ ≈ 1.618) defines aesthetically pleasing facial proportions.

### Symmetry
Facial symmetry is a strong predictor of perceived attractiveness. This analyzer measures deviation between left and right facial halves.

### Body Proportions
The ideal shoulder-to-hip ratio (≈1.618 for women, ≈1.2 for men) contributes to overall attractiveness assessment.

## License

MIT License - Feel free to use and modify!

## Author

Developed by ALZROA