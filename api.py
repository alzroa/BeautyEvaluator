#!/usr/bin/env python3
"""
BeautyEvaluator REST API Server
Run with: python api.py
"""

import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from main import BeautyAnalyzer

app = Flask(__name__)
analyzer = BeautyAnalyzer()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_bytes(stream):
    """Convert file stream to numpy array."""
    bytes_data = stream.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'BeautyEvaluator API',
        'version': '2.0.0'
    })


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read and decode image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Save temporarily for analysis
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            temp_path = tmp.name
        
        try:
            # Run analysis
            landmarks, _ = analyzer.detect_face_landmarks(temp_path)
            
            if not landmarks:
                return jsonify({'error': 'No face detected'}), 400
            
            # Get all metrics
            symmetry = analyzer.calculate_symmetry(landmarks)
            golden_ratio = analyzer.calculate_golden_ratio(landmarks)
            eye_analysis = analyzer.analyze_eyes(landmarks)
            nose_analysis = analyzer.analyze_nose(landmarks)
            lip_analysis = analyzer.analyze_lips(landmarks)
            brow_analysis = analyzer.analyze_eyebrows(landmarks)
            body_props = analyzer.analyze_body_proportions(temp_path)
            
            # Calculate overall score
            body_score = 100 if body_props and 1.4 < body_props['shoulder_hip_ratio'] < 1.8 else 70
            feature_score = sum([
                eye_analysis['symmetry'] if eye_analysis else 70,
                nose_analysis['width_score'] if nose_analysis else 70,
                lip_analysis['symmetry'] if lip_analysis else 70,
                brow_analysis['symmetry'] if brow_analysis else 70
            ]) / 4
            
            beauty_score = symmetry * 0.35 + golden_ratio * 0.35 + feature_score * 0.30
            
            return jsonify({
                'success': True,
                'beauty_score': round(beauty_score, 1),
                'symmetry': symmetry,
                'golden_ratio': golden_ratio,
                'eyes': eye_analysis,
                'nose': nose_analysis,
                'lips': lip_analysis,
                'eyebrows': brow_analysis,
                'body_proportions': body_props,
                'filename': secure_filename(file.filename)
            })
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze image from URL."""
    data = request.get_json()
    
    if not data or 'image_url' not in data:
        return jsonify({'error': 'No image_url provided'}), 400
    
    import requests
    from PIL import Image
    from io import BytesIO
    
    try:
        # Download image
        response = requests.get(data['image_url'], timeout=10)
        response.raise_for_status()
        
        # Convert to OpenCV format
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img_array = np.array(img)
        image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Save temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            temp_path = tmp.name
        
        try:
            # Run analysis
            landmarks, _ = analyzer.detect_face_landmarks(temp_path)
            
            if not landmarks:
                return jsonify({'error': 'No face detected'}), 400
            
            symmetry = analyzer.calculate_symmetry(landmarks)
            golden_ratio = analyzer.calculate_golden_ratio(landmarks)
            eye_analysis = analyzer.analyze_eyes(landmarks)
            nose_analysis = analyzer.analyze_nose(landmarks)
            lip_analysis = analyzer.analyze_lips(landmarks)
            brow_analysis = analyzer.analyze_eyebrows(landmarks)
            body_props = analyzer.analyze_body_proportions(temp_path)
            
            feature_score = sum([
                eye_analysis['symmetry'] if eye_analysis else 70,
                nose_analysis['width_score'] if nose_analysis else 70,
                lip_analysis['symmetry'] if lip_analysis else 70,
                brow_analysis['symmetry'] if brow_analysis else 70
            ]) / 4
            
            beauty_score = symmetry * 0.35 + golden_ratio * 0.35 + feature_score * 0.30
            
            return jsonify({
                'success': True,
                'beauty_score': round(beauty_score, 1),
                'symmetry': symmetry,
                'golden_ratio': golden_ratio,
                'eyes': eye_analysis,
                'nose': nose_analysis,
                'lips': lip_analysis,
                'eyebrows': brow_analysis,
                'body_proportions': body_props,
                'source_url': data['image_url']
            })
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🎨 BeautyEvaluator API Server")
    print("📡 Running at http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /analyze    - Analyze uploaded image")
    print("  POST /analyze-url - Analyze image from URL")
    app.run(host='0.0.0.0', port=5000, debug=False)