#!/usr/bin/env python3
"""
BeautyEvaluator - Analyze beauty based on Marquardt Mask, facial symmetry, and body proportions.
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
from pathlib import Path

# Constants
MARQUARDT_MASK_PATH = Path(__file__).parent / "marquardt_mask.png"

class BeautyAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Key facial landmarks indices (MediaPipe 468-point mesh)
        self.JAWLINE = list(range(0, 17))
        self.EYEBROW_LEFT = list(range(17, 22))
        self.EYEBROW_RIGHT = list(range(22, 27))
        self.NOSE = list(range(27, 36))
        self.EYE_LEFT = list(range(36, 42))
        self.EYE_RIGHT = list(range(42, 48))
        self.MOUTH = list(range(48, 61))
        
        # Golden ratio points (essential for beauty calculation)
        self.GOLDEN_POINTS = {
            'nose_tip': 1,  # Approximate
            'chin': 152,
            'left_eye_corner': 33,
            'right_eye_corner': 263,
            'mouth_center': 13,
        }
    
    def detect_face_landmarks(self, image_path):
        """Detect facial landmarks from an image."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, image
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return landmarks_px, image
    
    def calculate_symmetry(self, landmarks):
        """Calculate facial symmetry score (0-100)."""
        if not landmarks:
            return 0
        
        # Mirror points across facial center
        left_points = [
            landmarks[i] for i in self.EYE_LEFT + self.JAWLINE[:8]
        ]
        right_points = [
            landmarks[i] for i in self.EYE_RIGHT + self.JAWLINE[8:]
        ]
        
        # Calculate center of face
        center_x = sum(l[0] for l in landmarks) / len(landmarks)
        
        # Measure asymmetry
        asymmetries = []
        for lp, rp in zip(left_points, reversed(right_points)):
            mirrored_x = 2 * center_x - rp[0]
            asymmetry = abs(lp[0] - mirrored_x)
            asymmetries.append(asymmetry)
        
        avg_asymmetry = np.mean(asymmetries) if asymmetries else 0
        symmetry_score = max(0, 100 - (avg_asymmetry / 2))
        
        return round(symmetry_score, 1)
    
    def calculate_golden_ratio(self, landmarks):
        """Calculate golden ratio harmony score."""
        if not landmarks or len(landmarks) < 200:
            return 0
        
        # Key distances for golden ratio analysis
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Face height segments (Marquardt divisions)
        forehead_top = landmarks[10]    # Top of forehead
        eyebrow_top = landmarks[107]   # Above eyebrows
        nose_bridge = landmarks[6]     # Between eyebrows
        nose_tip = landmarks[1]        # Nose tip
        chin = landmarks[152]          # Chin
        
        # Calculate vertical ratios
        face_height = distance(forehead_top, chin)
        upper_face = distance(forehead_top, nose_bridge)
        middle_face = distance(nose_bridge, nose_tip)
        lower_face = distance(nose_tip, chin)
        
        # Golden ratio = 1.618
        phi = 1.618
        
        ratios = []
        if upper_face > 0:
            ratios.append(abs(face_height / upper_face - phi) / phi)
        if middle_face > 0:
            ratios.append(abs(lower_face / middle_face - phi) / phi)
        
        harmony_score = max(0, 100 - (np.mean(ratios) * 100)) if ratios else 0
        return round(harmony_score, 1)
    
    def analyze_body_proportions(self, image_path):
        """Analyze body proportions using pose estimation."""
        image = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        lm = results.pose_landmarks.landmark
        
        # Key body points
        shoulders = (lm[11], lm[12])  # Left, Right shoulder
        hips = (lm[23], lm[24])        # Left, Right hip
        nose = lm[0]
        
        # Calculate body proportions
        def get_xy(landmark):
            return int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        
        shoulder_width = abs(get_xy(shoulders[0])[0] - get_xy(shoulders[1])[0])
        hip_width = abs(get_xy(hips[0])[0] - get_xy(hips[1])[0])
        
        # Shoulder to hip ratio (ideal ≈ 1.618 for hourglass)
        if hip_width > 0:
            shoulder_hip_ratio = shoulder_width / hip_width
        else:
            shoulder_hip_ratio = 1
        
        # Upper body length (shoulder to hip)
        shoulder_y = (get_xy(shoulders[0])[1] + get_xy(shoulders[1])[1]) // 2
        hip_y = (get_xy(hips[0])[1] + get_xy(hips[1])[1]) // 2
        upper_body_length = abs(hip_y - shoulder_y)
        
        return {
            'shoulder_hip_ratio': round(shoulder_hip_ratio, 2),
            'upper_body_length': upper_body_length,
            'shoulder_width': shoulder_width
        }
    
    def draw_marquardt_mask_overlay(self, image, landmarks):
        """Draw golden ratio grid overlay (simplified Marquardt mask)."""
        if not landmarks or len(landmarks) < 468:
            return image
        
        output = image.copy()
        h, w = image.shape[:2]
        
        # Draw facial outline
        jaw_points = [landmarks[i] for i in self.JAWLINE]
        jaw_points.append(jaw_points[0])  # Close the loop
        cv2.polylines(output, [np.array(jaw_points)], False, (0, 255, 0), 2)
        
        # Draw golden ratio vertical lines
        center_x = landmarks[152][0]  # Chin x
        
        # Golden divisions
        for i in range(1, 4):
            x = int(center_x + (i * w * 0.05))
            cv2.line(output, (x, 0), (x, h), (255, 0, 255), 1)
            x = int(center_x - (i * w * 0.05))
            cv2.line(output, (x, 0), (x, h), (255, 0, 255), 1)
        
        return output
    
    def analyze(self, image_path, draw_overlay=False):
        """Run full beauty analysis on an image."""
        print(f"📷 Analyzing: {image_path}")
        
        # Detect landmarks
        landmarks, image = self.detect_face_landmarks(image_path)
        
        if not landmarks:
            print("❌ No face detected in image")
            return None
        
        print("✅ Face detected")
        
        # Calculate scores
        symmetry = self.calculate_symmetry(landmarks)
        print(f"⚖️  Symmetry Score: {symmetry}/100")
        
        golden_ratio = self.calculate_golden_ratio(landmarks)
        print(f"🥇 Golden Ratio Harmony: {golden_ratio}/100")
        
        # Body proportions (if full body in frame)
        body_props = self.analyze_body_proportions(image_path)
        if body_props:
            print(f"💪 Shoulder/Hip Ratio: {body_props['shoulder_hip_ratio']}")
        
        # Combined beauty score
        beauty_score = (symmetry * 0.4 + golden_ratio * 0.4 + 
                       (100 if body_props and 1.4 < body_props['shoulder_hip_ratio'] < 1.8 else 70) * 0.2)
        
        print(f"\n✨ Overall Beauty Score: {round(beauty_score, 1)}/100")
        
        # Draw overlay if requested
        if draw_overlay:
            output_path = Path(image_path).stem + "_analyzed.jpg"
            output = self.draw_marquardt_mask_overlay(image, landmarks)
            cv2.imwrite(str(output_path), output)
            print(f"💾 Saved analysis to: {output_path}")
        
        return {
            'symmetry': symmetry,
            'golden_ratio': golden_ratio,
            'body_proportions': body_props,
            'beauty_score': round(beauty_score, 1)
        }


def main():
    parser = argparse.ArgumentParser(description="BeautyEvaluator - AI Beauty Analysis")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--overlay", "-o", action="store_true", 
                       help="Save image with Marquardt mask overlay")
    parser.add_argument("--draw-landmarks", "-d", action="store_true",
                       help="Draw facial landmarks on output")
    
    args = parser.parse_args()
    
    analyzer = BeautyAnalyzer()
    result = analyzer.analyze(args.image, args.overlay)
    
    if result:
        print("\n📊 Analysis Complete!")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())