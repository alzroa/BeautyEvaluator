#!/usr/bin/env python3
"""
BeautyEvaluator - Analyze beauty based on Marquardt Mask, facial symmetry, and body proportions.
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Constants
MARQUARDT_MASK_PATH = Path(__file__).parent / "marquardt_mask.png"
PHI = 1.618  # Golden ratio


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
        self.MOUTH = list(range(48, 68))  # Extended to include full lips
        self.LIP_OUTER = list(range(61, 68))
        self.LIP_INNER = list(range(61, 68))
        
        # Pupils for eye analysis
        self.PUPIL_LEFT = 468  # Refined landmark
        self.PUPIL_RIGHT = 473
        
        # Golden ratio points
        self.GOLDEN_POINTS = {
            'nose_tip': 1,
            'chin': 152,
            'left_eye_corner': 33,
            'right_eye_corner': 263,
            'mouth_center': 13,
            'forehead_top': 10,
            'nose_bridge': 6,
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
        
        center_x = sum(l[0] for l in landmarks) / len(landmarks)
        
        asymmetries = []
        for lp, rp in zip(left_points, reversed(right_points)):
            mirrored_x = 2 * center_x - rp[0]
            asymmetry = abs(lp[0] - mirrored_x)
            asymmetries.append(asymmetry)
        
        avg_asymmetry = np.mean(asymmetries) if asymmetries else 0
        symmetry_score = max(0, 100 - (avg_asymmetry / 2))
        
        return round(symmetry_score, 1)
    
    def analyze_eyes(self, landmarks):
        """Analyze eye aesthetics."""
        if not landmarks or len(landmarks) < 48:
            return None
        
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Eye landmarks
        left_eye = [landmarks[i] for i in self.EYE_LEFT]
        right_eye = [landmarks[i] for i in self.EYE_RIGHT]
        
        # Eye dimensions
        left_width = distance(left_eye[0], left_eye[3])
        left_height = distance(left_eye[1], left_eye[5])
        right_width = distance(right_eye[0], right_eye[3])
        right_height = distance(right_eye[1], right_eye[5])
        
        # Eye aspect ratio (should be ~0.3 for open eye)
        def eye_aspect_ratio(eye):
            v1 = distance(eye[1], eye[5])
            v2 = distance(eye[2], eye[4])
            h = distance(eye[0], eye[3])
            if h == 0:
                return 0
            return (v1 + v2) / (2 * h)
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Eye symmetry
        eye_symmetry = 100 - abs(left_ear - right_ear) * 100
        
        # Eye size ratio (larger eyes are generally perceived as more attractive)
        avg_width = (left_width + right_width) / 2
        avg_height = (left_height + right_height) / 2
        eye_proportion = avg_height / avg_width if avg_width > 0 else 0
        
        # Ideal eye proportion ~0.35-0.4
        eye_score = max(0, min(100, (1 - abs(eye_proportion - 0.37) / 0.37) * 100))
        
        return {
            'symmetry': round(eye_symmetry, 1),
            'proportion_score': round(eye_score, 1),
            'left_ear': round(left_ear, 3),
            'right_ear': round(right_ear, 3),
            'is_open': left_ear > 0.25 and right_ear > 0.25
        }
    
    def analyze_nose(self, landmarks):
        """Analyze nose proportions."""
        if not landmarks or len(landmarks) < 36:
            return None
        
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        nose_points = [landmarks[i] for i in self.NOSE]
        
        # Nose width at nostrils
        nostril_width = distance(nose_points[3], nose_points[4])
        
        # Nose length (bridge)
        nose_length = distance(nose_points[0], nose_points[3])
        
        # Nose to face width ratio
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_width = distance(left_cheek, right_cheek)
        
        if face_width > 0:
            nose_ratio = nostril_width / face_width
        else:
            nose_ratio = 0
        
        # Ideal nose width ~1/5 of face width
        nose_score = max(0, min(100, (1 - abs(nose_ratio - 0.2) / 0.2) * 100))
        
        return {
            'width_score': round(nose_score, 1),
            'nostril_width': round(nostril_width, 1),
            'nose_length': round(nose_length, 1),
            'nose_ratio': round(nose_ratio, 3)
        }
    
    def analyze_lips(self, landmarks):
        """Analyze lip aesthetics."""
        if not landmarks or len(landmarks) < 68:
            return None
        
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Upper and lower lip
        upper_lip = [landmarks[i] for i in range(61, 65)]
        lower_lip = [landmarks[i] for i in range(65, 68)]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        
        lip_height = distance(mouth_top, mouth_bottom)
        mouth_width = distance(landmarks[61], landmarks[65])
        
        # Lip ratio (lower lip should be slightly fuller)
        if mouth_width > 0:
            lip_ratio = lip_height / mouth_width
        else:
            lip_ratio = 0
        
        # Ideal lip ratio ~0.4-0.5
        lip_score = max(0, min(100, (1 - abs(lip_ratio - 0.45) / 0.45) * 100))
        
        # Lip symmetry
        lip_center_x = (landmarks[61][0] + landmarks[65][0]) / 2
        left_lip_dev = abs(landmarks[62][0] - lip_center_x)
        right_lip_dev = abs(landmarks[64][0] - lip_center_x)
        lip_symmetry = max(0, 100 - abs(left_lip_dev - right_lip_dev))
        
        return {
            'ratio_score': round(lip_score, 1),
            'symmetry': round(lip_symmetry, 1),
            'lip_ratio': round(lip_ratio, 3),
            'width': round(mouth_width, 1),
            'height': round(lip_height, 1)
        }
    
    def analyze_eyebrows(self, landmarks):
        """Analyze eyebrow shape and position."""
        if not landmarks or len(landmarks) < 68:
            return None
        
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        left_brow = [landmarks[i] for i in self.EYEBROW_LEFT]
        right_brow = [landmarks[i] for i in self.EYEBROW_RIGHT]
        
        # Eyebrow thickness (average)
        left_thickness = distance(left_brow[1], left_brow[4])
        right_thickness = distance(right_brow[1], right_brow[4])
        
        # Eyebrow symmetry (position relative to eyes)
        left_eye_top = landmarks[39]
        right_eye_top = landmarks[273]
        
        left_brow_height = left_brow[2][1] - left_eye_top[1]
        right_brow_height = right_brow[2][1] - right_eye_top[1]
        
        # Ideal eyebrow sits above eye with ~1.5 eye heights clearance
        brow_eye_dist = abs(left_brow_height - right_brow_height)
        symmetry_score = max(0, 100 - brow_eye_dist)
        
        return {
            'symmetry': round(symmetry_score, 1),
            'thickness': round((left_thickness + right_thickness) / 2, 1)
        }
    
    def calculate_golden_ratio(self, landmarks):
        """Calculate golden ratio harmony score."""
        if not landmarks or len(landmarks) < 200:
            return 0
        
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        # Key facial points
        forehead_top = landmarks[10]
        eyebrow_top = landmarks[107]
        nose_bridge = landmarks[6]
        nose_tip = landmarks[1]
        chin = landmarks[152]
        
        # Calculate vertical segments
        face_height = distance(forehead_top, chin)
        upper_face = distance(forehead_top, nose_bridge)
        middle_face = distance(nose_bridge, nose_tip)
        lower_face = distance(nose_tip, chin)
        
        ratios = []
        if upper_face > 0:
            ratios.append(abs(face_height / upper_face - PHI) / PHI)
        if middle_face > 0:
            ratios.append(abs(lower_face / middle_face - PHI) / PHI)
        if lower_face > 0:
            ratios.append(abs(upper_face / lower_face - PHI) / PHI)
        
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
        
        def get_xy(landmark):
            return int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        
        shoulders = (lm[11], lm[12])
        hips = (lm[23], lm[24])
        
        shoulder_width = abs(get_xy(shoulders[0])[0] - get_xy(shoulders[1])[0])
        hip_width = abs(get_xy(hips[0])[0] - get_xy(hips[1])[0])
        
        if hip_width > 0:
            shoulder_hip_ratio = shoulder_width / hip_width
        else:
            shoulder_hip_ratio = 1
        
        shoulder_y = (get_xy(shoulders[0])[1] + get_xy(shoulders[1])[1]) // 2
        hip_y = (get_xy(hips[0])[1] + get_xy(hips[1])[1]) // 2
        upper_body_length = abs(hip_y - shoulder_y)
        
        return {
            'shoulder_hip_ratio': round(shoulder_hip_ratio, 2),
            'upper_body_length': upper_body_length,
            'shoulder_width': shoulder_width
        }
    
    def draw_marquardt_mask_overlay(self, image, landmarks, detailed=False):
        """Draw golden ratio grid overlay."""
        if not landmarks or len(landmarks) < 468:
            return image
        
        output = image.copy()
        h, w = image.shape[:2]
        
        # Colors
        GREEN = (0, 255, 0)
        MAGENTA = (255, 0, 255)
        CYAN = (255, 255, 0)
        
        # Draw facial outline
        jaw_points = [landmarks[i] for i in self.JAWLINE]
        jaw_points.append(jaw_points[0])
        cv2.polylines(output, [np.array(jaw_points)], False, GREEN, 2)
        
        # Draw eyes
        left_eye = [landmarks[i] for i in self.EYE_LEFT]
        right_eye = [landmarks[i] for i in self.EYE_RIGHT]
        cv2.polylines(output, [np.array(left_eye)], True, CYAN, 1)
        cv2.polylines(output, [np.array(right_eye)], True, CYAN, 1)
        
        # Draw nose
        nose_pts = [landmarks[i] for i in self.NOSE]
        cv2.polylines(output, [np.array(nose_pts)], False, MAGENTA, 2)
        
        # Draw lips
        lip_pts = [landmarks[i] for i in range(61, 68)]
        cv2.polylines(output, [np.array(lip_pts)], True, GREEN, 1)
        
        # Golden ratio vertical lines
        center_x = landmarks[152][0]
        for i in range(1, 5):
            x = int(center_x + (i * w * 0.04))
            cv2.line(output, (x, 0), (x, h), MAGENTA, 1, cv2.LINE_AA)
            x = int(center_x - (i * w * 0.04))
            cv2.line(output, (x, 0), (x, h), MAGENTA, 1, cv2.LINE_AA)
        
        # Horizontal golden lines
        face_top = landmarks[10][1]
        chin = landmarks[152][1]
        face_h = chin - face_top
        for i in range(1, 4):
            y = int(face_top + (i * face_h * 0.25))
            cv2.line(output, (0, y), (w, y), MAGENTA, 1, cv2.LINE_AA)
        
        return output
    
    def analyze(self, image_path, draw_overlay=False, detailed=False, output_json=None):
        """Run full beauty analysis on an image."""
        print(f"📷 Analyzing: {image_path}")
        
        landmarks, image = self.detect_face_landmarks(image_path)
        
        if not landmarks:
            print("❌ No face detected in image")
            return None
        
        print("✅ Face detected")
        
        # Core metrics
        symmetry = self.calculate_symmetry(landmarks)
        print(f"⚖️  Symmetry Score: {symmetry}/100")
        
        golden_ratio = self.calculate_golden_ratio(landmarks)
        print(f"🥇 Golden Ratio Harmony: {golden_ratio}/100")
        
        # Detailed facial features
        eye_analysis = self.analyze_eyes(landmarks)
        if eye_analysis:
            print(f"👁️  Eye Symmetry: {eye_analysis['symmetry']}/100")
        
        nose_analysis = self.analyze_nose(landmarks)
        if nose_analysis:
            print(f"👃 Nose Proportion Score: {nose_analysis['width_score']}/100")
        
        lip_analysis = self.analyze_lips(landmarks)
        if lip_analysis:
            print(f"👄 Lip Symmetry: {lip_analysis['symmetry']}/100")
        
        brow_analysis = self.analyze_eyebrows(landmarks)
        if brow_analysis:
            print(f"🩹 Eyebrow Symmetry: {brow_analysis['symmetry']}/100")
        
        # Body proportions
        body_props = self.analyze_body_proportions(image_path)
        if body_props:
            print(f"💪 Shoulder/Hip Ratio: {body_props['shoulder_hip_ratio']}")
        
        # Calculate weighted beauty score
        body_score = 100 if body_props and 1.4 < body_props['shoulder_hip_ratio'] < 1.8 else 70
        symmetry_weight = 0.35
        golden_weight = 0.35
        feature_weight = 0.30
        
        feature_score = 0
        feature_count = 0
        if eye_analysis:
            feature_score += eye_analysis['symmetry']
            feature_count += 1
        if nose_analysis:
            feature_score += nose_analysis['width_score']
            feature_count += 1
        if lip_analysis:
            feature_score += lip_analysis['symmetry']
            feature_count += 1
        if brow_analysis:
            feature_score += brow_analysis['symmetry']
            feature_count += 1
        
        feature_avg = feature_score / feature_count if feature_count > 0 else 70
        
        beauty_score = (symmetry * symmetry_weight + 
                       golden_ratio * golden_weight +
                       feature_avg * feature_weight)
        
        print(f"\n✨ Overall Beauty Score: {round(beauty_score, 1)}/100")
        
        # Save overlay
        if draw_overlay:
            output_path = Path(image_path).stem + "_analyzed.jpg"
            output = self.draw_marquardt_mask_overlay(image, landmarks, detailed)
            cv2.imwrite(str(output_path), output)
            print(f"💾 Saved analysis to: {output_path}")
        
        # Prepare result
        result = {
            'timestamp': datetime.now().isoformat(),
            'image': str(image_path),
            'beauty_score': round(beauty_score, 1),
            'symmetry': symmetry,
            'golden_ratio': golden_ratio,
            'eyes': eye_analysis,
            'nose': nose_analysis,
            'lips': lip_analysis,
            'eyebrows': brow_analysis,
            'body_proportions': body_props
        }
        
        # Save JSON
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"📄 Saved results to: {output_json}")
        
        return result


class RealtimeBeautyAnalyzer:
    """Real-time webcam beauty analyzer."""
    
    def __init__(self):
        self.analyzer = BeautyAnalyzer()
        self.cap = None
        
    def start(self):
        """Start real-time analysis."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("🎥 Starting real-time beauty analysis...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        results.multi_face_landmarks[0],
                        mp_face_mesh.FACEMESH_CONTOURS
                    )
                    
                    # Calculate and display scores
                    symmetry = self.analyzer.calculate_symmetry(landmarks_px)
                    golden = self.analyzer.calculate_golden_ratio(landmarks_px)
                    score = symmetry * 0.5 + golden * 0.5
                    
                    cv2.putText(frame, f"Score: {score:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Symmetry: {symmetry}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Golden: {golden}", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('BeautyEvaluator - Realtime', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('screenshot.jpg', frame)
                    print("📸 Screenshot saved!")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Realtime analysis ended")


def main():
    parser = argparse.ArgumentParser(description="BeautyEvaluator - AI Beauty Analysis")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--webcam", "-w", action="store_true", help="Use webcam for real-time analysis")
    parser.add_argument("--overlay", "-o", action="store_true", help="Save image with overlay")
    parser.add_argument("--detailed", "-d", action="store_true", help="Detailed analysis with all features")
    parser.add_argument("--json", "-j", help="Output results to JSON file")
    
    args = parser.parse_args()
    
    if args.webcam:
        realtime = RealtimeBeautyAnalyzer()
        realtime.start()
        return 0
    
    if not args.image:
        parser.print_help()
        print("\n📸 Example: python main.py selfie.jpg --overlay --json results.json")
        return 1
    
    analyzer = BeautyAnalyzer()
    result = analyzer.analyze(args.image, args.overlay, args.detailed, args.json)
    
    if result:
        print("\n📊 Analysis Complete!")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())