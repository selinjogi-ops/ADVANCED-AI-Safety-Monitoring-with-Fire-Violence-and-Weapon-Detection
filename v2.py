"""
ADVANCED AI Safety Monitoring with Fire, Violence, and Weapon Detection
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os
import sys
import requests
from PIL import Image
import torchvision.transforms as transforms
import threading
import queue

print("=" * 70)
print("🔥 ADVANCED AI SAFETY MONITORING SYSTEM - V2")
print("=" * 70)

class FireDetector:
    """Specialized fire detector"""
    
    def __init__(self):
        print("🔥 Initializing Fire Detection Model...")
        try:
            # Load fire detection model
            self.model = YOLO('yolov8n-fire.pt')  # Will use custom model
            self.has_fire_model = True
        except:
            print("⚠️  No fire model, using computer vision detection")
            self.has_fire_model = False
        
        # Fire detection using computer vision
        self.fire_history = []
        self.fire_threshold = 0.05  # 5% of frame
    
    def detect_fire_cv(self, frame):
        """Detect fire using computer vision (color-based)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire color ranges (red, orange, yellow)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Yellow/Orange for fire
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        fire_mask = cv2.bitwise_or(mask_red1, mask_red2)
        fire_mask = cv2.bitwise_or(fire_mask, mask_yellow)
        
        # Calculate fire percentage
        fire_pixels = np.sum(fire_mask > 0)
        total_pixels = fire_mask.size
        fire_ratio = fire_pixels / total_pixels
        
        # Check for flickering (fire characteristic)
        self.fire_history.append(fire_ratio)
        if len(self.fire_history) > 10:
            self.fire_history.pop(0)
        
        # Calculate flicker rate
        if len(self.fire_history) >= 5:
            flicker_rate = np.std(self.fire_history[-5:])
        else:
            flicker_rate = 0
        
        fire_detected = fire_ratio > self.fire_threshold and flicker_rate > 0.01
        
        return fire_detected, fire_ratio, fire_mask
    
    def detect(self, frame):
        """Detect fire only (smoke detection removed)"""
        fire_detected, fire_ratio, fire_mask = self.detect_fire_cv(frame)
        
        results = []
        
        if fire_detected:
            # Find fire region
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                results.append({
                    'class': 'fire',
                    'confidence': min(fire_ratio * 10, 0.95),
                    'bbox': (x, y, x + w, y + h),
                    'ratio': fire_ratio
                })
        
        return results

class ViolenceDetector:
    """Detect violent behavior using motion and pose analysis"""
    
    def __init__(self):
        print("🥊 Initializing Violence Detection...")
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
            self.has_pose_model = True
        except:
            print("⚠️  No pose model, using basic motion detection")
            self.has_pose_model = False
        
        # Motion history for violence detection
        self.motion_history = []
        self.violence_threshold = 0.15  # Motion threshold
    
    def detect_violence(self, frame, prev_frame=None):
        """Detect violent behavior through motion and pose"""
        if prev_frame is None:
            return []
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion percentage
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_ratio = motion_pixels / total_pixels
        
        # Update motion history
        self.motion_history.append(motion_ratio)
        if len(self.motion_history) > 20:
            self.motion_history.pop(0)
        
        # Check for sudden, violent motion
        violence_detected = False
        if len(self.motion_history) >= 5:
            recent_motion = self.motion_history[-5:]
            # Focus on sudden spikes characteristic of "hitting"
            motion_spike = max(recent_motion) > (self.violence_threshold * 0.7)
            motion_variance = np.std(recent_motion) > 0.02
            
            if motion_spike and motion_variance:
                violence_detected = True
        
        results = []
        if violence_detected:
            # Find motion region
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.vstack([c for c in contours])
                if len(all_points) > 0:
                    x, y, w, h = cv2.boundingRect(all_points)
                    results.append({
                        'class': 'violent_behavior',
                        'confidence': min(motion_ratio * 8, 0.9),
                        'bbox': (x, y, x + w, y + h),
                        'motion_level': motion_ratio
                    })
        
        return results

class AdvancedSafetyMonitor:
    def __init__(self):
        print("🚀 Initializing Advanced Safety System V2...")
        
        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            for cam_id in [1, 2, 3]:
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    print(f"✅ Using camera {cam_id}")
                    break
        
        if not self.cap.isOpened():
            print("❌ No camera found")
            sys.exit(1)
        
        # Try to set high resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check actual resolution
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✅ Camera resolution set to: {actual_w}x{actual_h}")
        
        # Load models
        print("📥 Loading YOLOv8n for object detection...")
        self.yolo_model = YOLO('yolov8n.pt')
        print("✅ Object detection model loaded")
        
        self.fire_detector = FireDetector()
        self.violence_detector = ViolenceDetector()
        
        self.config = {
            'width': 960,    # Medium display width
            'height': 540,   # Medium display height
            'yolo_confidence': 0.5,
            'alert_frames': 3
        }
        
        # Classes to monitor
        self.dangerous_classes = {
            'knife': 'WEAPON',
            'scissors': 'WEAPON',
            'cell phone': 'INFO',
            'bottle': 'WATER BOTTLE',
            'person': 'PERSON',
            'fire': 'FIRE_EMERGENCY',
            'violent_behavior': 'VIOLENCE_EMERGENCY'
        }
        
        self.alerts = []
        self.detection_history = {}
        self.frame_count = 0
        self.prev_frame = None
        
        # Performance optimizations
        self.inference_frequency = 2  # Run inference every 2 frames
        self.cached_detections = []
        self.last_time = time.time()
        
        # Threaded TTS
        self.voice_enabled = True
        self.info_mode = False  # Toggle for showing secondary objects
        self.speech_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        self.colors = {
            'FIRE_EMERGENCY': (0, 0, 255),        # Red
            'VIOLENCE_EMERGENCY': (255, 0, 255),  # Purple
            'WEAPON': (0, 0, 255),                # Red for Weapons too
            'SUSPICIOUS': (255, 255, 0),          # Cyan
            'PERSON': (0, 255, 255),              # Yellow
            'WATER BOTTLE': (0, 255, 0),          # Green
            'INFO': (0, 255, 0)
        }
        
        print("✅ Advanced Safety System V2 initialized!")
    
    def detect_all_hazards(self, frame):
        """Run all detection models on frame"""
        all_detections = []
        
        # 1. YOLO object detection
        yolo_results = self.yolo_model(frame, conf=self.config['yolo_confidence'], verbose=False)
        if yolo_results and len(yolo_results) > 0:
            boxes = yolo_results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                if hasattr(self.yolo_model, 'names') and class_id in self.yolo_model.names:
                    class_name = self.yolo_model.names[class_id]
                    danger_class = self.dangerous_classes.get(class_name, 'INFO')
                    
                    all_detections.append({
                        'class': danger_class,
                        'original_class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'detector': 'yolo'
                    })
        
        # 2. Fire detection
        fire_results = self.fire_detector.detect(frame)
        for fire_det in fire_results:
            all_detections.append({
                'class': 'FIRE_EMERGENCY',
                'original_class': fire_det['class'],
                'confidence': fire_det['confidence'],
                'bbox': fire_det['bbox'],
                'detector': 'fire'
            })
        
        # 3. Violence detection
        if self.prev_frame is not None:
            violence_results = self.violence_detector.detect_violence(frame, self.prev_frame)
            for violence_det in violence_results:
                all_detections.append({
                    'class': 'VIOLENCE_EMERGENCY',
                    'original_class': 'violent_behavior',
                    'confidence': violence_det['confidence'],
                    'bbox': violence_det['bbox'],
                    'detector': 'violence'
                })
        
        self.prev_frame = frame.copy()
        return all_detections
    
    def check_and_alert(self, detections):
        """Check detections and trigger alerts"""
        for det in detections:
            danger_class = det['class']
            confidence = det['confidence']
            
            if danger_class not in self.detection_history:
                self.detection_history[danger_class] = []
            
            self.detection_history[danger_class].append({
                'frame': self.frame_count,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            if len(self.detection_history[danger_class]) > 50:
                self.detection_history[danger_class].pop(0)
            
            recent = [d for d in self.detection_history[danger_class] 
                     if d['frame'] > self.frame_count - self.config['alert_frames']]
            
            if len(recent) >= self.config['alert_frames'] and confidence > 0.4:
                alert_active = any(
                    a['type'] == danger_class and 
                    a['frame'] > self.frame_count - 45  # 3 second cooldown
                    for a in self.alerts
                )
                
                if not alert_active:
                    alert_msg = self.get_alert_message(danger_class, det)
                    is_emergency = danger_class in ['FIRE_EMERGENCY', 'VIOLENCE_EMERGENCY', 'WEAPON']
                    
                    alert = {
                        'type': danger_class,
                        'message': alert_msg,
                        'frame': self.frame_count,
                        'timestamp': datetime.now(),
                        'acknowledged': not is_emergency
                    }
                    self.alerts.append(alert)
                    
                    if is_emergency:
                        print(f"🚨 {alert_msg}")
                        self.speak_alert(danger_class, alert_msg)
    
    def get_alert_message(self, danger_class, detection):
        """Simplified messages as requested"""
        messages = {
            'FIRE_EMERGENCY': "Fire Detected",
            'VIOLENCE_EMERGENCY': "Violence Detected",
            'WEAPON': "Weapon Detected",
            'PERSON': "Person Detected",
            'WATER BOTTLE': "Water Bottle",
            'INFO': "Object Detected"
        }
        
        # If it's an INFO class, try to get specific name
        if danger_class == 'INFO':
            return detection.get('original_class', 'Object').upper()
            
        return messages.get(danger_class, f"Alert: {danger_class}")
    
    def speak_alert(self, alert_type, message):
        """Queue alert for speaking"""
        if self.voice_enabled:
            self.speech_queue.put(message)

    def _tts_worker(self):
        """Worker thread for TTS"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            while True:
                msg = self.speech_queue.get()
                if msg is None: break
                engine.say(msg)
                engine.runAndWait()
                self.speech_queue.task_done()
        except:
            pass
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and simplified labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            danger_class = det['class']
            confidence = det['confidence']
            
            # Filter non-emergencies unless info_mode is ON
            is_emergency = danger_class in ['FIRE_EMERGENCY', 'VIOLENCE_EMERGENCY', 'WEAPON']
            if not is_emergency and not self.info_mode:
                continue

            color = self.colors.get(danger_class, (255, 255, 255))
            thickness = 4 if danger_class in ['FIRE_EMERGENCY', 'VIOLENCE_EMERGENCY', 'WEAPON'] else 2
            
            # Label should match the requested simple alerts
            label_text = self.get_alert_message(danger_class, det).upper()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Icons for emergencies
            if danger_class == 'FIRE_EMERGENCY':
                cv2.putText(frame, "FIRE", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif danger_class == 'VIOLENCE_EMERGENCY':
                cv2.putText(frame, "VIOLENCE", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            elif danger_class == 'WEAPON':
                cv2.putText(frame, "WEAPON", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def draw_alerts_dashboard(self, frame):
        """Alert dashboard"""
        active_alerts = [a for a in self.alerts[-5:] if not a.get('acknowledged', False)]
        panel_height = 140 if active_alerts else 60
        cv2.rectangle(frame, (0, 0), (self.config['width'], panel_height), (20, 20, 20), -1)
        
        cv2.putText(frame, "V2 AI SAFETY MONITOR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Keys info on HUD
        v_status = "ON" if self.voice_enabled else "OFF"
        i_status = "ON" if self.info_mode else "OFF"
        keys_info = f"[Q]Quit [A]Ack [V]Voice:{v_status} [I]Info:{i_status} [S]Save [1]Fire [2]Viol"
        cv2.putText(frame, keys_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        alert_count = len(active_alerts)
        color = (0, 0, 255) if alert_count > 0 else (0, 255, 0)
        cv2.putText(frame, f"Active Alerts: {alert_count}", (self.config['width'] - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if active_alerts:
            y = 80
            for alert in active_alerts[:3]:
                msg = alert['message']
                cv2.putText(frame, f"ALARM: {msg}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25
        
        return frame, panel_height
    
    def run(self):
        """Main loop"""
        window_name = "V2 AI Safety Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Force the window to be large on start
        cv2.resizeWindow(window_name, self.config['width'], self.config['height'])
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                self.frame_count += 1
                frame = cv2.resize(frame, (self.config['width'], self.config['height']))
                
                if self.frame_count % self.inference_frequency == 0:
                    self.cached_detections = self.detect_all_hazards(frame)
                    self.check_and_alert(self.cached_detections)
                
                frame = self.draw_detections(frame, self.cached_detections)
                frame, panel_height = self.draw_alerts_dashboard(frame)
                
                curr_time = time.time()
                fps = 1.0 / (curr_time - self.last_time) if (curr_time - self.last_time) > 0 else 0
                self.last_time = curr_time
                
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"{ts} | FPS: {fps:.1f}", (10, self.config['height'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("V2 AI Safety Monitor", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: break
                elif key == ord('a'):
                    for a in self.alerts: a['acknowledged'] = True
                elif key == ord('v'):
                    self.voice_enabled = not self.voice_enabled
                    print(f"🔊 Voice output: {'ENABLED' if self.voice_enabled else 'DISABLED'}")
                elif key == ord('i'):
                    self.info_mode = not self.info_mode
                    print(f"ℹ️ Info display mode: {'ENABLED' if self.info_mode else 'DISABLED'}")
                elif key == ord('1'):
                    print("🔥 Fire test")
                    self.speak_alert('FIRE', "Fire Detected")
                elif key == ord('2'):
                    print("🥊 Violence test")
                    self.speak_alert('VIOLENCE', "Violence Detected")
                elif key == ord('s'):
                    cv2.imwrite(f"alert_{ts.replace(':','')}.jpg", frame)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.speech_queue.put(None)
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = AdvancedSafetyMonitor()
    monitor.run()
