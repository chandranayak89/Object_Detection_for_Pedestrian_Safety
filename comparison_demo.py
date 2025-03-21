#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pedestrian Detection Method Comparison Demo
------------------------------------------
This script demonstrates the three different pedestrian detection methods:
1. Haar Cascades
2. HOG+SVM
3. YOLOv5

It displays all methods side by side for easy comparison.
"""

import cv2
import argparse
import numpy as np
import os
import time
import torch
import sys
from utils import open_video_capture, fps_counter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pedestrian Detection Methods Comparison')
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Additional parameters
    parser.add_argument('--model', type=str, default='yolov5s', 
                        help='YOLOv5 model size (yolov5s, yolov5m, yolov5l, yolov5x)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for detection')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--scale', type=float, default=0.7, 
                        help='Display scale factor')
    parser.add_argument('--device', type=str, default='', 
                        help='Device to run inference on (cuda device or cpu)')
    
    return parser.parse_args()

def init_detectors(args):
    """Initialize all three detectors."""
    detectors = {}
    
    # 1. Haar Cascade detector
    try:
        haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        detectors['haar'] = haar_detector
        print("Haar Cascade detector initialized")
    except Exception as e:
        print(f"Failed to initialize Haar Cascade detector: {e}")
        detectors['haar'] = None
    
    # 2. HOG+SVM detector
    try:
        hog_detector = cv2.HOGDescriptor()
        hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        detectors['hog'] = hog_detector
        print("HOG+SVM detector initialized")
    except Exception as e:
        print(f"Failed to initialize HOG+SVM detector: {e}")
        detectors['hog'] = None
    
    # 3. YOLO detector
    try:
        # Determine model path
        model_name = args.model
        if not model_name.startswith('yolov5'):
            model_name = f'yolov5{model_name}'
        
        try:
            yolo_detector = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        except Exception as e:
            print(f"Error loading model from torch hub: {e}")
            print("Trying to load a local model...")
            try:
                from ultralytics import YOLO
                yolo_detector = YOLO(f"{model_name}.pt")
            except Exception as e:
                print(f"Error loading local model: {e}")
                raise ValueError(f"Could not load YOLOv5 model: {model_name}")
        
        # Set model parameters
        yolo_detector.conf = args.confidence
        if args.device:
            yolo_detector.to(args.device)
        
        detectors['yolo'] = yolo_detector
        print(f"YOLO detector initialized ({model_name})")
    except Exception as e:
        print(f"Failed to initialize YOLO detector: {e}")
        detectors['yolo'] = None
    
    return detectors

def detect_haar(frame, detector):
    """Detect pedestrians using Haar Cascade."""
    start_time = time.time()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect pedestrians
    pedestrians = detector.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # Draw rectangles
    result = frame.copy()
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add method name and detection count
    cv2.putText(result, "Haar Cascade", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result, f"Detections: {len(pedestrians)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add processing time
    process_time = (time.time() - start_time) * 1000  # ms
    cv2.putText(result, f"Time: {process_time:.1f}ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return result

def detect_hog(frame, detector):
    """Detect pedestrians using HOG+SVM."""
    start_time = time.time()
    
    # Detect pedestrians
    pedestrians, weights = detector.detectMultiScale(
        frame, 
        winStride=(8, 8), 
        padding=(8, 8), 
        scale=1.05
    )
    
    # Draw rectangles
    result = frame.copy()
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add method name and detection count
    cv2.putText(result, "HOG + SVM", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result, f"Detections: {len(pedestrians)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add processing time
    process_time = (time.time() - start_time) * 1000  # ms
    cv2.putText(result, f"Time: {process_time:.1f}ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return result

def detect_yolo(frame, detector):
    """Detect pedestrians using YOLOv5."""
    start_time = time.time()
    
    # Convert BGR to RGB for YOLO
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = detector(rgb_image)
    
    # Get person detections (class 0 in COCO dataset)
    result = frame.copy()
    person_count = 0
    
    for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        if int(cls) == 0:  # Class 0 is person in COCO
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            person_count += 1
    
    # Add method name and detection count
    cv2.putText(result, "YOLOv5", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result, f"Detections: {person_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add processing time
    process_time = (time.time() - start_time) * 1000  # ms
    cv2.putText(result, f"Time: {process_time:.1f}ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return result

def create_comparison_frame(frames, scale=0.7):
    """Create a side-by-side comparison of all detection methods."""
    # Scale frames if needed
    if scale != 1.0:
        scaled_frames = []
        for frame in frames:
            if frame is not None:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(frame, (new_w, new_h))
                scaled_frames.append(scaled)
            else:
                # Create a black frame as placeholder
                placeholder = np.zeros((int(frames[0].shape[0] * scale), 
                                      int(frames[0].shape[1] * scale), 3), 
                                     dtype=np.uint8)
                cv2.putText(placeholder, "Detector Not Available", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                scaled_frames.append(placeholder)
    else:
        scaled_frames = frames
    
    # Create horizontal stacking
    comparison = np.hstack(scaled_frames)
    
    return comparison

def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize detectors
    detectors = init_detectors(args)
    
    # Check if at least one detector is initialized
    if all(detector is None for detector in detectors.values()):
        print("Error: Failed to initialize any detectors")
        sys.exit(1)
    
    # Open video capture
    source = 0 if args.webcam else args.video
    cap = open_video_capture(source)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer if output is specified
    if args.output:
        # Calculate the output dimensions
        output_width = int(width * args.scale * 3)  # 3 detection methods
        output_height = int(height * args.scale)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (output_width, output_height))
    
    # FPS counter
    get_fps, update_fps = fps_counter()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detections
            haar_result = detect_haar(frame, detectors['haar']) if detectors['haar'] is not None else None
            hog_result = detect_hog(frame, detectors['hog']) if detectors['hog'] is not None else None
            yolo_result = detect_yolo(frame, detectors['yolo']) if detectors['yolo'] is not None else None
            
            # Create comparison frame
            comparison = create_comparison_frame([haar_result, hog_result, yolo_result], args.scale)
            
            # Update and display FPS
            update_fps()
            cv2.putText(comparison, f"Overall FPS: {get_fps():.1f}", (10, comparison.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display result
            cv2.imshow("Pedestrian Detection Methods Comparison", comparison)
            
            # Write frame to output video if specified
            if args.output:
                out.write(comparison)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if args.output:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 