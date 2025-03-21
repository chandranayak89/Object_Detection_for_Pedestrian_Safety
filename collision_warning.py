#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pedestrian Collision Warning System
----------------------------------
This script implements a collision warning system that detects pedestrians
and estimates their proximity to trigger warnings when pedestrians are too close.
It can work with any of the detection methods (Haar, HOG+SVM, YOLO).
"""

import cv2
import argparse
import numpy as np
import os
import time
from utils import (
    open_video_capture, 
    fps_counter, 
    collision_warning_check, 
    display_collision_warning
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pedestrian Collision Warning System')
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Detection method
    parser.add_argument('--method', type=str, default='hog', choices=['haar', 'hog', 'yolo'],
                        help='Pedestrian detection method')
    
    # Additional parameters
    parser.add_argument('--model', type=str, default='yolov5s', 
                        help='YOLOv5 model size (for YOLO method)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for detection')
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help='Size threshold ratio for collision warning')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--device', type=str, default='', 
                        help='Device to run inference on (cuda device or cpu)')
    
    return parser.parse_args()

def init_detector(method, args):
    """
    Initialize pedestrian detector based on the selected method.
    
    Args:
        method (str): Detection method ('haar', 'hog', or 'yolo')
        args: Command line arguments
        
    Returns:
        detector: Initialized detector
    """
    if method == 'haar':
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    
    elif method == 'hog':
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    elif method == 'yolo':
        try:
            import torch
            model_name = args.model
            if not model_name.startswith('yolov5'):
                model_name = f'yolov5{model_name}'
                
            try:
                detector = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            except Exception as e:
                print(f"Error loading model from torch hub: {e}")
                print("Trying to load a local model...")
                try:
                    from ultralytics import YOLO
                    detector = YOLO(f"{model_name}.pt")
                except Exception as e:
                    print(f"Error loading local model: {e}")
                    raise ValueError(f"Could not load YOLOv5 model: {model_name}")
                
            # Set model parameters
            detector.conf = args.confidence
            if args.device:
                detector.to(args.device)
        except ImportError:
            raise ImportError("YOLOv5 requires PyTorch. Please install it with 'pip install torch'")
    
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    return detector

def detect_pedestrians(frame, method, detector, args):
    """
    Detect pedestrians in a frame using the specified method.
    
    Args:
        frame (numpy.ndarray): Input frame
        method (str): Detection method
        detector: Initialized detector
        args: Command line arguments
        
    Returns:
        tuple: (frame with detections, list of detections)
    """
    detections = []
    frame_with_detections = frame.copy()
    
    if method == 'haar':
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
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detections.append((x, y, w, h))
    
    elif method == 'hog':
        # Detect pedestrians
        pedestrians, weights = detector.detectMultiScale(
            frame, 
            winStride=(8, 8), 
            padding=(8, 8), 
            scale=1.05
        )
        
        # Filter by confidence if available
        if len(weights) > 0 and hasattr(args, 'confidence'):
            pedestrians = [ped for i, ped in enumerate(pedestrians) if weights[i] > args.confidence]
        
        # Draw rectangles
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detections.append((x, y, w, h))
    
    elif method == 'yolo':
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = detector(rgb_image)
        
        # Get person detections (class 0 in COCO dataset)
        for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            if int(cls) == 0:  # Class 0 is person in COCO
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Draw rectangle
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_with_detections, f"Ped: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append((x1, y1, w, h))
    
    # Add detection count
    cv2.putText(frame_with_detections, f"Pedestrians: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame_with_detections, detections

def process_video(args):
    """Process video with pedestrian detection and collision warning."""
    # Initialize detector
    detector = init_detector(args.method, args)
    
    # Open video capture
    source = 0 if args.webcam else args.video
    cap = open_video_capture(source)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer if output is specified
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # FPS counter
    get_fps, update_fps = fps_counter()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect pedestrians
            result_frame, detections = detect_pedestrians(frame, args.method, detector, args)
            
            # Check for collision warnings
            warning_active = False
            for detection in detections:
                if collision_warning_check(detection, width, height, args.threshold):
                    warning_active = True
                    break
            
            # Display warning if needed
            if warning_active:
                result_frame = display_collision_warning(result_frame)
            
            # Update and display FPS
            update_fps()
            cv2.putText(result_frame, f"FPS: {get_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display method
            cv2.putText(result_frame, f"Method: {args.method.upper()}", (width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display result
            cv2.imshow("Pedestrian Collision Warning", result_frame)
            
            # Write frame to output video if specified
            if args.output:
                out.write(result_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if args.output:
            out.release()
        cv2.destroyAllWindows()

def main():
    """Main function."""
    args = parse_arguments()
    process_video(args)

if __name__ == "__main__":
    main() 