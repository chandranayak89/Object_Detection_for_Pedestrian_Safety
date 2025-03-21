#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pedestrian Detection using YOLOv5
---------------------------------
This script implements pedestrian detection using YOLOv5,
a deep learning model for object detection. This provides
the highest accuracy for pedestrian detection compared to
traditional computer vision methods.
"""

import cv2
import argparse
import numpy as np
import os
import torch
from utils import open_video_capture, fps_counter, load_image, display_image

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pedestrian detection using YOLOv5')
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to input image')
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Additional parameters
    parser.add_argument('--model', type=str, default='yolov5s', 
                        help='YOLOv5 model size (yolov5s, yolov5m, yolov5l, yolov5x)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for detection')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--device', type=str, default='', 
                        help='Device to run inference on (cuda device or cpu)')
    
    return parser.parse_args()

def load_yolo_model(args):
    """
    Load YOLOv5 model.
    
    Args:
        args: Command line arguments
        
    Returns:
        YOLO model
    """
    # Determine model path
    model_name = args.model
    if not model_name.startswith('yolov5'):
        model_name = f'yolov5{model_name}'
    
    # Load model from ultralytics
    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    except Exception as e:
        print(f"Error loading model from torch hub: {e}")
        print("Trying to load a local model...")
        try:
            from ultralytics import YOLO
            model = YOLO(f"{model_name}.pt")
        except Exception as e:
            print(f"Error loading local model: {e}")
            raise ValueError(f"Could not load YOLOv5 model: {model_name}")
    
    # Set model parameters
    model.conf = args.confidence  # Confidence threshold
    if args.device:
        model.to(args.device)
    
    return model

def detect_pedestrians_yolo(image, model):
    """
    Detect pedestrians in an image using YOLOv5.
    
    Args:
        image (numpy.ndarray): Input image in BGR format (OpenCV)
        model: YOLOv5 model
        
    Returns:
        tuple: (image with detections, list of detections)
    """
    # Convert BGR to RGB (YOLOv5 expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(rgb_image)
    
    # Convert back to BGR for display with OpenCV
    result_image = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)
    
    # Get person detections (class 0 in COCO dataset)
    person_detections = []
    for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        if int(cls) == 0:  # Class 0 is person in COCO
            w = x2 - x1
            h = y2 - y1
            person_detections.append((int(x1), int(y1), int(w), int(h)))
    
    # Add detection count
    cv2.putText(result_image, f"Pedestrians: {len(person_detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return result_image, person_detections

def process_image(args):
    """Process a single image for pedestrian detection."""
    # Load model
    model = load_yolo_model(args)
    
    # Load image
    image, _ = load_image(args.image)
    
    # Detect pedestrians
    result_image, pedestrians = detect_pedestrians_yolo(image, model)
    
    # Display result
    print(f"Detected {len(pedestrians)} pedestrians")
    display_image(result_image, "Pedestrian Detection (YOLOv5)")
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")

def process_video(args):
    """Process video for pedestrian detection."""
    # Load model
    model = load_yolo_model(args)
    
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
            result_frame, pedestrians = detect_pedestrians_yolo(frame, model)
            
            # Update and display FPS
            update_fps()
            cv2.putText(result_frame, f"FPS: {get_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display result
            cv2.imshow("Pedestrian Detection (YOLOv5)", result_frame)
            
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
    
    if args.image:
        process_image(args)
    else:  # Video or webcam
        process_video(args)

if __name__ == "__main__":
    main() 