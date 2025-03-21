#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pedestrian Detection using HOG+SVM
---------------------------------
This script implements pedestrian detection using Histogram of Oriented Gradients (HOG)
with Support Vector Machine (SVM) classifier. This approach provides better accuracy
than Haar cascades for pedestrian detection.
"""

import cv2
import argparse
import numpy as np
import os
from utils import load_image, display_image, draw_bounding_boxes, open_video_capture, fps_counter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pedestrian detection using HOG+SVM')
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to input image')
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Additional parameters
    parser.add_argument('--scale', type=float, default=1.05, 
                        help='Scale factor for detection')
    parser.add_argument('--padding', type=tuple, default=(8, 8), 
                        help='Padding for detection')
    parser.add_argument('--winstride', type=tuple, default=(4, 4), 
                        help='Window stride for detection')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold for detection')
    
    return parser.parse_args()

def detect_pedestrians_hog(image, hog_detector, args):
    """
    Detect pedestrians in an image using HOG+SVM.
    
    Args:
        image (numpy.ndarray): Input image
        hog_detector: OpenCV HOG detector
        args: Command line arguments
        
    Returns:
        tuple: (image with detections, list of detections)
    """
    # Detect pedestrians
    pedestrians, weights = hog_detector.detectMultiScale(
        image,
        winStride=args.winstride,
        padding=args.padding,
        scale=args.scale,
        useMeanshiftGrouping=False
    )
    
    # Filter detections by confidence if weights are available
    if len(weights) > 0 and hasattr(args, 'confidence'):
        filtered_pedestrians = []
        for i, weight in enumerate(weights):
            if weight > args.confidence:
                filtered_pedestrians.append(pedestrians[i])
        pedestrians = filtered_pedestrians
    
    # Convert detections to the format (x, y, w, h)
    formatted_pedestrians = []
    for (x, y, w, h) in pedestrians:
        formatted_pedestrians.append((x, y, w, h))
    
    # Draw bounding boxes on a copy of the image
    result_image = draw_bounding_boxes(image, formatted_pedestrians, label="Person")
    
    # Add detection count
    cv2.putText(result_image, f"Pedestrians: {len(formatted_pedestrians)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return result_image, formatted_pedestrians

def process_image(args):
    """Process a single image for pedestrian detection."""
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Load image
    image, _ = load_image(args.image)
    
    # Detect pedestrians
    result_image, pedestrians = detect_pedestrians_hog(image, hog, args)
    
    # Display result
    print(f"Detected {len(pedestrians)} pedestrians")
    display_image(result_image, "Pedestrian Detection (HOG+SVM)")
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")

def process_video(args):
    """Process video for pedestrian detection."""
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
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
            result_frame, pedestrians = detect_pedestrians_hog(frame, hog, args)
            
            # Update and display FPS
            update_fps()
            cv2.putText(result_frame, f"FPS: {get_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display result
            cv2.imshow("Pedestrian Detection (HOG+SVM)", result_frame)
            
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
    
    # Fix tuple arguments (argparse doesn't handle tuples well)
    args.winstride = (4, 4)
    args.padding = (8, 8)
    
    if args.image:
        process_image(args)
    else:  # Video or webcam
        process_video(args)

if __name__ == "__main__":
    main() 