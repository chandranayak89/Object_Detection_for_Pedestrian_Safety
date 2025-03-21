#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pedestrian Detection using Haar Cascades
----------------------------------------
This script implements pedestrian detection using OpenCV's pre-trained Haar Cascade classifier.
It can process both images and video input.
"""

import cv2
import argparse
import numpy as np
import os
from utils import load_image, display_image, draw_bounding_boxes, open_video_capture, fps_counter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pedestrian detection using Haar Cascades')
    
    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to input image')
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Additional parameters
    parser.add_argument('--scale-factor', type=float, default=1.1, 
                        help='Scale factor for cascade detector')
    parser.add_argument('--min-neighbors', type=int, default=5, 
                        help='Minimum neighbors for cascade detector')
    parser.add_argument('--min-size', type=int, default=30, 
                        help='Minimum detection size')
    parser.add_argument('--output', type=str, help='Path to save output')
    
    return parser.parse_args()

def detect_pedestrians_haar(image, cascade_classifier, args):
    """
    Detect pedestrians in an image using Haar Cascade.
    
    Args:
        image (numpy.ndarray): Input image
        cascade_classifier: OpenCV Cascade Classifier
        args: Command line arguments
        
    Returns:
        tuple: (image with detections, list of detections)
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect pedestrians
    pedestrians = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=(args.min_size, args.min_size)
    )
    
    # Draw bounding boxes on a copy of the image
    result_image = draw_bounding_boxes(image, pedestrians, label="Pedestrian")
    
    # Add detection count
    cv2.putText(result_image, f"Pedestrians: {len(pedestrians)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return result_image, pedestrians

def process_image(args):
    """Process a single image for pedestrian detection."""
    # Load cascade classifier
    pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    
    # Load image
    image, _ = load_image(args.image)
    
    # Detect pedestrians
    result_image, pedestrians = detect_pedestrians_haar(image, pedestrian_cascade, args)
    
    # Display result
    print(f"Detected {len(pedestrians)} pedestrians")
    display_image(result_image, "Pedestrian Detection (Haar)")
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")

def process_video(args):
    """Process video for pedestrian detection."""
    # Load cascade classifier
    pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    
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
            result_frame, pedestrians = detect_pedestrians_haar(frame, pedestrian_cascade, args)
            
            # Update and display FPS
            update_fps()
            cv2.putText(result_frame, f"FPS: {get_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display result
            cv2.imshow("Pedestrian Detection (Haar)", result_frame)
            
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