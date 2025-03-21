import cv2
import numpy as np
import time
import os

def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (original image, grayscale image)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, gray

def display_image(image, window_name="Image", wait_time=0):
    """
    Display an image in a window.
    
    Args:
        image (numpy.ndarray): Image to display
        window_name (str): Name of the display window
        wait_time (int): Time to wait (ms), 0 for indefinite
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    if wait_time == 0:
        cv2.destroyWindow(window_name)

def draw_bounding_boxes(image, detections, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw bounding boxes around detected objects.
    
    Args:
        image (numpy.ndarray): Image to draw on
        detections (list): List of bounding boxes in format (x, y, w, h)
        color (tuple): RGB color for the bounding boxes
        thickness (int): Line thickness
        label (str): Optional label to display
        
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    result_image = image.copy()
    
    for i, (x, y, w, h) in enumerate(detections):
        # Draw the rectangle
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        # Add label if provided
        if label:
            label_text = f"{label} {i+1}"
            cv2.putText(result_image, label_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return result_image

def open_video_capture(source):
    """
    Open a video capture object from file or webcam.
    
    Args:
        source: Video file path or webcam index (0 for default camera)
        
    Returns:
        cv2.VideoCapture: Video capture object
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    return cap

def fps_counter():
    """
    Create a simple FPS counter.
    
    Returns:
        tuple: (get_fps function, update function)
    """
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    def update():
        nonlocal start_time, frame_count, fps
        frame_count += 1
        
        # Update FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
    
    def get_fps():
        return fps
    
    return get_fps, update

def collision_warning_check(detection, frame_width, frame_height, threshold_ratio=0.15):
    """
    Check if a detected pedestrian is likely to cause a collision based on size and position.
    
    Args:
        detection (tuple): Bounding box (x, y, w, h)
        frame_width (int): Width of the frame
        frame_height (int): Height of the frame
        threshold_ratio (float): Size threshold ratio to trigger warning
        
    Returns:
        bool: True if warning should be triggered
    """
    x, y, w, h = detection
    
    # Calculate area of detection relative to frame
    detection_area = w * h
    frame_area = frame_width * frame_height
    area_ratio = detection_area / frame_area
    
    # Check if detection is in lower half of frame (closer to camera)
    is_lower_half = (y + h/2) > (frame_height / 2)
    
    # Trigger warning if object is large enough and in lower half
    return area_ratio > threshold_ratio and is_lower_half

def display_collision_warning(frame, message="WARNING: PEDESTRIAN AHEAD!", 
                              position=(50, 50), color=(0, 0, 255), 
                              thickness=2, font_scale=1):
    """
    Display a collision warning message on the frame.
    
    Args:
        frame (numpy.ndarray): Image to display warning on
        message (str): Warning message to display
        position (tuple): Position (x, y) to place text
        color (tuple): RGB color for text
        thickness (int): Text thickness
        font_scale (float): Font scale factor
        
    Returns:
        numpy.ndarray: Frame with warning message
    """
    result = frame.copy()
    cv2.putText(result, message, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    return result 