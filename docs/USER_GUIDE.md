# Pedestrian Detection System: User Guide

This guide will help you set up and use the Pedestrian Detection System for safety applications.

## Table of Contents

1. [Installation](#installation)
2. [Running the Detection Systems](#running-the-detection-systems)
3. [Comparing Detection Methods](#comparing-detection-methods)
4. [Collision Warning System](#collision-warning-system)
5. [Troubleshooting](#troubleshooting)

## Installation

### Windows

1. Make sure Python 3.7+ is installed on your system
2. Run the `install.bat` script by double-clicking it or running it from the command prompt

```bash
install.bat
```

3. Activate the virtual environment

```bash
venv\Scripts\activate.bat
```

### Linux/macOS

1. Make sure Python 3.7+ is installed on your system
2. Make the installation script executable and run it

```bash
chmod +x install.sh
./install.sh
```

3. Activate the virtual environment

```bash
source venv/bin/activate
```

## Running the Detection Systems

### 1. Haar Cascade Detection

This is the simplest method, but may have lower accuracy:

```bash
# Process an image
python pedestrian_detection_haar.py --image data/pedestrian.jpg

# Process a video file
python pedestrian_detection_haar.py --video data/pedestrians.mp4

# Use webcam as input
python pedestrian_detection_haar.py --webcam

# Save the output
python pedestrian_detection_haar.py --webcam --output output.avi
```

### 2. HOG+SVM Detection

This method provides better accuracy than Haar cascades:

```bash
# Process an image
python pedestrian_detection_hog.py --image data/pedestrian.jpg

# Process a video file
python pedestrian_detection_hog.py --video data/pedestrians.mp4

# Use webcam as input
python pedestrian_detection_hog.py --webcam

# Adjust parameters
python pedestrian_detection_hog.py --webcam --scale 1.03 --confidence 0.6
```

### 3. YOLO Detection

This deep learning approach offers the highest accuracy:

```bash
# Process an image
python pedestrian_detection_yolo.py --image data/pedestrian.jpg

# Process a video file
python pedestrian_detection_yolo.py --video data/pedestrians.mp4

# Use webcam as input
python pedestrian_detection_yolo.py --webcam

# Use a larger model for better accuracy (but slower)
python pedestrian_detection_yolo.py --webcam --model yolov5m

# Adjust confidence threshold
python pedestrian_detection_yolo.py --webcam --confidence 0.7
```

## Comparing Detection Methods

Each detection method has its own strengths and weaknesses:

| Method | Accuracy | Speed | Memory Usage | Best Use Case |
|--------|----------|-------|--------------|---------------|
| Haar Cascade | Low-Medium | Fast | Low | Resource-constrained devices, simple scenarios |
| HOG+SVM | Medium | Medium | Low | Balanced approach, general use |
| YOLO | High | Slow-Medium | High | High accuracy requirements, powerful hardware |

## Collision Warning System

The collision warning system works with any of the detection methods:

```bash
# Use HOG+SVM detection with collision warning
python collision_warning.py --webcam --method hog

# Use YOLO detection with collision warning
python collision_warning.py --webcam --method yolo

# Adjust the proximity threshold (default 0.15)
python collision_warning.py --webcam --method yolo --threshold 0.2
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named...**
   - Make sure you've activated the virtual environment
   - Try reinstalling the dependencies: `pip install -r requirements.txt`

2. **CUDA/GPU errors**
   - If you have GPU issues, try forcing CPU mode: `--device cpu`

3. **Low detection accuracy**
   - Try a different detection method
   - For YOLO, use a larger model (yolov5m, yolov5l)
   - Adjust lighting conditions if using a camera

4. **Slow performance**
   - Reduce frame resolution if possible
   - Use a smaller model or faster detection method
   - Close other applications using GPU resources

### Getting Help

If you encounter issues not covered in this guide, please:
- Check the README.md for additional information
- Open an issue on the GitHub repository
- Try searching for the specific error message online 