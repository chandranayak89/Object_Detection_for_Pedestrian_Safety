<<<<<<< HEAD
# Object Detection for Pedestrian Safety

A machine vision project for real-time pedestrian detection to improve safety in autonomous vehicles and smart surveillance systems.

## Features

- Pedestrian detection in images using Haar Cascades
- Real-time pedestrian detection in video streams using HOG+SVM
- Advanced pedestrian detection with YOLOv5/YOLOv8
- Collision warning system for pedestrian safety
- Multiple detection methods for comparison

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/Object_Detection_for_Pedestrian_Safety.git
cd Object_Detection_for_Pedestrian_Safety
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `pedestrian_detection_haar.py`: Basic pedestrian detection using Haar Cascades
- `pedestrian_detection_hog.py`: Pedestrian detection using HOG+SVM
- `pedestrian_detection_yolo.py`: Advanced detection using YOLOv5
- `collision_warning.py`: Pedestrian detection with proximity warning
- `utils.py`: Helper functions for detection and visualization

## Usage

### Haar Cascade Detection
```bash
python pedestrian_detection_haar.py --image path/to/image.jpg
# or for video
python pedestrian_detection_haar.py --video path/to/video.mp4
```

### HOG+SVM Detection
```bash
python pedestrian_detection_hog.py --video path/to/video.mp4
# or for webcam
python pedestrian_detection_hog.py --webcam
```

### YOLO Detection
```bash
python pedestrian_detection_yolo.py --video path/to/video.mp4
```

### Collision Warning System
```bash
python collision_warning.py --video path/to/video.mp4
```

## Dataset

For testing, you can use public pedestrian datasets:
- INRIA Person Dataset
- CrowdHuman Dataset
- MOT Dataset

## Future Enhancements

- Custom YOLOv8 model trained on specialized pedestrian datasets
- Integration with LIDAR or stereo vision for accurate distance measurement
- Edge AI deployment on devices like NVIDIA Jetson Nano 
=======
# Object_Detection_for_Pedestrian_Safety
>>>>>>> 22add91e8885c1f0f943cc73285c96fea1d5ffb2
