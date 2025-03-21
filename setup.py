from setuptools import setup, find_packages

setup(
    name="pedestrian_safety",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "ultralytics>=8.0.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "torchaudio>=0.7.0",
        "matplotlib>=3.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Object Detection for Pedestrian Safety",
    keywords="computer vision, pedestrian detection, object detection, safety",
    url="https://github.com/your-username/Object_Detection_for_Pedestrian_Safety",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
) 