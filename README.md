Explainable Scene Understanding :

This project generates human-readable descriptions of images by combining object detection (YOLOv8) and caption generation (BLIP).

Features :

Object detection with YOLOv8
Scene graph construction
Caption generation with BLIP
CPU & GPU support

Explainable output showing detected objects and scene description

Tech Stack

Python

PyTorch

YOLOv8 (Ultralytics)

BLIP (Salesforce)

OpenCV

Run Instructions
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main script
python explainable_scene_understanding.py