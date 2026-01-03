EXPLAINABLE SCENE UNDERSTANDING :

This project generates human-readable descriptions of images by combining object detection (YOLOv8) and caption generation (BLIP).

Features :

Object detection with YOLOv8
Scene graph construction
Caption generation with BLIP
CPU & GPU support

Explainable output showing detected objects and scene description

FOLDER STRUCTURE

Explainable_Scene_Understanding/
│
├── app.py # Streamlit UI version (main file)

├── explainable_scene_understanding.py # Command-line version

├── yolov8n.pt # YOLO model (auto-downloads if missing)

├── images/ or input images # Your test images

├── outputs/ # YOLO + BLIP results (script version)

├── streamlit_outputs/ # Saved results (Streamlit version)

└── requirements.txt # Dependencies

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

Example Output

Input: sample.png
Detected Objects: ['person', 'bicycle', 'car']
Scene Description: a man riding a bike down a street

Output Files (in streamlit_outputs/):

labeled_sample.png
scene_graph_sample.png
details_sample.txt

