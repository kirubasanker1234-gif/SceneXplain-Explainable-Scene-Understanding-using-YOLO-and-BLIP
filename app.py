import os
import io
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

st.set_page_config(page_title="Explainable Scene Understanding", layout="wide")
device = "cpu"
yolo_model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
output_folder = "streamlit_outputs"
os.makedirs(output_folder, exist_ok=True)
st.title("Explainable Scene Understanding")
st.write("Upload any image to detect objects, generate a scene description, and visualize a connected scene graph.")

uploaded_file = st.file_uploader(" Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running YOLO object detection..."):
        results = yolo_model(image)
        detections = results[0].boxes
        object_names = list(set([results[0].names[int(cls)] for cls in detections.cls]))

    with st.spinner("Generating scene description (BLIP)..."):
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    
    st.subheader("Detected Objects")
    st.write(object_names)
    st.subheader("Scene Description")
    st.write(caption)
    draw = ImageDraw.Draw(image)
    for box, cls in zip(detections.xyxy, detections.cls):
        x1, y1, x2, y2 = box.tolist()
        label = results[0].names[int(cls)]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")
    st.image(image, caption="Labeled Image", use_column_width=True)
    relationships = []
    if "person" in object_names and "bicycle" in object_names:
        relationships.append(("person", "bicycle", "riding"))
    if "person" in object_names and "car" in object_names:
        relationships.append(("person", "car", "near"))
    if len(object_names) > 2 and not relationships:
        for i in range(len(object_names) - 1):
            relationships.append((object_names[i], object_names[i + 1], "near"))
    if relationships:
        G = nx.DiGraph()
        for subj, obj, rel in relationships:
            G.add_edge(subj, obj, label=rel)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(5, 3))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1800, font_size=10)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Scene Graph", use_column_width=False)
        plt.close()
        base_name = os.path.splitext(uploaded_file.name)[0]
        labeled_path = os.path.join(output_folder, f"labeled_{base_name}.png")
        graph_path = os.path.join(output_folder, f"scene_graph_{base_name}.png")
        text_path = os.path.join(output_folder, f"details_{base_name}.txt")
        image.save(labeled_path)
        with open(graph_path, "wb") as f:
            f.write(buf.getbuffer())
        with open(text_path, "w") as f:
            f.write(f"Scene Description: {caption}\n\nDetected Objects: {object_names}\n\nRelationships: {relationships}")
        st.success(f" Results saved in folder: {output_folder}")
        st.write(f"- Labeled image: `{labeled_path}`")
        st.write(f"-Scene graph: `{graph_path}`")
        st.write(f"-Description file: `{text_path}`")
    else:
        st.info("No strong object relationships detected.")
