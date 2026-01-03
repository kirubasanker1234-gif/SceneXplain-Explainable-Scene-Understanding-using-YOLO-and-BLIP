import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cpu"

yolo_model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)

image_path = "sample.png"
image = Image.open(image_path).convert("RGB")

results = yolo_model(image_path)
detections = results[0].boxes
object_names = list(set([results[0].names[int(cls)] for cls in detections.cls]))
print("Detected Objects:", object_names)

inputs = processor(images=image, return_tensors="pt").to(device)
out = blip_model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Scene Description:", caption)

draw = ImageDraw.Draw(image)
for box, cls in zip(detections.xyxy, detections.cls):
    x1, y1, x2, y2 = box.tolist()
    label = results[0].names[int(cls)]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), label, fill="red")

relationships = []
if "person" in object_names and "bicycle" in object_names:
    relationships.append("person riding bicycle")
if "person" in object_names and "car" in object_names:
    relationships.append("person near car")

print("Scene Graph Relationships:", relationships)

image.save("output.png")
print("Output image saved as output.png")
