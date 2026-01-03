from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
image_path = "sample.png"  # use your same image
raw_image = Image.open(image_path).convert('RGB')
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Scene Description:")
print(caption)
