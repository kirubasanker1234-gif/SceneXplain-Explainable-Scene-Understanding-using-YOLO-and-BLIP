from ultralytics import YOLO
import cv2
import networkx as nx
import matplotlib.pyplot as plt
model = YOLO('yolov8n.pt')
image_path = "sample.png"  # or .jpg, based on your file
image = cv2.imread(image_path)
results = model(image)
boxes = results[0].boxes
names = results[0].names
objects = []
for box in boxes:
    cls = int(box.cls)
    label = names[cls]
    objects.append(label)
objects = list(set(objects))
G = nx.Graph()
G.add_nodes_from(objects)
for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        G.add_edge(objects[i], objects[j], relation="near")
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Scene Graph Representation")
plt.show()
