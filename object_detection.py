from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
image_path = "sample.png"
image = cv2.imread(image_path)
results = model(image)
annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
