from ultralytics import YOLO
import torch
# Load the model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    name="yolov8n_custom"
)
model.save('model.pt')
print(model)
print(model.cfg)
print(model.device)