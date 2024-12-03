from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # Path to your YOLOv8 .pt file

# This will load the full model, including the architecture and weights
print(model)