# Export from ultralytics YOLO to TensorRT format in order to use it on Jetson Orin Nano

from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'best.engine'
