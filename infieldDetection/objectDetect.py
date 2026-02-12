from ultralytics import YOLO
import time
# Load a model
model = YOLO()

if __name__ == '__main__':

    # Use the model
    results = model.train(data="data.yaml", epochs=120)  # train the model
    success = YOLO("yolov8n.pt").export(format="onnx")

# todo before train add more padding to the soures






 



