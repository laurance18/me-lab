from ultralytics import YOLO
from PIL import Image
import cv2
import os

model_name = "v2.pt"
model = YOLO(os.path.join(os.path.dirname(__file__), "models", model_name))

# from PIL
im1 = Image.open("src\demo.png")
results = model.predict(source=im1, save=True)  # save plotted images
