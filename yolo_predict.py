from ultralytics import YOLO
import cv2

# Load model
model = YOLO('work_dirs/yolov10_shogi5/weights/best.pt')

# Predict on an image
results = model('data/shogi_detection/images/IMG_20251202_222342.jpg')
results[0].show()  # Display with bboxes