import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO('work_dirs/yolov10_shogi9/weights/best.pt')

# Function to get tile from pixel coordinates (assuming warped 640x640 image)
def get_tile(x, y, img_width=640, img_height=640):
    tile_size_x = img_width / 9
    tile_size_y = img_height / 9
    col = int(x // tile_size_x) + 1  # 1-9
    row = int(y // tile_size_y) + 1  # 1-9
    return f"{row}-{col}"  # Shogi notation (row-col)

# Predict
results = model.predict(source='0a582a6c-MVIMG_20251202_231419.jpg', conf=0.1)

for result in results:
    img = cv2.imread(result.path)
    # Optional: Warp image here if needed (use cv2.findHomography for corners)
    # For simplicity, assume image is already top-down
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        tile = get_tile(center_x, center_y)
        cls = result.names[int(box.cls)]
        print(f"Piece: {cls}")  # Removed tile from console print
        # Draw piece name on image instead of tile
        cv2.putText(img, cls, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    cv2.imwrite('output.jpg', img)