from ultralytics import YOLO

model = YOLO('yolo11m.pt')  
model = YOLO('yolo11m.pt')  
model.train(
    data='data/shogi_detection/data.yaml',
    epochs=300,               
    patience=50,              
    imgsz=640,
    device=0,             
    batch=8,
    workers=4,
    
    # Augmentation
    mosaic=1.0,               # Mosaic augmentation (combines 4 images)
    mixup=0.1,                # Mixup augmentation
    degrees=10.0,             # Rotation (+/- 10 degrees)
    translate=0.1,            # Translation
    scale=0.5,                # Scaling
    fliplr=0.5,               # Horizontal flip
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Adjusted HSV augmentation
    
    # Optimization
    optimizer='AdamW',       
    lr0=0.001,                # Initial learning rate
    cos_lr=True,              # Cosine LR scheduler
)