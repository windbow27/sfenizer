from ultralytics import YOLO

model = YOLO('yolov10m.pt')  #
model.train(
    data='data/shogi_detection/data.yaml',
    epochs=50,
    imgsz=640,
    augment=True,
    hsv_h=0.1, hsv_s=0.5, hsv_v=0.3,  
    project='work_dirs',
    name='yolov10_shogi'  
)