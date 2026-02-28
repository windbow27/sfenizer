from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2
import os
import base64
import json
from use_classifier import predict_board, to_sfen, to_csa
import generate_labels
from ultralytics import YOLO

# Load YOLO model for video/live detection
yolo_model = YOLO('yolo11m.pt')

def yolo_to_board(results, img_size=640):
    """Map YOLO bounding box detections to a 9x9 shogi board grid."""
    board = [["empty"] * 9 for _ in range(9)]
    tile_size = img_size / 9
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        col = int(center_x // tile_size)
        row = int(center_y // tile_size)
        cls = results[0].names[int(box.cls)]
        if 0 <= row < 9 and 0 <= col < 9:
            board[row][col] = cls
    return board

app = FastAPI(title="Sfenizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Sfenizer API is running"}

@app.post("/convert")
async def convert_board(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, opencv_image)
        
        try:
            warped_board, corners, original = generate_labels.detect_board(temp_path)
            
            os.remove(temp_path)
            
            predictions = predict_board(warped_board)
            
            sfen = to_sfen(predictions)
            csa = to_csa(predictions)
            
            return JSONResponse(content={
                "success": True,
                "sfen": sfen,
                "csa": csa,
                "board": predictions
            })
            
        except Exception as board_error:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Board detection failed: {str(board_error)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/video-capture")
async def video_capture(file: UploadFile = File(...)):
    """Process a single video frame using YOLO detection."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        temp_path = f"temp_video_frame.jpg"
        try:
            cv2.imwrite(temp_path, frame)
            warped, _, _ = generate_labels.detect_board(temp_path)
            os.remove(temp_path)
            input_frame = cv2.resize(warped, (640, 640))
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            input_frame = cv2.resize(frame, (640, 640))

        results = yolo_model.predict(input_frame, conf=0.3, verbose=False)
        annotated = results[0].plot()
        board = yolo_to_board(results)
        sfen = to_sfen(board)
        csa = to_csa(board)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "success": True,
            "frame": annotated_b64,
            "sfen": sfen,
            "csa": csa,
            "board": board,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """WebSocket endpoint for live camera board detection via YOLO."""
    await websocket.accept()
    temp_path = f"temp_ws_{id(websocket)}.jpg"
    try:
        while True:
            data = await websocket.receive_text()
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            try:
                cv2.imwrite(temp_path, frame)
                warped, _, _ = generate_labels.detect_board(temp_path)
                os.remove(temp_path)
                input_frame = cv2.resize(warped, (640, 640))
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                input_frame = cv2.resize(frame, (640, 640))

            results = yolo_model.predict(input_frame, conf=0.3, verbose=False)
            annotated = results[0].plot()
            board = yolo_to_board(results)
            sfen = to_sfen(board)
            csa = to_csa(board)

            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(json.dumps({
                "frame": annotated_b64,
                "sfen": sfen,
                "csa": csa,
            }))
    except WebSocketDisconnect:
        pass
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}