from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2
import os
from use_classifier import predict_board, to_sfen, to_csa
import generate_labels

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}