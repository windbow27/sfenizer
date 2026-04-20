from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import secrets
import sqlite3
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import cv2
import os
import base64
import json
from use_classifier import predict_board, to_sfen, to_csa
import generate_labels

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:
    YOLO = None

DB_PATH = Path(__file__).with_name('sfenizer.db')
SESSION_TTL_HOURS = 24 * 7

# Load YOLO model for video/live detection when the dependency is available.
yolo_model = YOLO('yolo11m.pt') if YOLO is not None else None


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)


class HistoryCreate(BaseModel):
    timestamp: int
    thumbnail: str
    sfen: str
    csa: str
    board: list[list[str]]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def hash_password(password: str, salt: str | None = None) -> str:
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt_value.encode('utf-8'), 120000)
    return f'{salt_value}${digest.hex()}'


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expected_hash = password_hash.split('$', 1)
    except ValueError:
        return False
    return hash_password(password, salt) == f'{salt}${expected_hash}'


def create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (now_utc() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    with get_db() as conn:
        conn.execute(
            'INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
            (token, user_id, now_utc().isoformat(), expires_at),
        )
        conn.commit()
    return token


def get_bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Authentication required')
    return authorization.removeprefix('Bearer ').strip()


def get_current_user(authorization: str | None = Header(default=None)):
    token = get_bearer_token(authorization)
    with get_db() as conn:
        row = conn.execute(
            '''
            SELECT users.id, users.username
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            ''',
            (token, now_utc().isoformat()),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail='Invalid or expired session')
    return {'id': row['id'], 'username': row['username'], 'token': token}


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                thumbnail TEXT NOT NULL,
                sfen TEXT NOT NULL,
                csa TEXT NOT NULL,
                board_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            '''
        )
        conn.execute('PRAGMA foreign_keys = ON')
        existing_user = conn.execute('SELECT id FROM users WHERE username = ?', ('demo',)).fetchone()
        if existing_user is None:
            conn.execute(
                'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                ('demo', hash_password('demo1234'), now_utc().isoformat()),
            )
        conn.commit()


def row_to_history_item(row: sqlite3.Row) -> dict:
    return {
        'id': row['id'],
        'timestamp': row['timestamp'],
        'thumbnail': row['thumbnail'],
        'sfen': row['sfen'],
        'csa': row['csa'],
        'board': json.loads(row['board_json']),
    }


def save_history_item(user_id: int, item: HistoryCreate) -> dict:
    history_id = secrets.token_urlsafe(12)
    with get_db() as conn:
        conn.execute(
            '''
            INSERT INTO history (id, user_id, timestamp, thumbnail, sfen, csa, board_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                history_id,
                user_id,
                item.timestamp,
                item.thumbnail,
                item.sfen,
                item.csa,
                json.dumps(item.board),
                now_utc().isoformat(),
            ),
        )
        conn.commit()
    return {
        'id': history_id,
        'timestamp': item.timestamp,
        'thumbnail': item.thumbnail,
        'sfen': item.sfen,
        'csa': item.csa,
        'board': item.board,
    }


def delete_session(token: str) -> None:
    with get_db() as conn:
        conn.execute('DELETE FROM sessions WHERE token = ?', (token,))
        conn.commit()


def authenticate(credentials: Credentials) -> dict:
    with get_db() as conn:
        user = conn.execute(
            'SELECT id, username, password_hash FROM users WHERE username = ?',
            (credentials.username,),
        ).fetchone()
        if user is None or not verify_password(credentials.password, user['password_hash']):
            raise HTTPException(status_code=401, detail='Invalid username or password')

    token = create_session_token(user['id'])
    return {'token': token, 'user': {'id': user['id'], 'username': user['username']}}


def register_user(credentials: Credentials) -> dict:
    with get_db() as conn:
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ?',
            (credentials.username,),
        ).fetchone()
        if existing_user is not None:
            raise HTTPException(status_code=409, detail='Username already exists')

        password_hash = hash_password(credentials.password)
        cursor = conn.execute(
            'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
            (credentials.username, password_hash, now_utc().isoformat()),
        )
        conn.commit()

    token = create_session_token(cursor.lastrowid)
    return {'token': token, 'user': {'id': cursor.lastrowid, 'username': credentials.username}}


init_db()

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Sfenizer API is running"}


@app.post('/auth/register')
async def register(credentials: Credentials):
    return register_user(credentials)


@app.post('/auth/login')
async def login(credentials: Credentials):
    return authenticate(credentials)


@app.post('/auth/logout')
async def logout(current_user=Depends(get_current_user)):
    delete_session(current_user['token'])
    return {'success': True}


@app.get('/auth/me')
async def me(current_user=Depends(get_current_user)):
    return {'user': {'id': current_user['id'], 'username': current_user['username']}}


@app.get('/history')
async def list_history(current_user=Depends(get_current_user)):
    with get_db() as conn:
        rows = conn.execute(
            '''
            SELECT id, timestamp, thumbnail, sfen, csa, board_json
            FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC, created_at DESC
            ''',
            (current_user['id'],),
        ).fetchall()
    return {'items': [row_to_history_item(row) for row in rows]}


@app.post('/history')
async def add_history(item: HistoryCreate, current_user=Depends(get_current_user)):
    return save_history_item(current_user['id'], item)


@app.delete('/history')
async def clear_history(current_user=Depends(get_current_user)):
    with get_db() as conn:
        conn.execute('DELETE FROM history WHERE user_id = ?', (current_user['id'],))
        conn.commit()
    return {'success': True}


@app.delete('/history/{history_id}')
async def delete_history_item(history_id: str, current_user=Depends(get_current_user)):
    with get_db() as conn:
        result = conn.execute(
            'DELETE FROM history WHERE id = ? AND user_id = ?',
            (history_id, current_user['id']),
        )
        conn.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail='History item not found')
    return {'success': True}

@app.post("/convert")
async def convert_board(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
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
        if yolo_model is None:
            raise HTTPException(status_code=503, detail='Video detection is unavailable on this server')

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
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
    temp_path = None
    try:
        if yolo_model is None:
            await websocket.send_text(json.dumps({'error': 'Video detection is unavailable on this server'}))
            await websocket.close(code=1011)
            return

        while True:
            data = await websocket.receive_text()
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                cv2.imwrite(temp_path, frame)
                warped, _, _ = generate_labels.detect_board(temp_path)
                os.remove(temp_path)
                temp_path = None
                input_frame = cv2.resize(warped, (640, 640))
            except Exception:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    temp_path = None
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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}