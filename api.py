import asyncio
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
    from main_shogi import (
        ShogiBoardMapper, StateStabilizer, ShogiEngine,
        draw_info_panel, draw_evaluation_bar, draw_engine_suggestions,
        is_valid_transition,
    )
    _shogi_pipeline = True
except Exception:
    _shogi_pipeline = False
    ShogiBoardMapper = StateStabilizer = ShogiEngine = None
    draw_info_panel = draw_evaluation_bar = draw_engine_suggestions = None
    is_valid_transition = None

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:
    YOLO = None

DB_PATH = Path(__file__).with_name('sfenizer.db')
SESSION_TTL_HOURS = 24 * 7

_SHOGI_MODEL_PATH = Path('runs/detect/train3/weights/best.pt')
shogi_model = YOLO(str(_SHOGI_MODEL_PATH)) if YOLO is not None and _SHOGI_MODEL_PATH.exists() else None


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
        if shogi_model is None:
            raise HTTPException(status_code=503, detail='Video detection is unavailable on this server')

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        input_frame = cv2.resize(frame, (640, 640))

        results = shogi_model.predict(input_frame, conf=0.3, verbose=False)
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


class VideoSession:
    def __init__(self):
        self.mapper = ShogiBoardMapper() if ShogiBoardMapper else None
        self.stabilizer = StateStabilizer(buffer_len=8, threshold=0.6) if StateStabilizer else None
        self.frame_count = 0
        self.current_turn = 'b'
        self.last_stable_sfen = None
        self.last_validated_sfen = None
        self.last_analyzed_state = None
        self.current_evaluation = 0
        self.best_moves: list = []
        self.evaluations: list = []
        self.engine = None
        self.use_engine = False
        if ShogiEngine is not None:
            try:
                eng = ShogiEngine()
                if eng.start():
                    self.engine = eng
                    self.use_engine = True
            except Exception:
                pass

    def process(self, frame: np.ndarray) -> tuple:
        self.frame_count += 1

        if shogi_model is None or self.mapper is None:
            return frame, [["empty"] * 9 for _ in range(9)], to_sfen([["empty"] * 9 for _ in range(9)]), ""

        input_frame = cv2.resize(frame, (640, 640))
        results = shogi_model.predict(input_frame, conf=0.1, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes_arr = results[0].boxes.cls.cpu().numpy()
        names = shogi_model.names
        annotated = input_frame.copy()

        # Calibration
        if self.mapper.homography_matrix is None:
            if not self.mapper.calibrate_dynamic(boxes, classes_arr, names):
                cv2.putText(annotated, "Calibrating — show board at starting position",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
                board = [["empty"] * 9 for _ in range(9)]
                return annotated, board, to_sfen(board), to_csa(board)
        elif self.frame_count % 30 == 0 and self.mapper.should_recalibrate(boxes, classes_arr, names):
            self.mapper.calibrate_dynamic(boxes, classes_arr, names, recalibrate=True)

        self.mapper.draw_grid(annotated)

        # Map detections to board grid
        board = [["empty"] * 9 for _ in range(9)]
        for box, cls_idx in zip(boxes, classes_arr):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            grid_pos = self.mapper.get_grid_pos(cx, cy)
            cls_name = names[int(cls_idx)]
            if grid_pos:
                col, row = grid_pos
                if 0 <= row < 9 and 0 <= col < 9:
                    board[row][col] = cls_name
                    label = self.mapper.class_map.get(cls_name, cls_name.split('_')[0])
                    cv2.putText(annotated, label, (int(cx) - 10, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Stabilise SFEN
        raw_sfen = to_sfen(board).split()[0]
        stable_sfen = self.stabilizer.update(raw_sfen)

        # Validate transition + track turn
        if stable_sfen and stable_sfen != self.last_stable_sfen:
            if is_valid_transition is not None:
                is_valid, _, next_turn, _ = is_valid_transition(self.last_validated_sfen, stable_sfen)
            else:
                is_valid, next_turn = True, None
            if is_valid and next_turn:
                self.last_validated_sfen = stable_sfen
                self.last_stable_sfen = stable_sfen
                self.current_turn = next_turn
            elif not self.last_validated_sfen:
                self.last_validated_sfen = stable_sfen
                self.last_stable_sfen = stable_sfen

        # Engine analysis
        if self.use_engine and stable_sfen and stable_sfen != self.last_analyzed_state:
            self.engine.analyze_position(stable_sfen, time_ms=500, turn=self.current_turn)
            self.last_analyzed_state = stable_sfen
        if self.use_engine and self.engine:
            self.best_moves = self.engine.get_best_moves()
            self.evaluations = self.engine.get_evaluations()
            if self.evaluations and self.evaluations[0] is not None:
                self.current_evaluation = self.evaluations[0]
            if self.best_moves and draw_engine_suggestions:
                draw_engine_suggestions(annotated, self.mapper, self.best_moves, self.evaluations, self.current_turn)

        # Draw UI overlays
        if draw_evaluation_bar:
            draw_evaluation_bar(annotated, self.current_evaluation)
        if draw_info_panel:
            draw_info_panel(annotated, self.current_turn, self.best_moves, self.evaluations,
                            stable_sfen, "OK", True)

        sfen = to_sfen(board)
        csa = to_csa(board)
        return annotated, board, sfen, csa

    def cleanup(self):
        if self.use_engine and self.engine:
            self.engine.stop()


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()
    session = VideoSession()
    try:
        while True:
            data = await websocket.receive_text()
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            annotated, _, sfen, csa = await asyncio.get_event_loop().run_in_executor(
                None, session.process, frame
            )

            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(json.dumps({
                "frame": frame_b64,
                "sfen": sfen,
                "csa": csa,
            }))
    except WebSocketDisconnect:
        pass
    finally:
        session.cleanup()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}