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
from use_classifier import to_sfen, to_csa
import generate_labels

try:
    from main_shogi import (
        ShogiBoardMapper, StateStabilizer, ShogiEngine,
        draw_evaluation_bar, draw_engine_suggestions,
    )
except Exception:
    ShogiBoardMapper = StateStabilizer = ShogiEngine = None
    draw_evaluation_bar = draw_engine_suggestions = None

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:
    YOLO = None

# Auth / DB constants
DB_PATH = Path(__file__).with_name('sfenizer.db')
SESSION_TTL_HOURS = 24 * 7
PASSWORD_HASH_ITERATIONS = 120000
SESSION_TOKEN_BYTES = 32
HISTORY_ID_BYTES = 12
PASSWORD_SALT_BYTES = 16

# YOLO / video constants
BOARD_GRID_SIZE = 9
YOLO_IMAGE_SIZE_PX = 640
MAX_VIDEO_FRAME_DIMENSION_PX = 720
HAND_VERTICAL_TOLERANCE_RATIO = 0.3
SHOGI_DETECTION_CONFIDENCE = 0.3
VIDEO_DETECTION_CONFIDENCE = 0.1
JPEG_QUALITY_HIGH = 85
JPEG_QUALITY_STREAM = 75
ENGINE_ANALYSIS_TIME_MS = 500
RECALIBRATION_INTERVAL_FRAMES = 30

_SHOGI_MODEL_PATH = Path('runs/detect/train11/weights/best.pt')
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
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    return connection


def hash_password(password: str, salt: str | None = None) -> str:
    salt_value = salt or secrets.token_hex(PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt_value.encode('utf-8'),
        PASSWORD_HASH_ITERATIONS,
    )
    return f'{salt_value}${digest.hex()}'


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expected_hash = password_hash.split('$', 1)
    except ValueError:
        return False
    return hash_password(password, salt) == f'{salt}${expected_hash}'


def create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    expires_at = (now_utc() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    with get_db() as connection:
        connection.execute(
            'INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
            (token, user_id, now_utc().isoformat(), expires_at),
        )
        connection.commit()
    return token


def get_bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Authentication required')
    return authorization.removeprefix('Bearer ').strip()


def get_current_user(authorization: str | None = Header(default=None)):
    token = get_bearer_token(authorization)
    with get_db() as connection:
        row = connection.execute(
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
    with get_db() as connection:
        connection.executescript(
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
        connection.execute('PRAGMA foreign_keys = ON')
        existing_user = connection.execute('SELECT id FROM users WHERE username = ?', ('demo',)).fetchone()
        if existing_user is None:
            connection.execute(
                'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                ('demo', hash_password('demo1234'), now_utc().isoformat()),
            )
        connection.commit()


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
    history_id = secrets.token_urlsafe(HISTORY_ID_BYTES)
    with get_db() as connection:
        connection.execute(
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
        connection.commit()
    return {
        'id': history_id,
        'timestamp': item.timestamp,
        'thumbnail': item.thumbnail,
        'sfen': item.sfen,
        'csa': item.csa,
        'board': item.board,
    }


def delete_session(token: str) -> None:
    with get_db() as connection:
        connection.execute('DELETE FROM sessions WHERE token = ?', (token,))
        connection.commit()


def authenticate(credentials: Credentials) -> dict:
    with get_db() as connection:
        user = connection.execute(
            'SELECT id, username, password_hash FROM users WHERE username = ?',
            (credentials.username,),
        ).fetchone()
        if user is None or not verify_password(credentials.password, user['password_hash']):
            raise HTTPException(status_code=401, detail='Invalid username or password')

    token = create_session_token(user['id'])
    return {'token': token, 'user': {'id': user['id'], 'username': user['username']}}


def register_user(credentials: Credentials) -> dict:
    with get_db() as connection:
        existing_user = connection.execute(
            'SELECT id FROM users WHERE username = ?',
            (credentials.username,),
        ).fetchone()
        if existing_user is not None:
            raise HTTPException(status_code=409, detail='Username already exists')

        password_hash = hash_password(credentials.password)
        cursor = connection.execute(
            'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
            (credentials.username, password_hash, now_utc().isoformat()),
        )
        connection.commit()

    token = create_session_token(cursor.lastrowid)
    return {'token': token, 'user': {'id': cursor.lastrowid, 'username': credentials.username}}


init_db()

def yolo_to_board(results, image_size=YOLO_IMAGE_SIZE_PX):
    """Map YOLO bounding box detections to a 9x9 shogi board grid."""
    board = [["empty"] * BOARD_GRID_SIZE for _ in range(BOARD_GRID_SIZE)]
    tile_size = image_size / BOARD_GRID_SIZE
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        col = int(center_x // tile_size)
        row = int(center_y // tile_size)
        class_name = results[0].names[int(box.cls)]
        if 0 <= row < BOARD_GRID_SIZE and 0 <= col < BOARD_GRID_SIZE:
            board[row][col] = class_name
    return board


def hand_dict_to_sfen(hand_dict: dict) -> str:
    """Encode a hand dict ({'P': 2, 'p': 1, ...}) as SFEN hand notation."""
    if not hand_dict:
        return "-"
    sfen_order = ['R', 'B', 'G', 'S', 'N', 'L', 'P',
                  'r', 'b', 'g', 's', 'n', 'l', 'p']
    result = ""
    for piece in sfen_order:
        count = hand_dict.get(piece, 0)
        if count > 0:
            if count > 1:
                result += str(count)
            result += piece
    return result or "-"


def split_hand_for_client(hand_dict: dict) -> dict:
    """Split a flat hand dict into {'black': {...}, 'white': {...}} for the frontend."""
    return {
        'black': {key: count for key, count in hand_dict.items() if key.isupper()},
        'white': {key: count for key, count in hand_dict.items() if key.islower()},
    }


def yolo_to_board_and_hand(results, homography_matrix, board_size):
    """
    Map YOLO detections (run on the ORIGINAL image) into a 9x9 board and
    a hand dict, using the homography from detect_board.

    board_size is (max_width, max_height) of the warped board.
    Detections whose center, after applying the homography, falls inside
    [0,max_width] x [0,max_height] are board pieces. Detections to the right
    of the board (transformed_x >= max_width) are Black's (sente) hand; to
    the left (transformed_x < 0) are White's (gote) hand.
    """
    board = [["empty"] * BOARD_GRID_SIZE for _ in range(BOARD_GRID_SIZE)]
    hand: dict[str, int] = {}

    max_width, max_height = board_size
    cell_width = max_width / BOARD_GRID_SIZE
    cell_height = max_height / BOARD_GRID_SIZE

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        class_name = results[0].names[int(box.cls)]

        point = np.array([[[center_x, center_y]]], dtype='float32')
        transformed = cv2.perspectiveTransform(point, homography_matrix)
        transformed_x, transformed_y = transformed[0][0]

        col = int(transformed_x // cell_width)
        row = int(transformed_y // cell_height)

        if 0 <= col < BOARD_GRID_SIZE and 0 <= row < BOARD_GRID_SIZE:
            board[row][col] = class_name
            continue

        # Hand zone: must be vertically near the board.
        if not (-max_height * HAND_VERTICAL_TOLERANCE_RATIO
                <= transformed_y
                <= max_height * (1 + HAND_VERTICAL_TOLERANCE_RATIO)):
            continue

        piece_letter = class_name.split('_')[0]
        if piece_letter == 'K':
            # Kings cannot be in hand.
            continue

        if transformed_x >= max_width:
            hand_char = piece_letter  # Black / sente
        elif transformed_x < 0:
            hand_char = piece_letter.lower()  # White / gote
        else:
            continue
        hand[hand_char] = hand.get(hand_char, 0) + 1

    return board, hand

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
    with get_db() as connection:
        rows = connection.execute(
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
    with get_db() as connection:
        connection.execute('DELETE FROM history WHERE user_id = ?', (current_user['id'],))
        connection.commit()
    return {'success': True}


@app.delete('/history/{history_id}')
async def delete_history_item(history_id: str, current_user=Depends(get_current_user)):
    with get_db() as connection:
        result = connection.execute(
            'DELETE FROM history WHERE id = ? AND user_id = ?',
            (history_id, current_user['id']),
        )
        connection.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail='History item not found')
    return {'success': True}

@app.post("/convert")
async def convert_board(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        if shogi_model is None:
            raise HTTPException(status_code=503, detail="YOLO model is unavailable on this server")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
        cv2.imwrite(temp_path, opencv_image)

        try:
            _, homography_matrix, board_size = generate_labels.detect_board(temp_path)
            os.remove(temp_path)

            # Run YOLO on the ORIGINAL image so off-board (hand) pieces are visible.
            # The homography then projects each detection into warped board-space,
            # where x<0 -> White's hand, x>=max_width -> Black's hand.
            results = shogi_model.predict(opencv_image, conf=SHOGI_DETECTION_CONFIDENCE, verbose=False)
            board, hand = yolo_to_board_and_hand(results, homography_matrix, board_size)

            hand_sfen = hand_dict_to_sfen(hand)
            sfen = to_sfen(board, hands=hand_sfen)
            csa = to_csa(board)

            return JSONResponse(content={
                "success": True,
                "sfen": sfen,
                "csa": csa,
                "board": board,
                "hand": split_hand_for_client(hand),
            })

        except HTTPException:
            raise
        except Exception as board_error:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Board detection failed: {str(board_error)}")

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(error)}")

@app.post("/video-capture")
async def video_capture(file: UploadFile = File(...)):
    """Process a single video frame using YOLO detection."""
    try:
        if shogi_model is None:
            raise HTTPException(status_code=503, detail='Video detection is unavailable on this server')

        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        input_frame = cv2.resize(frame, (YOLO_IMAGE_SIZE_PX, YOLO_IMAGE_SIZE_PX))

        results = shogi_model.predict(input_frame, conf=SHOGI_DETECTION_CONFIDENCE, verbose=False)
        annotated = results[0].plot()
        board = yolo_to_board(results)
        sfen = to_sfen(board)
        csa = to_csa(board)

        _, encoded_buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_HIGH])
        annotated_base64 = base64.b64encode(encoded_buffer).decode('utf-8')

        return JSONResponse(content={
            "success": True,
            "frame": annotated_base64,
            "sfen": sfen,
            "csa": csa,
            "board": board,
        })
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(error)}")


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
                engine_instance = ShogiEngine()
                if engine_instance.start():
                    self.engine = engine_instance
                    self.use_engine = True
            except Exception:
                pass

    def process(self, frame: np.ndarray) -> tuple:
        self.frame_count += 1

        if shogi_model is None or self.mapper is None:
            empty_board = [["empty"] * BOARD_GRID_SIZE for _ in range(BOARD_GRID_SIZE)]
            return frame, empty_board, to_sfen(empty_board), ""

        frame_height, frame_width = frame.shape[:2]
        if max(frame_height, frame_width) > MAX_VIDEO_FRAME_DIMENSION_PX:
            scale = MAX_VIDEO_FRAME_DIMENSION_PX / max(frame_height, frame_width)
            frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
        results = shogi_model.predict(frame, conf=VIDEO_DETECTION_CONFIDENCE,
                                      imgsz=YOLO_IMAGE_SIZE_PX, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()
        names = shogi_model.names
        annotated = frame.copy()

        # Calibration
        if self.mapper.homography_matrix is None:
            if not self.mapper.calibrate_dynamic(boxes, class_indices, names):
                cv2.putText(annotated, "Calibrating — show board at starting position",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
                board = [["empty"] * BOARD_GRID_SIZE for _ in range(BOARD_GRID_SIZE)]
                return annotated, board, to_sfen(board), to_csa(board)
        elif self.frame_count % RECALIBRATION_INTERVAL_FRAMES == 0 and \
                self.mapper.should_recalibrate(boxes, class_indices, names):
            self.mapper.calibrate_dynamic(boxes, class_indices, names, recalibrate=True)

        self.mapper.draw_grid(annotated)

        # Map detections to board grid and hand zones
        board = [["empty"] * BOARD_GRID_SIZE for _ in range(BOARD_GRID_SIZE)]
        hand: dict[str, int] = {}
        for box, class_index in zip(boxes, class_indices):
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            class_name = names[int(class_index)]
            piece_letter = class_name.split('_')[0]

            position = self.mapper.get_position(center_x, center_y)
            if position is None:
                continue

            kind = position[0]
            if kind == 'board':
                col, row = position[1], position[2]
                board[row][col] = class_name
                label = self.mapper.class_map.get(class_name, piece_letter)
                cv2.putText(annotated, label, (int(center_x) - 10, int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif kind in ('hand_black', 'hand_white'):
                if piece_letter == 'K':
                    continue
                hand_char = piece_letter if kind == 'hand_black' else piece_letter.lower()
                hand[hand_char] = hand.get(hand_char, 0) + 1
                cv2.putText(annotated, hand_char, (int(center_x) - 10, int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Stabilise on board + hand together
        hand_sfen = hand_dict_to_sfen(hand)
        raw_board_sfen = to_sfen(board).split()[0]
        raw_sfen = f"{raw_board_sfen} {hand_sfen}"
        stable_sfen = self.stabilizer.update(raw_sfen)

        # Track turn: toggle on each confirmed stable SFEN change
        if stable_sfen and stable_sfen != self.last_stable_sfen:
            if self.last_stable_sfen is not None:
                self.current_turn = 'w' if self.current_turn == 'b' else 'b'
            self.last_stable_sfen = stable_sfen

        # Engine analysis - stable_sfen is "board hand"; engine wants them split
        if self.use_engine and stable_sfen and stable_sfen != self.last_analyzed_state:
            parts = stable_sfen.split(" ", 1)
            stable_board = parts[0]
            stable_hand = parts[1] if len(parts) > 1 and parts[1] else "-"
            self.engine.analyze_position(stable_board, stable_hand,
                                         time_ms=ENGINE_ANALYSIS_TIME_MS, turn=self.current_turn)
            self.last_analyzed_state = stable_sfen
        if self.use_engine and self.engine:
            self.best_moves = self.engine.get_best_moves()
            self.evaluations = self.engine.get_evaluations()
            if self.evaluations and self.evaluations[0] is not None:
                self.current_evaluation = self.evaluations[0]
            if self.best_moves and draw_engine_suggestions:
                draw_engine_suggestions(annotated, self.mapper, self.best_moves,
                                        self.evaluations, self.current_turn)

        # Draw eval bar
        if draw_evaluation_bar:
            draw_evaluation_bar(annotated, self.current_evaluation)

        # Info panel — drawn from x=10 so it sits flush left on any frame width
        annotated_height = annotated.shape[0]
        panel_height = min(110, int(annotated_height * 0.13))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], panel_height), (100, 100, 100), 2)

        turn_text = 'Turn: BLACK' if self.current_turn == 'b' else 'Turn: WHITE'
        turn_color = (255, 255, 255) if self.current_turn == 'b' else (200, 200, 200)
        cv2.putText(annotated, turn_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, turn_color, 1)
        if self.best_moves:
            cv2.putText(annotated, f'Best: {self.best_moves[0]}',
                        (10 + annotated.shape[1] // 3 + 16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if self.evaluations and self.evaluations[0] is not None:
            eval_color = (100, 255, 100) if self.evaluations[0] > 0 else (100, 100, 255)
            cv2.putText(annotated, f'Eval: {self.evaluations[0]:+d}',
                        (10 + 2 * annotated.shape[1] // 3, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, eval_color, 2)
        sfen_display = (
            stable_sfen[:40] + '...'
            if stable_sfen and len(stable_sfen) > 40
            else (stable_sfen or 'Stabilizing...')
        )
        cv2.putText(annotated, f'SFEN: {sfen_display}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)
        if self.best_moves:
            moves_text = ' | '.join(f'{rank + 1}. {move}' for rank, move in enumerate(self.best_moves[:3]))
            cv2.putText(annotated, f'Top: {moves_text}', (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        sfen = to_sfen(board, hands=hand_sfen)
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
            np_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            annotated, _, sfen, csa = await asyncio.get_event_loop().run_in_executor(
                None, session.process, frame
            )

            _, encoded_buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_STREAM])
            frame_base64 = base64.b64encode(encoded_buffer).decode('utf-8')

            await websocket.send_text(json.dumps({
                "frame": frame_base64,
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
