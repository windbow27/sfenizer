import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import shogi
import subprocess
import threading
import queue

# ==========================================
# 1. BOARD MAPPING & CALIBRATION LOGIC
# ==========================================
class ShogiBoardMapper:
    def __init__(self):
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.calibration_confidence = 0.0
        self.last_calibration_pieces = {}
        
        # shogi piece class to SFEN character mapping
        self.class_map = {
            'L_black': 'L', 'L_white': 'l',
            'N_black': 'N', 'N_white': 'n',
            'S_black': 'S', 'S_white': 's',
            'G_black': 'G', 'G_white': 'g',
            'K_black': 'K', 'K_white': 'k',
            'R_black': 'R', 'R_white': 'r',
            'B_black': 'B', 'B_white': 'b',
            'P_black': 'P', 'P_white': 'p',
        }
        
        # starting positions 
        self.starting_positions = {
            'L_white': [(0, 0), (8, 0)],
            'L_black': [(0, 8), (8, 8)],
            'N_white': [(1, 0), (7, 0)],
            'N_black': [(1, 8), (7, 8)],
            'S_white': [(2, 0), (6, 0)],
            'S_black': [(2, 8), (6, 8)],
            'G_white': [(3, 0), (5, 0)],
            'G_black': [(3, 8), (5, 8)],
            'K_white': [(4, 0)],
            'K_black': [(4, 8)],
            'R_white': [(7, 1)],
            'B_white': [(1, 1)],
            'R_black': [(1, 7)],
            'B_black': [(7, 7)],
        }

    def calibrate(self, boxes, classes, names):
        """Original calibration method using 4 corner Lances."""
        lances = []
        
        for box, cls_idx in zip(boxes, classes):
            cls_name = names[int(cls_idx)]
            if 'L_' in cls_name:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                lances.append((cx, cy))

        if len(lances) != 4:
            return False

        lances = sorted(lances, key=lambda p: p[1])
        top_row = sorted(lances[:2], key=lambda p: p[0])
        btm_row = sorted(lances[2:], key=lambda p: p[0])
        
        src_points = np.float32([top_row[0], top_row[1], btm_row[1], btm_row[0]])

        self.dst_w = 900
        self.dst_h = 990
        
        cell_w = self.dst_w / 9
        cell_h = self.dst_h / 9
        
        dst_points = np.float32([
            [0.5 * cell_w, 0.5 * cell_h],
            [self.dst_w - 0.5 * cell_w, 0.5 * cell_h],
            [self.dst_w - 0.5 * cell_w, self.dst_h - 0.5 * cell_h],
            [0.5 * cell_w, self.dst_h - 0.5 * cell_h]
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_homography_matrix = cv2.invert(self.homography_matrix)[1]
        
        self.cell_w = cell_w
        self.cell_h = cell_h
        
        return True

    def calibrate_dynamic(self, boxes, classes, names, recalibrate=False):
        """Dynamic calibration using multiple piece types."""
        detected_pieces = {}
        
        for box, cls_idx in zip(boxes, classes):
            cls_name = names[int(cls_idx)]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if cls_name not in detected_pieces:
                detected_pieces[cls_name] = []
            detected_pieces[cls_name].append((cx, cy))
        
        src_points = []
        dst_points = []
        
        for piece_type, expected_positions in self.starting_positions.items():
            if piece_type in detected_pieces:
                detected = detected_pieces[piece_type]
                
                if len(detected) == len(expected_positions):
                    if len(detected) == 2:
                        detected = sorted(detected, key=lambda p: p[0])
                    
                    for (det_x, det_y), (grid_col, grid_row) in zip(detected, expected_positions):
                        src_points.append([det_x, det_y])
                        dst_x = (grid_col + 0.5) * (900 / 9)
                        dst_y = (grid_row + 0.5) * (990 / 9)
                        dst_points.append([dst_x, dst_y])
        
        if len(src_points) < 4:
            return False
        
        src_array = np.float32(src_points)
        dst_array = np.float32(dst_points)
        
        try:
            M, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 5.0)
            
            if M is not None:
                confidence = np.sum(mask) / len(mask) if mask is not None else 0.0
                
                if recalibrate or self.homography_matrix is None or confidence > self.calibration_confidence:
                    self.homography_matrix = M
                    self.inv_homography_matrix = cv2.invert(M)[1]
                    self.calibration_confidence = confidence
                    self.last_calibration_pieces = detected_pieces
                    
                    self.dst_w = 900
                    self.dst_h = 990
                    self.cell_w = self.dst_w / 9
                    self.cell_h = self.dst_h / 9
                    
                    return True
        except:
            pass
        
        return False

    def should_recalibrate(self, boxes, classes, names):
        """Checks if board position has shifted significantly."""
        if self.homography_matrix is None:
            return True
        
        current_pieces = {}
        for box, cls_idx in zip(boxes, classes):
            cls_name = names[int(cls_idx)]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if cls_name not in current_pieces:
                current_pieces[cls_name] = []
            current_pieces[cls_name].append((cx, cy))
        
        shift_threshold = 50
        
        for piece_type, last_positions in self.last_calibration_pieces.items():
            if piece_type in current_pieces:
                curr_pos = current_pieces[piece_type]
                
                for last_p in last_positions:
                    min_dist = min([np.linalg.norm(np.array(curr_p) - np.array(last_p)) 
                                   for curr_p in curr_pos], default=float('inf'))
                    if min_dist > shift_threshold:
                        return True
        
        return False

    def get_grid_pos(self, cx, cy):
        """Maps pixel (x,y) to (col, row) indices 0-8."""
        if self.homography_matrix is None:
            return None
            
        point = np.array([[[cx, cy]]], dtype='float32')
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        tx, ty = transformed[0][0]
        
        col = int(tx // self.cell_w)
        row = int(ty // self.cell_h)
        
        if 0 <= col < 9 and 0 <= row < 9:
            return (col, row)
        return None

    def draw_grid(self, img):
        """Visualizes the grid overlay."""
        if self.inv_homography_matrix is None:
            return img
            
        for i in range(10):
            x = i * self.cell_w
            pt1 = np.array([[[x, 0]]], dtype='float32')
            pt2 = np.array([[[x, self.dst_h]]], dtype='float32')
            
            t_pt1 = cv2.perspectiveTransform(pt1, self.inv_homography_matrix)[0][0]
            t_pt2 = cv2.perspectiveTransform(pt2, self.inv_homography_matrix)[0][0]
            
            cv2.line(img, (int(t_pt1[0]), int(t_pt1[1])), (int(t_pt2[0]), int(t_pt2[1])), (0, 255, 255), 1)
            
        for i in range(10):
            y = i * self.cell_h
            pt1 = np.array([[[0, y]]], dtype='float32')
            pt2 = np.array([[[self.dst_w, y]]], dtype='float32')
            
            t_pt1 = cv2.perspectiveTransform(pt1, self.inv_homography_matrix)[0][0]
            t_pt2 = cv2.perspectiveTransform(pt2, self.inv_homography_matrix)[0][0]
            
            cv2.line(img, (int(t_pt1[0]), int(t_pt1[1])), (int(t_pt2[0]), int(t_pt2[1])), (0, 255, 255), 1)
            
        return img

# ==========================================
# 2. STATE STABILIZATION
# ==========================================
class StateStabilizer:
    def __init__(self, buffer_len=10, threshold=0.5):
        self.buffer = deque(maxlen=buffer_len)
        self.threshold = threshold
        self.last_valid_sfen = None

    def update(self, sfen, is_valid=True):
        """Update buffer and return stable state."""
        self.buffer.append(sfen)
        if len(self.buffer) < self.buffer.maxlen:
            return None
        
        most_common, count = Counter(self.buffer).most_common(1)[0]
        if count / len(self.buffer) >= self.threshold:
            if is_valid:
                self.last_valid_sfen = most_common
            return most_common
        return None
    
    def get_last_valid(self):
        """Return last validated stable state."""
        return self.last_valid_sfen

# ==========================================
# 3. SHOGI ENGINE INTEGRATION
# ==========================================
class ShogiEngine:
    """
    Wrapper for Shogi engine 
    """
    def __init__(self, engine_path=None):
        self.engine_path = engine_path or "fairy-stockfish"
        self.process = None
        self.move_queue = queue.Queue()
        self.best_moves = []
        self.evaluations = []
        self.current_turn = 'b'
        
    def start(self):
        """Start the engine process."""
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            self._send_command("usi")
            self._wait_for("usiok")
            self._send_command("setoption name MultiPV value 3")
            self._send_command("isready")
            self._wait_for("readyok")
            
            print(">>> SHOGI ENGINE STARTED <<<")
            return True
        except Exception as e:
            print(f"Failed to start engine: {e}")
            return False
    
    def _send_command(self, cmd):
        """Send command to engine."""
        if self.process:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
    
    def _wait_for(self, expected):
        """Wait for specific response from engine."""
        if self.process:
            while True:
                line = self.process.stdout.readline().strip()
                if expected in line:
                    return line
    
    def analyze_position(self, sfen, time_ms=1000, turn='b'):
        """
        Analyze position and get top 3 moves for current player.
        """
        self.current_turn = turn
        
        def _analyze():
            try:
                self.best_moves = []
                self.evaluations = []
                
                self._send_command(f"position sfen {sfen} {turn} - 1")
                self._send_command(f"go movetime {time_ms}")
                
                temp_moves = {}
                
                while True:
                    line = self.process.stdout.readline().strip()
                    
                    if "bestmove" in line:
                        break
                    
                    if "multipv" in line and "pv" in line:
                        try:
                            parts = line.split()
                            multipv_idx = parts.index("multipv")
                            pv_idx = parts.index("pv")
                            
                            multipv_num = int(parts[multipv_idx + 1])
                            move = parts[pv_idx + 1]
                            
                            eval_score = None
                            if "score cp" in line:
                                cp_idx = parts.index("cp")
                                eval_score = int(parts[cp_idx + 1])
                                if turn == 'w':
                                    eval_score = -eval_score
                            
                            temp_moves[multipv_num] = (move, eval_score)
                        except:
                            pass
                
                for i in range(1, 4):
                    if i in temp_moves:
                        move, eval_score = temp_moves[i]
                        self.best_moves.append(move)
                        self.evaluations.append(eval_score)
                            
            except Exception as e:
                print(f"Engine analysis error: {e}")
        
        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()
    
    def get_best_moves(self):
        """Get the top 3 moves (if available)."""
        return self.best_moves
    
    def get_evaluations(self):
        """Get position evaluations for top moves."""
        return self.evaluations
    
    def get_current_turn(self):
        """Get whose turn it is being analyzed."""
        return self.current_turn
    
    def stop(self):
        """Stop the engine."""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process = None

# ==========================================
# 4. HELPER: GRID TO SFEN & VALIDATION
# ==========================================
def grid_to_sfen(grid_dict):
    """Converts a dict {(col, row): 'P'} into an SFEN string."""
    sfen_rows = []
    for r in range(9):
        row_str = ""
        empty_count = 0
        for c in range(9):
            piece = grid_dict.get((c, r))
            if piece:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece
            else:
                empty_count += 1
        if empty_count > 0:
            row_str += str(empty_count)
        sfen_rows.append(row_str)
    return "/".join(sfen_rows)

def is_valid_transition(prev_sfen, new_sfen):
    """
    Check if new_sfen is reachable from prev_sfen in one legal move.
    Returns (is_valid, error_message, turn, move_usi)
    """
    if not prev_sfen or not new_sfen:
        return True, "No previous state", None, None
    
    if prev_sfen == new_sfen:
        return True, "Same state", None, None
    
    try:
        for turn in ['b', 'w']:
            board = shogi.Board(f"{prev_sfen} {turn} - 1")
            legal_moves = list(board.legal_moves)
            
            for move in legal_moves:
                test_board = shogi.Board(board.sfen())
                test_board.push(move)
                result_sfen = test_board.sfen().split()[0]
                
                if result_sfen == new_sfen:
                    player = "Black" if turn == 'b' else "White"
                    next_turn = 'w' if turn == 'b' else 'b'
                    return True, f"Valid move: {move.usi()} ({player})", next_turn, move.usi()
        
        return False, "Not reachable in 1 move", None, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", None, None

def detect_incomplete_board(grid_dict):
    """
    Check if detected board is missing critical pieces.
    Returns (is_complete, missing_count)
    """
    has_black_king = 'K' in [p for p in grid_dict.values()]
    has_white_king = 'k' in [p for p in grid_dict.values()]
    
    if not has_black_king or not has_white_king:
        return False, "Missing king(s)"
    
    total_pieces = len(grid_dict)
    if total_pieces < 15:
        return False, f"Only {total_pieces} pieces detected"
    
    return True, "Complete"

def draw_evaluation_bar(frame, evaluation, bar_x=20, bar_y=180, bar_width=40, bar_height=400):
    """Draw a vertical evaluation bar showing position advantage."""
    # Clamp evaluation to reasonable range (-2000 to +2000 centipawns)
    eval_clamped = max(-2000, min(2000, evaluation if evaluation is not None else 0))
    
    # Calculate bar fill percentage (0.0 = white advantage, 1.0 = black advantage)
    fill_ratio = (eval_clamped + 2000) / 4000
    fill_height = int(bar_height * fill_ratio)
    
    # Draw background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    
    # Draw white advantage (top)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + (bar_height - fill_height)), (255, 255, 255), -1)
    
    # Draw black advantage (bottom)
    cv2.rectangle(frame, (bar_x, bar_y + (bar_height - fill_height)), 
                 (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
    
    # Draw center line
    center_y = bar_y + bar_height // 2
    cv2.line(frame, (bar_x, center_y), (bar_x + bar_width, center_y), (128, 128, 128), 2)
    
    # Draw labels
    cv2.putText(frame, "WHITE", (bar_x - 5, bar_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "BLACK", (bar_x - 5, bar_y + bar_height + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw evaluation value
    if evaluation is not None:
        eval_text = f"{evaluation:+d}" if abs(evaluation) < 10000 else ("M+" if evaluation > 0 else "M-")
        text_x = bar_x + bar_width + 10
        text_y = bar_y + bar_height // 2
        cv2.putText(frame, eval_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_enhanced_arrow(frame, start, end, color, thickness, label="", eval_text=""):
    """Draw an enhanced arrow with glow effect and labels."""
    # Draw glow effect
    for i in range(3, 0, -1):
        alpha = 0.3 / i
        glow_color = tuple(int(c * alpha + 20) for c in color)
        cv2.arrowedLine(frame, start, end, glow_color, thickness + i * 2, tipLength=0.25)
    
    # Draw main arrow
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.25)
    
    # Draw move label
    if label:
        # Calculate label position (slightly offset from arrow end)
        mid_x = int((start[0] + end[0]) / 2)
        mid_y = int((start[1] + end[1]) / 2)
        
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (mid_x - 5, mid_y - text_size[1] - 5),
                     (mid_x + text_size[0] + 5, mid_y + 5),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, 
                     (mid_x - 5, mid_y - text_size[1] - 5),
                     (mid_x + text_size[0] + 5, mid_y + 5),
                     color, 2)
        
        cv2.putText(frame, label, (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw evaluation below
        if eval_text:
            cv2.putText(frame, eval_text, (mid_x, mid_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_engine_suggestions(frame, mapper, best_moves, evaluations, turn):
    """Draw top 3 engine suggestions with enhanced arrows."""
    if not best_moves or mapper.inv_homography_matrix is None:
        return
    
    colors = [(0, 255, 0), (255, 180, 0), (255, 100, 255)]  # Green, Orange, Pink
    thicknesses = [4, 3, 2]
    
    for idx, (best_move, color, thickness) in enumerate(zip(best_moves[:3], colors, thicknesses)):
        try:
            if len(best_move) >= 4:
                from_col = 9 - int(best_move[0])
                from_row = ord(best_move[1]) - ord('a')
                to_col = 9 - int(best_move[2])
                to_row = ord(best_move[3]) - ord('a')
                
                from_x = (from_col + 0.5) * mapper.cell_w
                from_y = (from_row + 0.5) * mapper.cell_h
                to_x = (to_col + 0.5) * mapper.cell_w
                to_y = (to_row + 0.5) * mapper.cell_h
                
                from_pt = cv2.perspectiveTransform(
                    np.array([[[from_x, from_y]]], dtype='float32'),
                    mapper.inv_homography_matrix
                )[0][0]
                
                to_pt = cv2.perspectiveTransform(
                    np.array([[[to_x, to_y]]], dtype='float32'),
                    mapper.inv_homography_matrix
                )[0][0]
                
                # Create label
                label = f"#{idx+1}"
                eval_text = f"({evaluations[idx]})" if idx < len(evaluations) and evaluations[idx] is not None else ""
                
                draw_enhanced_arrow(frame, 
                                  (int(from_pt[0]), int(from_pt[1])),
                                  (int(to_pt[0]), int(to_pt[1])),
                                  color, thickness, label, eval_text)
                
        except Exception as e:
            print(f"Error drawing move {idx+1}: {e}")

def draw_info_panel(frame, current_turn, best_moves, evaluations, stable_sfen, validation_msg, is_complete):
    """Draw information panel with game state."""
    panel_height = 120
    panel_color = (40, 40, 40)
    
    # Draw semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), panel_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_height), (100, 100, 100), 2)
    
    y_offset = 30
    x_start = 100
    
    # Turn indicator
    turn_text = "BLACK" if current_turn == 'b' else "WHITE"
    turn_color = (255, 255, 255) if current_turn == 'b' else (200, 200, 200)
    cv2.putText(frame, f"Turn: {turn_text}", (x_start, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2)
    
    # Best move
    if best_moves:
        cv2.putText(frame, f"Best: {best_moves[0]}", (x_start + 250, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Evaluation
    if evaluations and evaluations[0] is not None:
        eval_color = (100, 255, 100) if evaluations[0] > 0 else (100, 100, 255)
        cv2.putText(frame, f"Eval: {evaluations[0]:+d}", (x_start + 500, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eval_color, 2)
    
    # SFEN
    y_offset += 35
    sfen_display = stable_sfen[:50] + "..." if stable_sfen and len(stable_sfen) > 50 else (stable_sfen or "Stabilizing...")
    cv2.putText(frame, f"SFEN: {sfen_display}", (x_start, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
    
    # Status
    # y_offset += 30
    # if not is_complete:
    #     status_text = "⚠ Incomplete Board"
    #     status_color = (0, 165, 255)
    # elif "Valid" in validation_msg or "Initial" in validation_msg:
    #     status_text = "✓ Valid Position"
    #     status_color = (0, 255, 0)
    # else:
    #     status_text = f"✗ {validation_msg}"
    #     status_color = (0, 0, 255)
    
    # cv2.putText(frame, status_text, (x_start, y_offset), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Top 3 moves list
    y_offset += 35
    if best_moves:
        moves_text = " | ".join([f"{i+1}. {move}" for i, move in enumerate(best_moves[:3])])
        cv2.putText(frame, f"Top Moves: {moves_text}", (x_start, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
def main():
    # SETUP
    video_path = 'vid/board_07.mp4' 
    model_path = 'runs/detect/train9/weights/best.pt'
    engine_path = None
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    mapper = ShogiBoardMapper()
    stabilizer = StateStabilizer(buffer_len=12, threshold=0.6)
    engine = ShogiEngine(engine_path)
    
    use_engine = engine.start()
    
    output_file = open('sfen_predictions.txt', 'w')
    output_file.write("Frame | Raw SFEN | Stable SFEN | Valid Transition | Turn | Best Moves | Evaluations | Detections\n")
    output_file.write("="*140 + "\n")
    
    frame_count = 0
    recalibration_interval = 30
    use_dynamic_calibration = True
    last_stable_sfen = None
    last_validated_sfen = None
    current_turn = 'b'
    last_analyzed_state = None
    current_evaluation = 0
    
    print("Starting processing... Press 'q' to quit.")
    print(f"Calibration mode: {'Dynamic (Multi-piece)' if use_dynamic_calibration else 'Static (4-lance)'}")
    print(f"Engine: {'Enabled' if use_engine else 'Disabled'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
            
        results = model.predict(frame, conf=0.1, verbose=False)  
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = model.names

        # 1. CALIBRATION
        if mapper.homography_matrix is None:
            if use_dynamic_calibration:
                success = mapper.calibrate_dynamic(boxes, classes, names)
                if success:
                    print(f">>> DYNAMIC CALIBRATION SUCCESSFUL (Confidence: {mapper.calibration_confidence:.2f}) <<<")
                else:
                    cv2.putText(frame, "Calibrating... (Detecting starting position)", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Shogi AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue
            else:
                success = mapper.calibrate(boxes, classes, names)
                if success:
                    print(">>> CALIBRATION SUCCESSFUL <<<")
                else:
                    cv2.putText(frame, "Calibrating... (Need 4 Corner Lances)", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Shogi AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue
        
        if use_dynamic_calibration and frame_count % recalibration_interval == 0:
            if mapper.should_recalibrate(boxes, classes, names):
                print(">>> BOARD MOVEMENT DETECTED - RECALIBRATING <<<")
                mapper.calibrate_dynamic(boxes, classes, names, recalibrate=True)
                print(f">>> RECALIBRATION COMPLETE (Confidence: {mapper.calibration_confidence:.2f}) <<<")

        # 2. DRAW GRID
        mapper.draw_grid(frame)

        # 3. MAP PIECES
        current_grid = {}
        detections = []
        for box, cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            grid_pos = mapper.get_grid_pos(cx, cy)
            cls_name = names[int(cls_idx)]
            
            if grid_pos:
                sfen_char = mapper.class_map.get(cls_name, '?')
                current_grid[grid_pos] = sfen_char
                detections.append(f"{cls_name}@{grid_pos}")
                
                # Draw piece character
                cv2.putText(frame, sfen_char, (int(cx)-10, int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                detections.append(f"{cls_name}@({int(cx)},{int(cy)})")

        is_complete, completeness_msg = detect_incomplete_board(current_grid)

        # 4. GENERATE SFEN
        raw_sfen = grid_to_sfen(current_grid)
        
        # 5. STABILIZE
        stable_sfen = stabilizer.update(raw_sfen, is_valid=is_complete)

        # 6. VALIDATE TRANSITION
        is_valid = False
        validation_msg = "N/A"
        detected_move = None
        
        if stable_sfen and stable_sfen != last_stable_sfen and is_complete:
            is_valid, validation_msg, next_turn, detected_move = is_valid_transition(
                last_validated_sfen, stable_sfen
            )
            
            if is_valid and next_turn:
                last_validated_sfen = stable_sfen
                last_stable_sfen = stable_sfen
                print(f"✓ Valid transition: {validation_msg}")
                print(f"Turn changed: {current_turn} -> {next_turn}")
                current_turn = next_turn
            elif not last_validated_sfen:
                last_validated_sfen = stable_sfen
                last_stable_sfen = stable_sfen
                is_valid = True
                validation_msg = "Initial state"
                print(f"Initial position set: {stable_sfen[:30]}...")
            else:
                print(f"✗ Invalid transition: {validation_msg}")

        # 7. ENGINE ANALYSIS
        best_moves_str = "N/A"
        eval_str = "N/A"
        
        should_analyze = (
            use_engine and 
            stable_sfen and 
            is_complete and
            is_valid and
            stable_sfen != last_analyzed_state
        )
        
        if should_analyze:
            full_sfen = f"{stable_sfen}"
            player = "Black" if current_turn == 'b' else "White"
            print(f"Analyzing for {player}'s turn: {stable_sfen[:40]}...")
            engine.analyze_position(full_sfen, time_ms=500, turn=current_turn)
            last_analyzed_state = stable_sfen
        
        best_moves = []
        evaluations = []
        
        if use_engine:
            best_moves = engine.get_best_moves()
            evaluations = engine.get_evaluations()
            
            if best_moves:
                best_moves_str = ", ".join(best_moves)
                eval_str = ", ".join([str(e) if e is not None else "N/A" for e in evaluations])
                
                if evaluations and evaluations[0] is not None:
                    current_evaluation = evaluations[0]
                
                draw_engine_suggestions(frame, mapper, best_moves, evaluations, current_turn)

        # Write to file
        detections_str = ", ".join(detections) if detections else "None"
        stable_sfen_str = stable_sfen if stable_sfen else "Stabilizing..."
        turn_str = "Black" if current_turn == 'b' else "White"
        output_file.write(f"{frame_count} | {raw_sfen} | {stable_sfen_str} | {validation_msg} | {turn_str} | {best_moves_str} | {eval_str} | {detections_str}\n")

        # Draw UI Elements
        draw_evaluation_bar(frame, current_evaluation)
        draw_info_panel(frame, current_turn, best_moves, evaluations, stable_sfen, validation_msg, is_complete)

        cv2.imshow("Shogi Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    output_file.close()
    cap.release()
    cv2.destroyAllWindows()
    if use_engine:
        engine.stop()
    
    print(f"\n>>> Output saved to sfen_predictions.txt <<<")

if __name__ == "__main__":
    main()