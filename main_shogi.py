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
        
        # Mapping YOLO classes to SFEN pieces
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
        
        # Known starting positions for Shogi pieces (col, row) in 9x9 grid
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
            
            cv2.line(img, (int(t_pt1[0]), int(t_pt1[1])), (int(t_pt2[0]), int(t_pt2[1])), (0, 255, 255), 2)
            
        for i in range(10):
            y = i * self.cell_h
            pt1 = np.array([[[0, y]]], dtype='float32')
            pt2 = np.array([[[self.dst_w, y]]], dtype='float32')
            
            t_pt1 = cv2.perspectiveTransform(pt1, self.inv_homography_matrix)[0][0]
            t_pt2 = cv2.perspectiveTransform(pt2, self.inv_homography_matrix)[0][0]
            
            cv2.line(img, (int(t_pt1[0]), int(t_pt1[1])), (int(t_pt2[0]), int(t_pt2[1])), (0, 255, 255), 2)
            
        return img

# ==========================================
# 2. STATE STABILIZATION
# ==========================================
class StateStabilizer:
    def __init__(self, buffer_len=10, threshold=0.5):
        self.buffer = deque(maxlen=buffer_len)
        self.threshold = threshold

    def update(self, sfen):
        self.buffer.append(sfen)
        if len(self.buffer) < self.buffer.maxlen:
            return None
        
        most_common, count = Counter(self.buffer).most_common(1)[0]
        if count / len(self.buffer) >= self.threshold:
            return most_common
        return None

# ==========================================
# 3. SHOGI ENGINE INTEGRATION
# ==========================================
class ShogiEngine:
    """
    Wrapper for Shogi engine (YaneuraOu or Fairy-Stockfish).
    Communicates via USI protocol.
    """
    def __init__(self, engine_path=None):
        self.engine_path = engine_path or "fairy-stockfish"  # or path to YaneuraOu
        self.process = None
        self.move_queue = queue.Queue()
        self.best_move = None
        self.evaluation = None
        
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
            
            # Initialize engine
            self._send_command("usi")
            self._wait_for("usiok")
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
    
    def analyze_position(self, sfen, time_ms=1000):
        """
        Analyze position and get best move.
        Runs in separate thread to avoid blocking.
        """
        def _analyze():
            try:
                # Set position
                self._send_command(f"position sfen {sfen}")
                
                # Start search
                self._send_command(f"go movetime {time_ms}")
                
                # Read engine output
                while True:
                    line = self.process.stdout.readline().strip()
                    
                    if "bestmove" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            self.best_move = parts[1]
                        break
                    
                    # Parse evaluation
                    if "score cp" in line:
                        try:
                            cp_idx = line.split().index("cp")
                            self.evaluation = int(line.split()[cp_idx + 1])
                        except:
                            pass
                            
            except Exception as e:
                print(f"Engine analysis error: {e}")
        
        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()
    
    def get_best_move(self):
        """Get the best move (if available)."""
        return self.best_move
    
    def get_evaluation(self):
        """Get position evaluation in centipawns."""
        return self.evaluation
    
    def stop(self):
        """Stop the engine."""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process = None

# ==========================================
# 4. HELPER: GRID TO SFEN
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

def draw_engine_suggestions(frame, mapper, best_move, evaluation):
    """Draw engine suggestions on the frame."""
    if not best_move or mapper.inv_homography_matrix is None:
        return
    
    try:
        # Parse USI move (e.g., "7g7f")
        if len(best_move) >= 4:
            from_col = 9 - int(best_move[0])
            from_row = ord(best_move[1]) - ord('a')
            to_col = 9 - int(best_move[2])
            to_row = ord(best_move[3]) - ord('a')
            
            # Convert to pixel coordinates
            from_x = (from_col + 0.5) * mapper.cell_w
            from_y = (from_row + 0.5) * mapper.cell_h
            to_x = (to_col + 0.5) * mapper.cell_w
            to_y = (to_row + 0.5) * mapper.cell_h
            
            # Transform back to image space
            from_pt = cv2.perspectiveTransform(
                np.array([[[from_x, from_y]]], dtype='float32'),
                mapper.inv_homography_matrix
            )[0][0]
            
            to_pt = cv2.perspectiveTransform(
                np.array([[[to_x, to_y]]], dtype='float32'),
                mapper.inv_homography_matrix
            )[0][0]
            
            # Draw arrow
            cv2.arrowedLine(frame, 
                          (int(from_pt[0]), int(from_pt[1])),
                          (int(to_pt[0]), int(to_pt[1])),
                          (0, 255, 0), 4, tipLength=0.3)
            
            # Draw move text
            cv2.putText(frame, best_move, 
                       (int(to_pt[0]) + 10, int(to_pt[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error drawing move: {e}")

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
def main():
    # SETUP
    video_path = 'vid/VID_20251211_231620.mp4' 
    model_path = 'work_dirs/yolov10_shogi9/weights/best.pt'
    engine_path = None  # Set to path of YaneuraOu or use "fairy-stockfish" if installed
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    mapper = ShogiBoardMapper()
    stabilizer = StateStabilizer(buffer_len=12, threshold=0.6)
    engine = ShogiEngine(engine_path)
    
    # Start engine
    use_engine = engine.start()
    
    frame_count = 0
    recalibration_interval = 30
    use_dynamic_calibration = True
    last_stable_sfen = None
    
    print("Starting processing... Press 'q' to quit.")
    print(f"Calibration mode: {'Dynamic (Multi-piece)' if use_dynamic_calibration else 'Static (4-lance)'}")
    print(f"Engine: {'Enabled' if use_engine else 'Disabled'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
            
        # Run YOLO
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
        
        # Periodic recalibration check
        if use_dynamic_calibration and frame_count % recalibration_interval == 0:
            if mapper.should_recalibrate(boxes, classes, names):
                print(">>> BOARD MOVEMENT DETECTED - RECALIBRATING <<<")
                mapper.calibrate_dynamic(boxes, classes, names, recalibrate=True)
                print(f">>> RECALIBRATION COMPLETE (Confidence: {mapper.calibration_confidence:.2f}) <<<")

        # 2. DRAW GRID
        mapper.draw_grid(frame)

        # 3. MAP PIECES
        current_grid = {}
        for box, cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            grid_pos = mapper.get_grid_pos(cx, cy)
            
            if grid_pos:
                cls_name = names[int(cls_idx)]
                sfen_char = mapper.class_map.get(cls_name, '?')
                current_grid[grid_pos] = sfen_char
                
                cv2.putText(frame, sfen_char, (int(cx)-10, int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 4. GENERATE SFEN
        raw_sfen = grid_to_sfen(current_grid)
        
        # 5. STABILIZE
        stable_sfen = stabilizer.update(raw_sfen)

        # 6. ENGINE ANALYSIS
        if use_engine and stable_sfen and stable_sfen != last_stable_sfen:
            # New stable position detected - analyze it
            full_sfen = f"{stable_sfen} b - 1"  # Add turn and move number
            engine.analyze_position(full_sfen, time_ms=500)
            last_stable_sfen = stable_sfen
            print(f"Analyzing: {stable_sfen}")
        
        # Draw engine suggestions
        if use_engine:
            best_move = engine.get_best_move()
            evaluation = engine.get_evaluation()
            
            if best_move:
                draw_engine_suggestions(frame, mapper, best_move, evaluation)
                
                # Display engine info
                eval_text = f"Eval: {evaluation}" if evaluation else "Analyzing..."
                cv2.putText(frame, f"Best: {best_move} | {eval_text}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display Stats
        cv2.putText(frame, f"Raw: {raw_sfen[:30]}...", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if stable_sfen:
            cv2.putText(frame, f"Stable: {stable_sfen[:30]}...", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Stabilizing...", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        if use_dynamic_calibration:
            cv2.putText(frame, f"Confidence: {mapper.calibration_confidence:.2f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Shogi AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if use_engine:
        engine.stop()

if __name__ == "__main__":
    main()