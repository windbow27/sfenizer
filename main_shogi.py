import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import shogi
import subprocess
import threading
import queue

BOARD_GRID_SIZE = 9                      
BOARD_WIDTH_PX = 900                      
BOARD_HEIGHT_PX = 990                     
NOISY_DETECTION_THRESHOLD = 55            # max raw detections before skipping a frame
EVAL_CLAMP_CENTIPAWNS = 2000              # eval bar saturation point
HAND_VERTICAL_TOLERANCE_RATIO = 0.3       # how far above/below the board a hand piece may sit
BOARD_SHIFT_THRESHOLD_PX = 50             # piece movement triggering recalibration
MIN_BOARD_PIECES = 15                     # treat the position as incomplete 
DEFAULT_RECALIBRATION_INTERVAL = 30       # frames between recalibration checks
ANALYSIS_DEBOUNCE_FRAMES = 30             # ~1s at 30fps between engine analyses

# 1. BOARD MAPPING & CALIBRATION LOGIC
class ShogiBoardMapper:
    def __init__(self):
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.calibration_confidence = 0.0
        self.last_calibration_pieces = {}

        # Shogi piece class to SFEN
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

        # Starting positions
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

        for box, class_index in zip(boxes, classes):
            class_name = names[int(class_index)]
            if 'L_' in class_name:
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                lances.append((center_x, center_y))

        if len(lances) != 4:
            return False

        lances = sorted(lances, key=lambda point: point[1])
        top_row = sorted(lances[:2], key=lambda point: point[0])
        bottom_row = sorted(lances[2:], key=lambda point: point[0])

        src_points = np.float32([top_row[0], top_row[1], bottom_row[1], bottom_row[0]])

        self.board_width = BOARD_WIDTH_PX
        self.board_height = BOARD_HEIGHT_PX

        cell_width = self.board_width / BOARD_GRID_SIZE
        cell_height = self.board_height / BOARD_GRID_SIZE

        dst_points = np.float32([
            [0.5 * cell_width, 0.5 * cell_height],
            [self.board_width - 0.5 * cell_width, 0.5 * cell_height],
            [self.board_width - 0.5 * cell_width, self.board_height - 0.5 * cell_height],
            [0.5 * cell_width, self.board_height - 0.5 * cell_height]
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_homography_matrix = cv2.invert(self.homography_matrix)[1]

        self.cell_width = cell_width
        self.cell_height = cell_height

        return True

    def calibrate_dynamic(self, boxes, classes, names, recalibrate=False):
        """Dynamic calibration using multiple piece types."""
        detected_pieces = {}

        for box, class_index in zip(boxes, classes):
            class_name = names[int(class_index)]
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if class_name not in detected_pieces:
                detected_pieces[class_name] = []
            detected_pieces[class_name].append((center_x, center_y))

        src_points = []
        dst_points = []

        for piece_type, expected_positions in self.starting_positions.items():
            if piece_type in detected_pieces:
                detected = detected_pieces[piece_type]

                if len(detected) == len(expected_positions):
                    if len(detected) == 2:
                        detected = sorted(detected, key=lambda point: point[0])

                    for (detected_x, detected_y), (grid_col, grid_row) in zip(detected, expected_positions):
                        src_points.append([detected_x, detected_y])
                        target_x = (grid_col + 0.5) * (BOARD_WIDTH_PX / BOARD_GRID_SIZE)
                        target_y = (grid_row + 0.5) * (BOARD_HEIGHT_PX / BOARD_GRID_SIZE)
                        dst_points.append([target_x, target_y])

        if len(src_points) < 4:
            return False

        src_array = np.float32(src_points)
        dst_array = np.float32(dst_points)

        try:
            homography, inlier_mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 5.0)

            if homography is not None:
                confidence = np.sum(inlier_mask) / len(inlier_mask) if inlier_mask is not None else 0.0

                if recalibrate or self.homography_matrix is None or confidence > self.calibration_confidence:
                    self.homography_matrix = homography
                    self.inv_homography_matrix = cv2.invert(homography)[1]
                    self.calibration_confidence = confidence
                    self.last_calibration_pieces = detected_pieces

                    self.board_width = BOARD_WIDTH_PX
                    self.board_height = BOARD_HEIGHT_PX
                    self.cell_width = self.board_width / BOARD_GRID_SIZE
                    self.cell_height = self.board_height / BOARD_GRID_SIZE

                    return True
        except Exception:
            pass

        return False

    def should_recalibrate(self, boxes, classes, names):
        """Checks if board position has shifted significantly."""
        if self.homography_matrix is None:
            return True

        current_pieces = {}
        for box, class_index in zip(boxes, classes):
            class_name = names[int(class_index)]
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if class_name not in current_pieces:
                current_pieces[class_name] = []
            current_pieces[class_name].append((center_x, center_y))

        for piece_type, last_positions in self.last_calibration_pieces.items():
            if piece_type in current_pieces:
                current_positions = current_pieces[piece_type]

                for last_position in last_positions:
                    min_distance = min(
                        [np.linalg.norm(np.array(current_position) - np.array(last_position))
                         for current_position in current_positions],
                        default=float('inf'),
                    )
                    if min_distance > BOARD_SHIFT_THRESHOLD_PX:
                        return True

        return False

    def get_grid_pos(self, center_x, center_y):
        """Maps pixel (x,y) to (col, row) indices 0-8."""
        if self.homography_matrix is None:
            return None

        point = np.array([[[center_x, center_y]]], dtype='float32')
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        transformed_x, transformed_y = transformed[0][0]

        col = int(transformed_x // self.cell_width)
        row = int(transformed_y // self.cell_height)

        if 0 <= col < BOARD_GRID_SIZE and 0 <= row < BOARD_GRID_SIZE:
            return (col, row)
        return None

    def get_position(self, center_x, center_y):
        """
        Classify a detection as ('board', col, row), ('hand_black',), ('hand_white',), or None.
        Right side of board (transformed_x >= board_width) is Black's (sente) hand.
        Left side (transformed_x < 0) is White's (gote) hand.
        """
        if self.homography_matrix is None:
            return None

        point = np.array([[[center_x, center_y]]], dtype='float32')
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        transformed_x, transformed_y = transformed[0][0]

        col = int(transformed_x // self.cell_width)
        row = int(transformed_y // self.cell_height)

        if 0 <= col < BOARD_GRID_SIZE and 0 <= row < BOARD_GRID_SIZE:
            return ('board', col, row)

        # Vertical tolerance for hand area
        y_min = -self.board_height * HAND_VERTICAL_TOLERANCE_RATIO
        y_max = self.board_height * (1 + HAND_VERTICAL_TOLERANCE_RATIO)
        if not (y_min <= transformed_y <= y_max):
            return None

        if transformed_x >= self.board_width:
            return ('hand_black',)
        if transformed_x < 0:
            return ('hand_white',)
        return None

    def draw_grid(self, image):
        """Visualizes the grid overlay."""
        if self.inv_homography_matrix is None:
            return image

        for grid_step in range(BOARD_GRID_SIZE + 1):
            x = grid_step * self.cell_width
            start_point = np.array([[[x, 0]]], dtype='float32')
            end_point = np.array([[[x, self.board_height]]], dtype='float32')

            transformed_start = cv2.perspectiveTransform(start_point, self.inv_homography_matrix)[0][0]
            transformed_end = cv2.perspectiveTransform(end_point, self.inv_homography_matrix)[0][0]

            cv2.line(image,
                     (int(transformed_start[0]), int(transformed_start[1])),
                     (int(transformed_end[0]), int(transformed_end[1])),
                     (0, 255, 255), 1)

        for grid_step in range(BOARD_GRID_SIZE + 1):
            y = grid_step * self.cell_height
            start_point = np.array([[[0, y]]], dtype='float32')
            end_point = np.array([[[self.board_width, y]]], dtype='float32')

            transformed_start = cv2.perspectiveTransform(start_point, self.inv_homography_matrix)[0][0]
            transformed_end = cv2.perspectiveTransform(end_point, self.inv_homography_matrix)[0][0]

            cv2.line(image,
                     (int(transformed_start[0]), int(transformed_start[1])),
                     (int(transformed_end[0]), int(transformed_end[1])),
                     (0, 255, 255), 1)

        return image

# 2. STATE STABILIZATION
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


# 3. SHOGI ENGINE INTEGRATION
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
        except Exception as error:
            print(f"Failed to start engine: {error}")
            return False

    def _send_command(self, command):
        """Send command to engine."""
        if self.process:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _wait_for(self, expected):
        """Wait for specific response from engine."""
        if self.process:
            while True:
                line = self.process.stdout.readline().strip()
                if expected in line:
                    return line

    def analyze_position(self, board_sfen, hand_sfen='-', time_ms=1000, turn='b'):
        """
        Analyze position and get top 3 moves for current player.
        board_sfen: just the board portion. hand_sfen: SFEN hand notation ('-' if empty).
        """
        self.current_turn = turn

        def _analyze():
            try:
                self.best_moves = []
                self.evaluations = []

                self._send_command(f"position sfen {board_sfen} {turn} {hand_sfen} 1")
                self._send_command(f"go movetime {time_ms}")

                ranked_moves = {}

                while True:
                    line = self.process.stdout.readline().strip()

                    if "bestmove" in line:
                        break

                    if "multipv" in line and "pv" in line:
                        try:
                            parts = line.split()
                            multipv_keyword_index = parts.index("multipv")
                            pv_keyword_index = parts.index("pv")

                            multipv_rank = int(parts[multipv_keyword_index + 1])
                            move = parts[pv_keyword_index + 1]

                            eval_score = None
                            if "score cp" in line:
                                centipawn_index = parts.index("cp")
                                eval_score = int(parts[centipawn_index + 1])
                                if turn == 'w':
                                    eval_score = -eval_score

                            ranked_moves[multipv_rank] = (move, eval_score)
                        except Exception:
                            pass

                for rank in range(1, 4):
                    if rank in ranked_moves:
                        move, eval_score = ranked_moves[rank]
                        self.best_moves.append(move)
                        self.evaluations.append(eval_score)

            except Exception as error:
                print(f"Engine analysis error: {error}")

        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()

    def get_best_moves(self):
        """Get the top 3 moves."""
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

# 4. HELPER: GRID TO SFEN & VALIDATION
def grid_to_sfen(grid_dict):
    """Converts a dict {(col, row): 'P'} into an SFEN string."""
    sfen_rows = []
    for row in range(BOARD_GRID_SIZE):
        row_str = ""
        empty_count = 0
        for col in range(BOARD_GRID_SIZE):
            piece = grid_dict.get((col, row))
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

def hand_to_sfen(hand_dict):
    """
    Convert hand piece counts to SFEN hand notation
    """
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
    return result if result else "-"


def _split_full_sfen(full_sfen):
    """Split a 'board hand' string into (board, hand)."""
    if not full_sfen:
        return None, "-"
    parts = full_sfen.split(" ", 1)
    board = parts[0]
    hand = parts[1] if len(parts) > 1 and parts[1] else "-"
    return board, hand


def is_valid_transition(prev_full_sfen, new_full_sfen):
    """
    Check if new_full_sfen is reachable from prev_full_sfen in one or two legal moves.
    """
    if not prev_full_sfen or not new_full_sfen:
        return True, "No previous state", None, None

    if prev_full_sfen == new_full_sfen:
        return True, "Same state", None, None

    prev_board, prev_hand = _split_full_sfen(prev_full_sfen)
    new_board, new_hand = _split_full_sfen(new_full_sfen)

    def _matches(candidate_board):
        parts = candidate_board.sfen().split()
        result_board = parts[0]
        result_hand = parts[2] if len(parts) > 2 else "-"
        return result_board == new_board and result_hand == new_hand

    # a move can change at most 4 squares (2 source + 2 destination).
    # if more squares changed than that, 2-ply is hopeless and skip the search
    def _expand_board(board_sfen):
        cells = []
        for row in board_sfen.split('/'):
            for character in row:
                if character.isdigit():
                    cells.extend(['.'] * int(character))
                else:
                    cells.append(character)
        return cells
    prev_cells = _expand_board(prev_board)
    new_cells = _expand_board(new_board)
    squares_diff = (
        sum(1 for prev, new in zip(prev_cells, new_cells) if prev != new)
        if len(prev_cells) == len(new_cells) else 99
    )

    try:
        # 1-ply search
        for turn in ['b', 'w']:
            board = shogi.Board(f"{prev_board} {turn} {prev_hand} 1")
            for move in list(board.legal_moves):
                test_board = shogi.Board(board.sfen())
                test_board.push(move)
                if _matches(test_board):
                    player = "Black" if turn == 'b' else "White"
                    next_turn = 'w' if turn == 'b' else 'b'
                    return True, f"Valid move: {move.usi()} ({player})", next_turn, move.usi()

        # 2-ply search
        if squares_diff <= 4:
            for turn in ['b', 'w']:
                board = shogi.Board(f"{prev_board} {turn} {prev_hand} 1")
                for first_move in list(board.legal_moves):
                    board_after_first = shogi.Board(board.sfen())
                    board_after_first.push(first_move)
                    for second_move in list(board_after_first.legal_moves):
                        board_after_second = shogi.Board(board_after_first.sfen())
                        board_after_second.push(second_move)
                        if _matches(board_after_second):
                            player = "Black" if turn == 'b' else "White"
                            # Two plies elapsed → it's the original mover's turn again
                            next_turn = turn
                            moves_usi = f"{first_move.usi()}+{second_move.usi()}"
                            return True, f"Valid 2-move: {moves_usi} (starting {player})", next_turn, moves_usi

        return False, f"Not reachable in 1-2 moves (diff={squares_diff})", None, None

    except Exception as error:
        return False, f"Validation error: {str(error)}", None, None

def detect_incomplete_board(grid_dict):
    """
    Check if detected board is missing critical pieces.
    Returns (is_complete, message)
    """
    has_black_king = 'K' in [piece for piece in grid_dict.values()]
    has_white_king = 'k' in [piece for piece in grid_dict.values()]

    if not has_black_king or not has_white_king:
        return False, "Missing king(s)"

    total_pieces = len(grid_dict)
    if total_pieces < MIN_BOARD_PIECES:
        return False, f"Only {total_pieces} pieces detected"

    return True, "Complete"

def draw_evaluation_bar(frame, evaluation, bar_x=20, bar_y=180, bar_width=40, bar_height=400):
    """Draw a vertical evaluation bar showing position advantage."""
    # Clamp evaluation to reasonable range
    eval_clamped = max(
        -EVAL_CLAMP_CENTIPAWNS,
        min(EVAL_CLAMP_CENTIPAWNS, evaluation if evaluation is not None else 0),
    )

    # Calculate bar fill percentage (0.0 = white advantage, 1.0 = black advantage)
    fill_ratio = (eval_clamped + EVAL_CLAMP_CENTIPAWNS) / (2 * EVAL_CLAMP_CENTIPAWNS)
    fill_height = int(bar_height * fill_ratio)

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    # White advantage (top)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + (bar_height - fill_height)),
                  (255, 255, 255), -1)

    # Black advantage (bottom)
    cv2.rectangle(frame, (bar_x, bar_y + (bar_height - fill_height)),
                  (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)

    # Center line
    center_y = bar_y + bar_height // 2
    cv2.line(frame, (bar_x, center_y), (bar_x + bar_width, center_y), (128, 128, 128), 2)

    # Labels
    cv2.putText(frame, "WHITE", (bar_x - 5, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "BLACK", (bar_x - 5, bar_y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Evaluation value
    if evaluation is not None:
        eval_text = f"{evaluation:+d}" if abs(evaluation) < 10000 else ("M+" if evaluation > 0 else "M-")
        text_x = bar_x + bar_width + 10
        text_y = bar_y + bar_height // 2
        cv2.putText(frame, eval_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_enhanced_arrow(frame, start, end, color, thickness, label="", eval_text=""):
    """Draw an enhanced arrow with glow effect and labels."""
    # Glow effect
    for glow_layer in range(3, 0, -1):
        alpha = 0.3 / glow_layer
        glow_color = tuple(int(channel * alpha + 20) for channel in color)
        cv2.arrowedLine(frame, start, end, glow_color, thickness + glow_layer * 2, tipLength=0.25)

    # Main arrow
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.25)

    # Move label
    if label:
        mid_x = int((start[0] + end[0]) / 2)
        mid_y = int((start[1] + end[1]) / 2)

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

        if eval_text:
            cv2.putText(frame, eval_text, (mid_x, mid_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_engine_suggestions(frame, mapper, best_moves, evaluations, turn):
    """Draw top 3 engine suggestions with enhanced arrows."""
    if not best_moves or mapper.inv_homography_matrix is None:
        return

    arrow_colors = [(0, 255, 0), (255, 180, 0), (255, 100, 255)]  # Green, Orange, Pink
    arrow_thicknesses = [4, 3, 2]

    for rank, (best_move, color, thickness) in enumerate(zip(best_moves[:3], arrow_colors, arrow_thicknesses)):
        try:
            if len(best_move) >= 4:
                from_col = BOARD_GRID_SIZE - int(best_move[0])
                from_row = ord(best_move[1]) - ord('a')
                to_col = BOARD_GRID_SIZE - int(best_move[2])
                to_row = ord(best_move[3]) - ord('a')

                from_x = (from_col + 0.5) * mapper.cell_width
                from_y = (from_row + 0.5) * mapper.cell_height
                to_x = (to_col + 0.5) * mapper.cell_width
                to_y = (to_row + 0.5) * mapper.cell_height

                from_point = cv2.perspectiveTransform(
                    np.array([[[from_x, from_y]]], dtype='float32'),
                    mapper.inv_homography_matrix,
                )[0][0]

                to_point = cv2.perspectiveTransform(
                    np.array([[[to_x, to_y]]], dtype='float32'),
                    mapper.inv_homography_matrix,
                )[0][0]

                label = f"#{rank + 1}"
                eval_text = (
                    f"({evaluations[rank]})"
                    if rank < len(evaluations) and evaluations[rank] is not None
                    else ""
                )

                draw_enhanced_arrow(
                    frame,
                    (int(from_point[0]), int(from_point[1])),
                    (int(to_point[0]), int(to_point[1])),
                    color, thickness, label, eval_text,
                )

        except Exception as error:
            print(f"Error drawing move {rank + 1}: {error}")

def draw_hand_panels(frame, black_hand, white_hand):
    """Draw small overlays listing each player's pieces in hand."""
    frame_height, frame_width = frame.shape[:2]

    def hand_text(hand):
        if not hand:
            return "(empty)"
        piece_order = ['R', 'B', 'G', 'S', 'N', 'L', 'P']
        parts = []
        for piece in piece_order:
            for key in (piece, piece.lower()):
                if hand.get(key, 0) > 0:
                    parts.append(f"{key}x{hand[key]}" if hand[key] > 1 else key)
        return " ".join(parts) if parts else "(empty)"

    # Black (sente) hand - right side
    black_text = "Black: " + hand_text(black_hand)
    (black_text_width, black_text_height), _ = cv2.getTextSize(black_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    black_box_x, black_box_y = frame_width - black_text_width - 20, frame_height - 30
    cv2.rectangle(frame,
                  (black_box_x - 8, black_box_y - black_text_height - 8),
                  (black_box_x + black_text_width + 8, black_box_y + 8),
                  (0, 0, 0), -1)
    cv2.rectangle(frame,
                  (black_box_x - 8, black_box_y - black_text_height - 8),
                  (black_box_x + black_text_width + 8, black_box_y + 8),
                  (180, 180, 180), 1)
    cv2.putText(frame, black_text, (black_box_x, black_box_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # White (gote) hand - left side
    white_text = "White: " + hand_text(white_hand)
    (white_text_width, white_text_height), _ = cv2.getTextSize(white_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    white_box_x, white_box_y = 20, frame_height - 30
    cv2.rectangle(frame,
                  (white_box_x - 8, white_box_y - white_text_height - 8),
                  (white_box_x + white_text_width + 8, white_box_y + 8),
                  (0, 0, 0), -1)
    cv2.rectangle(frame,
                  (white_box_x - 8, white_box_y - white_text_height - 8),
                  (white_box_x + white_text_width + 8, white_box_y + 8),
                  (180, 180, 180), 1)
    cv2.putText(frame, white_text, (white_box_x, white_box_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def draw_info_panel(frame, current_turn, best_moves, evaluations, stable_sfen, validation_msg, is_complete):
    """Draw information panel with game state."""
    panel_height = 120
    panel_color = (40, 40, 40)

    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), panel_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

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
    sfen_display = (
        stable_sfen[:50] + "..."
        if stable_sfen and len(stable_sfen) > 50
        else (stable_sfen or "Stabilizing...")
    )
    cv2.putText(frame, f"SFEN: {sfen_display}", (x_start, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

    # Top 3 moves list
    y_offset += 35
    if best_moves:
        moves_text = " | ".join([f"{rank + 1}. {move}" for rank, move in enumerate(best_moves[:3])])
        cv2.putText(frame, f"Top Moves: {moves_text}", (x_start, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 5. MAIN EXECUTION LOOP
def main():
    # SETUP
    video_path = 'vid/board_07.mp4'
    model_path = 'runs/detect/train11/weights/best.pt'
    engine_path = None

    video_capture = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    mapper = ShogiBoardMapper()
    stabilizer = StateStabilizer(buffer_len=6, threshold=0.7)
    engine = ShogiEngine(engine_path)

    use_engine = engine.start()

    output_file = open('sfen_predictions.txt', 'w')
    output_file.write("Frame | Raw SFEN | Stable SFEN | Valid Transition | Turn | Best Moves | Evaluations | Detections\n")
    output_file.write("=" * 140 + "\n")

    frame_count = 0
    recalibration_interval = DEFAULT_RECALIBRATION_INTERVAL
    use_dynamic_calibration = True
    last_stable_sfen = None
    last_validated_sfen = None
    current_turn = 'b'
    last_analyzed_state = None
    current_evaluation = 0
    last_analysis_frame = -10 ** 9
    analysis_debounce_frames = ANALYSIS_DEBOUNCE_FRAMES

    print("Starting processing... Press 'q' to quit.")
    print(f"Calibration mode: {'Dynamic (Multi-piece)' if use_dynamic_calibration else 'Static (4-lance)'}")
    print(f"Engine: {'Enabled' if use_engine else 'Disabled'}")

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1

        results = model.predict(frame, conf=0.35, iou=0.5, max_det=80, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = model.names

        # Noisy-frame skip
        noisy_frame = len(boxes) > NOISY_DETECTION_THRESHOLD
        if noisy_frame and mapper.homography_matrix is not None:
            mapper.draw_grid(frame)
            cv2.putText(frame, "Hand detected — pausing state update", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("Shogi Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1. CALIBRATION
        if mapper.homography_matrix is None:
            if use_dynamic_calibration:
                calibrated = mapper.calibrate_dynamic(boxes, classes, names)
                if calibrated:
                    print(f">>> DYNAMIC CALIBRATION SUCCESSFUL (Confidence: {mapper.calibration_confidence:.2f}) <<<")
                else:
                    cv2.putText(frame, "Calibrating... (Detecting starting position)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Shogi AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
            else:
                calibrated = mapper.calibrate(boxes, classes, names)
                if calibrated:
                    print(">>> CALIBRATION SUCCESSFUL <<<")
                else:
                    cv2.putText(frame, "Calibrating... (Need 4 Corner Lances)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Shogi AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
        current_hand = {}  # SFEN piece char -> count (uppercase=Black, lowercase=White)
        detections = []
        for box, class_index in zip(boxes, classes):
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            class_name = names[int(class_index)]
            piece_letter = class_name.split('_')[0]  # 'L','N','S','G','K','R','B','P'

            position = mapper.get_position(center_x, center_y)

            if position is None:
                detections.append(f"{class_name}@({int(center_x)},{int(center_y)})")
                continue

            kind = position[0]

            if kind == 'board':
                col, row = position[1], position[2]
                sfen_char = mapper.class_map.get(class_name, '?')
                current_grid[(col, row)] = sfen_char
                detections.append(f"{class_name}@({col},{row})")
                cv2.putText(frame, sfen_char, (int(center_x) - 10, int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif kind in ('hand_black', 'hand_white'):
                if piece_letter == 'K':
                    # Kings cannot be in hand 
                    detections.append(f"{class_name}@{kind}(ignored)")
                    continue
                # Side of board determines ownership, regardless of detected orientation
                hand_char = piece_letter if kind == 'hand_black' else piece_letter.lower()
                current_hand[hand_char] = current_hand.get(hand_char, 0) + 1
                detections.append(f"{class_name}@{kind}")
                cv2.putText(frame, hand_char, (int(center_x) - 10, int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        is_complete, completeness_msg = detect_incomplete_board(current_grid)

        # 4. GENERATE SFEN (board + hand)
        raw_board_sfen = grid_to_sfen(current_grid)
        raw_hand_sfen = hand_to_sfen(current_hand)
        raw_sfen = f"{raw_board_sfen} {raw_hand_sfen}"

        # 5. STABILIZE
        stable_sfen = stabilizer.update(raw_sfen, is_valid=is_complete)

        # 6. VALIDATE TRANSITION
        is_valid = False
        validation_msg = "N/A"
        detected_move = None

        if stable_sfen and stable_sfen != last_stable_sfen and is_complete:
            # Mark this stable SFEN as already attempted 
            last_stable_sfen = stable_sfen

            is_valid, validation_msg, next_turn, detected_move = is_valid_transition(
                last_validated_sfen, stable_sfen
            )

            if is_valid and next_turn:
                last_validated_sfen = stable_sfen
                print(f"✓ Valid transition: {validation_msg}")
                print(f"Turn changed: {current_turn} -> {next_turn}")
                current_turn = next_turn
            elif not last_validated_sfen:
                last_validated_sfen = stable_sfen
                is_valid = True
                validation_msg = "Initial state"
                print(f"Initial position set: {stable_sfen[:30]}...")
            else:
                print(f"✗ Invalid transition: {validation_msg}")

        # 7. ENGINE ANALYSIS
        best_moves_str = "N/A"
        evaluations_str = "N/A"

        should_analyze = (
            use_engine and
            stable_sfen and
            is_complete and
            is_valid and
            stable_sfen != last_analyzed_state and
            (frame_count - last_analysis_frame) >= analysis_debounce_frames
        )

        if should_analyze:
            stable_board, stable_hand = _split_full_sfen(stable_sfen)
            player = "Black" if current_turn == 'b' else "White"
            print(f"Analyzing for {player}'s turn: {stable_sfen[:60]}...")
            engine.analyze_position(stable_board, stable_hand, time_ms=500, turn=current_turn)
            last_analyzed_state = stable_sfen
            last_analysis_frame = frame_count

        best_moves = []
        evaluations = []

        if use_engine:
            best_moves = engine.get_best_moves()
            evaluations = engine.get_evaluations()

            if best_moves:
                best_moves_str = ", ".join(best_moves)
                evaluations_str = ", ".join(
                    [str(eval_score) if eval_score is not None else "N/A" for eval_score in evaluations]
                )

                if evaluations and evaluations[0] is not None:
                    current_evaluation = evaluations[0]

                draw_engine_suggestions(frame, mapper, best_moves, evaluations, current_turn)

        # Write to file
        detections_str = ", ".join(detections) if detections else "None"
        stable_sfen_str = stable_sfen if stable_sfen else "Stabilizing..."
        turn_str = "Black" if current_turn == 'b' else "White"
        output_file.write(
            f"{frame_count} | {raw_sfen} | {stable_sfen_str} | {validation_msg} | "
            f"{turn_str} | {best_moves_str} | {evaluations_str} | {detections_str}\n"
        )

        # Draw UI Elements
        draw_evaluation_bar(frame, current_evaluation)
        draw_info_panel(frame, current_turn, best_moves, evaluations, stable_sfen, validation_msg, is_complete)
        # Split current_hand into Black/White subdicts for display
        black_hand_view = {key: count for key, count in current_hand.items() if key.isupper()}
        white_hand_view = {key: count for key, count in current_hand.items() if key.islower()}
        draw_hand_panels(frame, black_hand_view, white_hand_view)

        cv2.imshow("Shogi Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    output_file.close()
    video_capture.release()
    cv2.destroyAllWindows()
    if use_engine:
        engine.stop()

    print(f"\n>>> Output saved to sfen_predictions.txt <<<")

if __name__ == "__main__":
    main()
