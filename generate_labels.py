import cv2
import numpy as np
import os
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
import random

def detect_board(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{image_path}'.")
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    _, thresh2 = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)

    def is_roughly_rectangular(pts):
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        ratio1 = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
        ratio2 = abs(sides[1] - sides[3]) / max(sides[1], sides[3])
        return ratio1 < 0.3 and ratio2 < 0.3

    def find_board_contour(thresh_img, min_area_ratio=0.05):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = thresh_img.shape[0] * thresh_img.shape[1]
        min_area = img_area * min_area_ratio
        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                if len(approx) == 4 and is_roughly_rectangular(approx.reshape(4, 2)):
                    return approx.reshape(4, 2), cnt
        return None, None

    for name, thresh in [("adaptive", thresh1), ("otsu", thresh2), ("cleaned", thresh_cleaned)]:
        approx, board_cnt = find_board_contour(thresh)
        if approx is not None:
            break
    else:
        raise ValueError("Could not detect a valid board contour.")

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]    # top-left
        rect[2] = pts[np.argmax(s)]    # bottom-right
        rect[1] = pts[np.argmin(diff)] # top-right
        rect[3] = pts[np.argmax(diff)] # bottom-left
        return rect

    rect = order_points(approx)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warped, M, (maxWidth, maxHeight)

def extract_cells(image_path, out_dir="shogi_cells"):
    warped_board, _, _ = detect_board(image_path)
    cell_w = warped_board.shape[1] // 9
    cell_h = warped_board.shape[0] // 9
    os.makedirs(out_dir, exist_ok=True)
    for row in range(9):
        for col in range(9):
            x = col * cell_w
            y = row * cell_h
            cell = warped_board[y:y+cell_h, x:x+cell_w]
            cell_filename = f"{row}_{col}.jpg"
            cv2.imwrite(os.path.join(out_dir, cell_filename), cell)
    print(f"[Success] 9x9 cells saved to '{out_dir}'. Ready for labeling.")

def label_cells(cell_dir="shogi_cells", labeled_dir="labeled_cells"):
    os.makedirs(labeled_dir, exist_ok=True)
    cell_files = sorted([f for f in os.listdir(cell_dir) if f.endswith(".jpg")])
    index = 0
    img_output = widgets.Output()
    label_input = widgets.Text(description="Label:", placeholder="e.g. P_black, empty")
    save_button = widgets.Button(description="Save & Next", button_style='success')

    def show_image(idx):
        img_output.clear_output(wait=True)
        img_path = os.path.join(cell_dir, cell_files[idx])
        with img_output:
            img = Image.open(img_path)
            display(img)
        label_input.value = ""

    def save_and_next(b):
        nonlocal index
        label = label_input.value.strip()
        if not label:
            label = "empty"
        label_path = os.path.join(labeled_dir, label)
        os.makedirs(label_path, exist_ok=True)
        base_name = os.path.splitext(cell_files[index])[0]
        ext = os.path.splitext(cell_files[index])[1]
        while True:
            rand_num = random.randint(1000, 9999)
            new_name = f"{base_name}_{rand_num}{ext}"
            if not os.path.exists(os.path.join(label_path, new_name)):
                break
        src = os.path.join(cell_dir, cell_files[index])
        dst = os.path.join(label_path, new_name)
        os.rename(src, dst)
        index += 1
        if index < len(cell_files):
            show_image(index)
        else:
            print("[Success] Labeling complete!")

    save_button.on_click(save_and_next)
    show_image(index)
    display(img_output, label_input, save_button)