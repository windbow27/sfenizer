import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# --- Mappings ---
SFEN_MAP = {
    "K_black": "K", "R_black": "R", "B_black": "B", "G_black": "G", "S_black": "S", "N_black": "N", "L_black": "L", "P_black": "P",
    "+R_black": "+R", "+B_black": "+B", "+S_black": "+S", "+N_black": "+N", "+L_black": "+L", "+P_black": "+P",
    "K_white": "k", "R_white": "r", "B_white": "b", "G_white": "g", "S_white": "s", "N_white": "n", "L_white": "l", "P_white": "p",
    "+R_white": "+r", "+B_white": "+b", "+S_white": "+s", "+N_white": "+n", "+L_white": "+l", "+P_white": "+p",
    "empty": ""
}
CSA_MAP = {
    "K_black": "+OU", "R_black": "+HI", "B_black": "+KA", "G_black": "+KI", "S_black": "+GI", "N_black": "+KE", "L_black": "+KY", "P_black": "+FU",
    "+R_black": "+RY", "+B_black": "+UM", "+S_black": "+NG", "+N_black": "+NK", "+L_black": "+NY", "+P_black": "+TO",
    "K_white": "-OU", "R_white": "-HI", "B_white": "-KA", "G_white": "-KI", "S_white": "-GI", "N_white": "-KE", "L_white": "-KY", "P_white": "-FU",
    "+R_white": "-RY", "+B_white": "-UM", "+S_white": "-NG", "+N_white": "-NK", "+L_white": "-NY", "+P_white": "-TO",
    "empty": "*"
}

def to_sfen(board, turn="b", hands="-", move_number=1):
    sfen_rows = []
    for row in board:
        row_str = ""
        empty_count = 0
        for cell in row:
            piece = SFEN_MAP.get(cell, "")
            if piece == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece
        if empty_count > 0:
            row_str += str(empty_count)
        sfen_rows.append(row_str)
    sfen = "/".join(sfen_rows)
    return f"{sfen} {turn} {hands} {move_number}"

def to_csa(board):
    csa_lines = []
    for i, row in enumerate(board):
        line = f"P{i+1} "
        for cell in row:
            piece = CSA_MAP.get(cell, "*")
            if piece == "*":
                line += "* "
            else:
                line += piece
        line = line.rstrip()
        csa_lines.append(line)
    csa_lines.append("+")
    return "\n".join(csa_lines)

# --- Model and transform setup ---
classes = ['+B_black', '+B_white', '+L_black', '+L_white', '+N_black', 
           '+N_white', '+P_black', '+P_white', '+R_black', '+R_white', 
           '+S_black', '+S_white', 'B_black', 'B_white', 'G_black', 
           'G_white', 'K_black', 'K_white', 'L_black', 'L_white', 
           'N_black', 'N_white', 'P_black', 'P_white', 'R_black', 
           'R_white', 'S_black', 'S_white', 'empty']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("sfennizer.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_cell(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return classes[pred_idx]

def predict_board(warped_board):
    cell_w = warped_board.shape[1] // 9
    cell_h = warped_board.shape[0] // 9
    board_predictions = []
    for row in range(9):
        row_preds = []
        for col in range(9):
            x = col * cell_w
            y = row * cell_h
            cell = warped_board[y:y+cell_h, x:x+cell_w]
            pil_img = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            pred_label = predict_cell(pil_img)
            row_preds.append(pred_label)
        board_predictions.append(row_preds)
    return board_predictions