import os
import shutil
from sklearn.model_selection import train_test_split

image_dir = 'data/shogi_detection/images'
label_dir = 'data/shogi_detection/labels'
train_dir = 'data/shogi_detection/images/train'
val_dir = 'data/shogi_detection/images/val'
train_label_dir = 'data/shogi_detection/labels/train'
val_label_dir = 'data/shogi_detection/labels/val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, img))
    label = img.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(label_dir, label)):
        shutil.move(os.path.join(label_dir, label), os.path.join(train_label_dir, label))

for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, img))
    label = img.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(label_dir, label)):
        shutil.move(os.path.join(label_dir, label), os.path.join(val_label_dir, label))