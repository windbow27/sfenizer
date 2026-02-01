import os
import shutil

labels_dir = 'data/shogi_detection/labels'
images_dir = 'data/shogi_detection/images'

image_names = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}

for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        parts = label_file.split('-', 1)
        if len(parts) > 1:
            new_name = parts[1] 
            if new_name.rsplit('.', 1)[0] in image_names: # Check if corresponding image exists
                old_path = os.path.join(labels_dir, label_file)
                new_path = os.path.join(labels_dir, new_name)
                shutil.move(old_path, new_path)
                print(f'Renamed: {label_file} -> {new_name}')
            else:
                print(f'Skipped (no matching image): {label_file}')