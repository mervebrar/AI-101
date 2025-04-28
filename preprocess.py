import os
import cv2
import numpy as np
from tqdm import tqdm

source_folder = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata"
destination_folder = '/Users/merveebrardemirel/Desktop/practice_datasets/processed_catdogdata'

img_size = (128, 128)

subfolders = ['train/cats', 'train/dogs', 'test/cats', 'test/dogs',"val/cats", "val/dogs"]

for subfolder in subfolders:
    source_path = os.path.join(source_folder, subfolder)
    dest_path = os.path.join(destination_folder, subfolder)
    os.makedirs(dest_path, exist_ok=True)

    print(f"Processing {subfolder}...")
    for img_name in tqdm(os.listdir(source_path)):
        img_path = os.path.join(source_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, img_size)

        img = cv2.GaussianBlur(img, (3,3), 0)


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        img = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)

        img = np.clip(img, 0, 255).astype(np.uint8)

        save_path = os.path.join(dest_path, img_name)
        cv2.imwrite(save_path, img)

print("Images are processed successfully!")