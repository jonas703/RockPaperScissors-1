import os
import shutil
import random
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
SOURCE_ROOT = None # Path to your annotated dataset root folder
OUT_ROOT    = None # Path to output the YOLO formatted dataset
CLASSES     = ["paper", "rock", "scissors"]
TRAIN_SPLIT = 0.8
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp")
# ------------------------------------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    
    # Create output directories - YOLO format
    img_train = os.path.join(OUT_ROOT, "images/train")
    img_val   = os.path.join(OUT_ROOT, "images/val")
    lbl_train = os.path.join(OUT_ROOT, "labels/train")
    lbl_val   = os.path.join(OUT_ROOT, "labels/val")

    # Ensure directories exist before copying files
    for d in (img_train, img_val, lbl_train, lbl_val):
        ensure_dir(d)

    samples = []

    for cls in CLASSES:
        folder = os.path.join(SOURCE_ROOT, cls)

        # Gather all labeled images
        for fname in os.listdir(folder):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            
            img_path = os.path.join(folder, fname)
            label_path = os.path.join(folder, Path(fname).stem + ".txt")

            if not os.path.exists(label_path):
                print("[WARN] Missing label:", img_path)
                continue

            samples.append((img_path, label_path, cls))

    print(f"Found {len(samples)} labeled images.")


    # Shuffle and split dataset
    random.shuffle(samples)
    split = int(len(samples)*TRAIN_SPLIT)

    train_samples = samples[:split]
    val_samples   = samples[split:]

    # Copy files to respective folders
    def move_set(dataset, img_dst, lbl_dst):
        for img_path, lbl_path, clsname in dataset:
            stem = Path(img_path).stem

            new_img = f"{clsname}_{stem}{Path(img_path).suffix}"
            new_lbl = f"{clsname}_{stem}.txt"

            shutil.copy2(img_path, os.path.join(img_dst, new_img))
            shutil.copy2(lbl_path, os.path.join(lbl_dst, new_lbl))
    
    move_set(train_samples, img_train, lbl_train)
    move_set(val_samples,   img_val,   lbl_val)

    # Create YAML file
    yaml_path = os.path.join(OUT_ROOT, "rps.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for idx, name in enumerate(CLASSES):
            f.write(f"  {idx}: {name}\n")

    print("\nDataset created successfully!")
    print("Train images:", len(train_samples))
    print("Val images:", len(val_samples))
    print("\nYAML saved to:", yaml_path)


if __name__ == "__main__":
    main()
