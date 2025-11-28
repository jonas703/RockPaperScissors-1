from ultralytics import YOLO
import cv2
import os
import numpy as np



ROOT = None # Path to your dataset root folder


model = YOLO("yolo11x.pt")  # pretrained model that can detect hands

def process_folder(folder):
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue

        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)

        results = model(img, conf=0.2)

        # no detection → skip
        if len(results[0].boxes) == 0:
            print("No hand found:", fname)
            continue

        # choose the largest detected hand
        boxes = results[0].boxes.xywhn.cpu().numpy()   # [cx,cy,w,h] normalized
        areas = boxes[:,2] * boxes[:,3]
        best = boxes[np.argmax(areas)]

        # build YOLO label path
        txt_path = os.path.join(folder, fname.rsplit(".",1)[0] + ".txt")

        # write class id + bounding box
        # class mapping: 0=paper,1=rock,2=scissors
        class_id = CLASS_MAP[os.path.basename(folder).lower()]

        with open(txt_path, "w") as f:
            f.write(f"{class_id} {best[0]:.6f} {best[1]:.6f} {best[2]:.6f} {best[3]:.6f}\n")

        print("Labeled:", fname)



# ensure class map follows folder names and order
CLASS_MAP = {"paper":0, "rock":1, "scissors":2}

#Go through each class folder and process images
for cls in CLASS_MAP:
    folder = os.path.join(ROOT, cls)
    process_folder(folder)
print("All done.")

