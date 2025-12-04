from ultralytics import YOLO



def main():
    data_path = None # Path to your dataset YAML file
    #using a pretrained YOLOv11s model
    model = YOLO("yolo11s.pt")

    # Start training on the dataset
    results = model.train(
        data=data_path,
        epochs=80,
        imgsz=640,
        batch=16,
        workers=2,
        pretrained=True,
        cos_lr=True,
        patience=50,

        # augmentation parameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30,
        translate=0.20,
        scale=0.50,
        shear=10,
        perspective=0.0005,
        flipud=0.05,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,

        project="runs/detect",
        name="rps_yolo11",
        exist_ok=True,
    )

    #Use the best model for RPS inference
    print("\nTraining finished!")
    print("Best weights at:", results.save_dir, "/weights/best.pt")

if __name__ == "__main__":
    main()

