"""
train_yolo.py
====================
Huấn luyện YOLOv11s để phát hiện vùng biển số xe trong ảnh.
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    print(" BẮT ĐẦU HUẤN LUYỆN MODEL PHÁT HIỆN BIỂN SỐ...")

    # model.train(
    #     data="./train_data/detect_data/data.yaml",
    #     epochs=150,
    #     imgsz=640,
    #     batch=4,
    #     device=0,
    #     workers=2,
    #     cache=False,
    #     name="bien_so_detector",
    #     exist_ok=False,

    #     # ── Optimizer ──────────────────────────────────────────
    #     optimizer="AdamW",
    #     lr0=0.0005,
    #     lrf=0.01,
    #     warmup_epochs=3,
    #     cos_lr=True, 
    #     weight_decay=0.0005,

    #     # ── Augmentation ───────────────────────────────────────
    #     # Màu sắc & sáng (biển số ngoài trời, nhiều điều kiện)
    #     hsv_h=0.015,
    #     hsv_s=0.5,
    #     hsv_v=0.4,

    #     # Góc nghiêng (xe đỗ không thẳng hoàn toàn)
    #     degrees=10,
    #     shear=2.0,

    #     # Scale (camera xa/gần khác nhau)
    #     scale=0.5,
    #     mosaic=1.0,           # ghép 4 ảnh, giúp detect biển số nhỏ
    #     fliplr=0.0,
    #     flipud=0.0,
    # )

    model.train(
        data="./train_data/detect_data/data.yaml",
        epochs=200,
        imgsz=640,
        batch=4,  
        device=0,
        workers=4,
        cache=False,
        name="bien_so_detector",
        exist_ok=False,

        # ── Optimizer & LR ─────────────────────────────────────
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,                    
        warmup_epochs=5, 
        cos_lr=True,                 
        weight_decay=0.0005,

        # ── Augmentation ───────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,            

        degrees=10,
        shear=2.0,
        scale=0.3,
        
        mosaic=1.0,           
        close_mosaic=20, 

        fliplr=0.0,
        flipud=0.0,
    )
    print(" HUẤN LUYỆN HOÀN TẤT!")
    print(" Model lưu tại: runs/detect/bien_so_detector/weights/best.pt")


if __name__ == "__main__":
    main()