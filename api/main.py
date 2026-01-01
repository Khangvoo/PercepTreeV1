
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict
import io

# --- DETECTRON2 IMPORTS ---
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ================= CẤU HÌNH =================
HUMAN_HEIGHT_CM = 170.0 
# Paths are relative to the project root (E:\mathstem)
PERCEPTREE_MODEL_WEIGHTS = "output/X-101_RGB_60k.pth" 
YOLO_PERSON_MODEL_PATH = 'yolov8n.pt'
YOLO_WORLD_MODEL_PATH = 'inference_pretrained/yolov8s-worldv2.pt'
# ============================================

# Dictionary to hold the loaded models
models: Dict[str, any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("⏳ Đang khởi tạo và tải các mô hình AI...")
    
    # 1. LOAD YOLO (Để tìm người & tính tỷ lệ)
    models["yolo_person"] = YOLO(YOLO_PERSON_MODEL_PATH)
    print("✅ Đã tải Model YOLO phát hiện người.")

    # 2. LOAD YOLO WORLD (Để tìm toàn bộ cây)
    yolo_world = YOLO(YOLO_WORLD_MODEL_PATH)
    yolo_world.set_classes(["tree"])
    models["yolo_tree"] = yolo_world
    print("✅ Đã tải Model YOLO World phát hiện cây.")

    # 3. LOAD PERCEPTREE (DETECTRON2)
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = PERCEPTREE_MODEL_WEIGHTS
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Use CPU for broader compatibility. Change to "cuda" if a GPU is available.
        cfg.MODEL.DEVICE = "cuda" 
        models["perceptree"] = DefaultPredictor(cfg)
        print("✅ Đã tải Model PercepTree (Detectron2).")
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi tải Model PercepTree (Detectron2): {e}")
        print("👉 Hãy chắc chắn bạn đã cài đặt detectron2 đúng cách và đường dẫn tới file .pth là chính xác.")
        # In a real app, you might want to handle this more gracefully
        raise RuntimeError(f"Could not load PercepTree model: {e}") from e
        
    print("✅✅✅ Hệ thống đã sẵn sàng! ✅✅✅")
    yield
    # Clean up models on shutdown
    models.clear()
    print("✅ Đã dọn dẹp mô hình.")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
# This allows the frontend (running on a different port) to communicate with this API.
# In production, you should restrict the origins to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def analyze_image(img_bytes: bytes, real_human_height: float, override_person_box: list = None, override_tree_box: list = None) -> dict:
    # Decode the image from bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh. Định dạng có thể không hợp lệ.")
    h_img, w_img, _ = img.shape

    # --- Initialize results ---
    results = {
        "tree_height_m": 0.0,
        "dbh_cm": 0.0,
        "boxes": {
            "person": None,
            "tree": None
        }
    }

    # --- BƯỚC 1: XÁC ĐỊNH BOX CỦA NGƯỜI ---
    scale_person_box = None
    if override_person_box is not None and len(override_person_box) == 4:
        print("🔧 Sử dụng box người từ người dùng.")
        scale_person_box = np.array(override_person_box, dtype=int)
    else:
        # Chạy model tìm người
        person_results = models["yolo_person"](img, classes=[0], verbose=False)
        all_people_boxes = [box.xyxy[0].cpu().numpy().astype(int) for r in person_results for box in r.boxes]
        
        best_human_h = 0
        if not all_people_boxes:
            print("⚠️ Không thấy người nào! Dùng giả định chiều cao.")
            # Default fallback box
            best_human_h = h_img / 3
            scale_person_box = np.array([10, h_img - int(best_human_h), 100, h_img], dtype=int)
        else:
            for p_box in all_people_boxes:
                h = p_box[3] - p_box[1]
                if h > best_human_h:
                    best_human_h = h
                    scale_person_box = p_box

    # --- BƯỚC 2: XÁC ĐỊNH BOX CỦA CÂY ---
    best_full_tree_box = None
    if override_tree_box is not None and len(override_tree_box) == 4:
        print("🔧 Sử dụng box cây từ người dùng.")
        best_full_tree_box = np.array(override_tree_box, dtype=int)
    else:
        # Chạy model tìm cây
        full_tree_results = models["yolo_tree"].predict(img, verbose=False, conf=0.1)
        best_tree_confidence = -1.0
        for res in full_tree_results:
            for box in res.boxes:
                if box.conf[0] > best_tree_confidence:
                    best_tree_confidence = box.conf[0]
                    best_full_tree_box = box.xyxy[0].cpu().numpy().astype(int)

    # Save found/used boxes to results
    if scale_person_box is not None:
        results["boxes"]["person"] = scale_person_box.tolist()
    if best_full_tree_box is not None:
        results["boxes"]["tree"] = best_full_tree_box.tolist()

    # --- Kiểm tra dữ liệu trước khi tính toán ---
    if best_full_tree_box is None:
        print("❌ Không xác định được cây.")
        return results

    person_height_px = scale_person_box[3] - scale_person_box[1]
    if person_height_px == 0:
         return results

    # --- BƯỚC 3: TÍNH TOÁN CHIỀU CAO ---
    ground_y = scale_person_box[3]
    scale_cm_per_px = real_human_height / person_height_px
    results["tree_height_m"] = ((best_full_tree_box[3] - best_full_tree_box[1]) * scale_cm_per_px) / 100.0

    # --- BƯỚC 4: TÌM THÂN CÂY VÀ TÍNH DBH ---
    pixel_1m3 = 130.0 / scale_cm_per_px
    dbh_y_global = int(ground_y - pixel_1m3)
    
    slice_height = int((best_full_tree_box[3] - best_full_tree_box[1]) * 0.4)
    slice_y1 = max(0, dbh_y_global - slice_height // 2)
    slice_y2 = min(h_img, dbh_y_global + slice_height // 2)
    slice_x1 = max(0, best_full_tree_box[0])
    slice_x2 = min(w_img, best_full_tree_box[2])
    
    cropped_slice_img = img[slice_y1:slice_y2, slice_x1:slice_x2]

    main_trunk_instance = None
    if cropped_slice_img.size > 0:
        tree_trunk_outputs = models["perceptree"](cropped_slice_img)
        tree_instances = tree_trunk_outputs["instances"].to("cpu")
        if len(tree_instances) > 0:
            main_trunk_idx = tree_instances.pred_masks.sum(axis=(1, 2)).argmax()
            main_trunk_instance = tree_instances[main_trunk_idx:main_trunk_idx+1]

    if main_trunk_instance is not None:
        # Create a full-size mask
        main_trunk_mask_cropped = main_trunk_instance.pred_masks[0].numpy()
        
        # Calculate DBH
        dbh_y_on_slice = dbh_y_global - slice_y1
        # Ensure we are within the cropped slice bounds
        if 0 <= dbh_y_on_slice < cropped_slice_img.shape[0]:
            y_slice_start = max(0, dbh_y_on_slice - 2)
            y_slice_end = min(cropped_slice_img.shape[0], dbh_y_on_slice + 3)
            
            mask_slice_for_dbh = main_trunk_mask_cropped[y_slice_start:y_slice_end, :]
            widths = np.sum(mask_slice_for_dbh, axis=1)
            avg_width_px = np.mean(widths) if len(widths) > 0 else 0
            
            if avg_width_px > 0:
                results["dbh_cm"] = avg_width_px * scale_cm_per_px

    return results

@app.get("/")
async def root():
    return {"message": "Welcome to the PercepTree API. Send a POST request to /predict to analyze an image."}

from fastapi import Form
import json

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    person_box: str = Form(None), # Receives JSON string "[x1, y1, x2, y2]"
    tree_box: str = Form(None)    # Receives JSON string "[x1, y1, x2, y2]"
):
    """
    Analyzes an uploaded image. Can accept optional bounding boxes to override detection.
    """
    try:
        img_bytes = await file.read()
        
        p_box = json.loads(person_box) if person_box else None
        t_box = json.loads(tree_box) if tree_box else None

        analysis_results = analyze_image(
            img_bytes, 
            real_human_height=HUMAN_HEIGHT_CM,
            override_person_box=p_box,
            override_tree_box=t_box
        )
        return analysis_results
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi trong quá trình phân tích: {str(e)}")

# To run this API:
# 1. Make sure you are in the root directory (E:\mathstem)
# 2. Run the command: uvicorn api.main:app --reload
