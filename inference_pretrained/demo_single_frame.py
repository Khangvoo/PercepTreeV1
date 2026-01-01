import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- DETECTRON2 IMPORTS ---
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes


# ================= CẤU HÌNH =================
HUMAN_HEIGHT_CM = 170.0 
MODEL_WEIGHTS = "output/X-101_RGB_60k.pth" 
# ============================================

print("⏳ Đang khởi tạo hệ thống...")

# 1. LOAD YOLO (Để tìm người & tính tỷ lệ)
yolo_model = YOLO('yolov8n.pt')

# 2. LOAD YOLO WORLD (Để tìm toàn bộ cây)
print("🌳 Đang tải World Model...")
full_tree_model = YOLO('inference_pretrained/yolov8s-worldv2.pt')
full_tree_model.set_classes(["tree"])
print("✅ Đã tải World Model thành công!")

# 3. LOAD PERCEPTREE (DETECTRON2)
def setup_perceptree():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)

try:
    tree_predictor = setup_perceptree()
    print("✅ Đã load Model PercepTree (X-101) thành công!")
except Exception as e:
    print(f"❌ Lỗi Detectron2: {e}")
    print("👉 Hãy chắc chắn bạn đã cài detectron2 và đường dẫn file .pth đúng.")
    exit()

def run_analysis(image_path, real_human_height, output_path):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"❌ Không thể đọc ảnh từ: {image_path}")
        return
    h_img, w_img, _ = img.shape

    # --- BƯỚC 1: TÌM NGƯỜI VÀ CÂY TỔNG THỂ ---
    print("👉 Bước 1: Đang phát hiện người và cây tổng thể...")
    person_results = yolo_model(img, classes=[0], verbose=False)
    full_tree_results = full_tree_model.predict(img, verbose=False, conf=0.1)

    all_people_boxes = [box.xyxy[0].cpu().numpy().astype(int) for r in person_results for box in r.boxes]

    # --- BƯỚC 2: CHỌN CÂY TỐT NHẤT VÀ TÍNH TOÁN TỶ LỆ ---
    print("👉 Bước 2: Đang chọn cây tốt nhất và tính toán tỷ lệ...")
    best_tree_confidence = -1.0
    best_full_tree_box = None
    for res in full_tree_results:
        for box in res.boxes:
            if box.conf[0] > best_tree_confidence:
                best_tree_confidence = box.conf[0]
                best_full_tree_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_full_tree_box is None:
        print("❌ World Model không tìm thấy cây nào. Dừng xử lý.")
        cv2.imwrite(output_path, img)
        return
    print(f"✅ Đã chọn cây tốt nhất với độ tin cậy {best_tree_confidence:.2f}")

    best_human_h = 0
    scale_person_box = None
    if not all_people_boxes:
        print("⚠️ Không thấy người nào! Dùng giả định.")
        best_human_h = h_img / 3
        scale_person_box = [10, h_img - int(best_human_h), 100, h_img]
    else:
        for p_box in all_people_boxes:
            h = p_box[3] - p_box[1]
            if h > best_human_h:
                best_human_h = h
                scale_person_box = p_box
    
    ground_y = scale_person_box[3]
    scale_cm_per_px = real_human_height / best_human_h
    print(f"✅ Tỷ lệ ảnh: 1 pixel = {scale_cm_per_px:.2f} cm")

    # --- BƯỚC 3: TÌM THÂN CÂY TRONG LÁT CẮT NGANG NGỰC ---
    print("👉 Bước 3: Đang tìm thân cây trong lát cắt ngang ngực...")
    
    # Tính toán vị trí và kích thước của lát cắt
    pixel_1m3 = 130.0 / scale_cm_per_px
    dbh_y_global = int(ground_y - pixel_1m3)
    
    slice_height = int((best_full_tree_box[3] - best_full_tree_box[1]) * 0.4) # Lấy 40% chiều cao cây
    slice_y1 = max(0, dbh_y_global - slice_height // 2)
    slice_y2 = min(h_img, dbh_y_global + slice_height // 2)
    slice_x1 = best_full_tree_box[0]
    slice_x2 = best_full_tree_box[2]
    
    # Cắt ảnh theo lát cắt
    cropped_slice_img = img[slice_y1:slice_y2, slice_x1:slice_x2]

    main_trunk_instance = None
    if cropped_slice_img.size > 0:
        tree_trunk_outputs = tree_predictor(cropped_slice_img)
        tree_instances = tree_trunk_outputs["instances"].to("cpu")
        if len(tree_instances) > 0:
            main_trunk_idx = tree_instances.pred_masks.sum(axis=(1, 2)).argmax()
            main_trunk_instance = tree_instances[main_trunk_idx:main_trunk_idx+1]
            print("✅ Đã tìm thấy thân cây trong lát cắt.")
        else:
            print("❌ PercepTree không tìm thấy thân cây trong lát cắt.")
    else:
        print("❌ Lát cắt bị lỗi (kích thước 0).")

    # --- BƯỚC 4: LỌC NGƯỜI VÀ TÍNH TOÁN CÁC THÔNG SỐ ---
    print("👉 Bước 4: Đang lọc người và tính toán...")
    filtered_people_boxes = []
    if main_trunk_instance is not None:
        # Tạo mask toàn cục của thân cây để lọc
        main_trunk_mask_cropped = main_trunk_instance.pred_masks[0].numpy()
        full_trunk_mask = np.zeros((h_img, w_img), dtype=bool)
        full_trunk_mask[slice_y1:slice_y2, slice_x1:slice_x2][main_trunk_mask_cropped] = True
        
        for p_box in all_people_boxes:
            px1, py1, px2, py2 = max(0, p_box[0]), max(0, p_box[1]), min(w_img, p_box[2]), min(h_img, p_box[3])
            person_area = (px2 - px1) * (py2 - py1)
            if person_area == 0: continue
            
            overlap_area = np.sum(full_trunk_mask[py1:py2, px1:px2])
            if (overlap_area / person_area) < 0.3:
                filtered_people_boxes.append(p_box)
            else:
                print(f"   -> Đã loại bỏ 'người' giả tại box {p_box}")
    else:
        filtered_people_boxes = all_people_boxes

    # Tính chiều cao cây tổng thể (dựa vào box tím)
    tree_height_pixels = best_full_tree_box[3] - best_full_tree_box[1]
    tree_height_m = (tree_height_pixels * scale_cm_per_px) / 100.0
    print(f"🌳 CHIỀU CAO CÂY (ước tính): {tree_height_m:.2f} m")

    # --- BƯỚC 5: VẼ KẾT QUẢ ---
    print("👉 Bước 5: Đang vẽ kết quả...")
    vis_img = img.copy()

    # Vẽ box cho cây tổng thể
    cv2.rectangle(vis_img, (best_full_tree_box[0], best_full_tree_box[1]), (best_full_tree_box[2], best_full_tree_box[3]), (128, 0, 128), 3)
    cv2.putText(vis_img, f"Selected Tree {best_tree_confidence:.2f}", (best_full_tree_box[0], best_full_tree_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
    
    # Vẽ người
    for p_box in filtered_people_boxes:
        cv2.rectangle(vis_img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
    
    cv2.line(vis_img, (0, ground_y), (w_img, ground_y), (255, 255, 0), 2)

    # Chỉ vẽ và tính toán DBH nếu tìm thấy thân cây
    if main_trunk_instance is not None:
        main_trunk_box_cropped = main_trunk_instance.pred_boxes.tensor[0].numpy()
        main_trunk_box_global = main_trunk_box_cropped + np.array([slice_x1, slice_y1, slice_x1, slice_y1])

        # Vẽ mask thân cây
        my_tree_metadata = MetadataCatalog.get("my_tree_dataset_v4").set(thing_classes=["Tree Trunk"])
        visualizer = Visualizer(vis_img[:, :, ::-1], metadata=my_tree_metadata, scale=1.0)
        remapped_instances = Instances((h_img, w_img))
        remapped_instances.pred_boxes = Boxes(main_trunk_box_global.reshape(1, 4))
        remapped_instances.pred_masks = full_trunk_mask.reshape(1, h_img, w_img)
        remapped_instances.scores = main_trunk_instance.scores
        out = visualizer.draw_instance_predictions(remapped_instances)
        vis_img = out.get_image()[:, :, ::-1].copy()

        # Tính và vẽ DBH
        dbh_y_on_slice = dbh_y_global - slice_y1
        y_slice_start, y_slice_end = max(0, dbh_y_on_slice - 2), min(cropped_slice_img.shape[0], dbh_y_on_slice + 3)
        mask_slice = main_trunk_instance.pred_masks[0].numpy()[y_slice_start:y_slice_end, :]
        widths = np.sum(mask_slice, axis=1)
        avg_width_px = np.mean(widths) if len(widths) > 0 else 0
        dbh_cm = avg_width_px * scale_cm_per_px
        print(f"🎯 ĐƯỜNG KÍNH (DBH): {dbh_cm:.2f} cm")
        
        row_pixels = full_trunk_mask[dbh_y_global, :]
        indices = np.where(row_pixels)[0]
        if len(indices) > 0:
            cv2.line(vis_img, (indices[0], dbh_y_global), (indices[-1], dbh_y_global), (0, 0, 255), 4)
            cv2.putText(vis_img, f"{dbh_cm:.1f}cm", (indices[0], dbh_y_global-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Vẽ chiều cao cuối cùng lên ảnh
    cv2.putText(vis_img, f"Cao: {tree_height_m:.2f}m", (int(best_full_tree_box[0]), int(best_full_tree_box[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- LƯU KẾT QUẢ CUỐI CÙNG ---
    print(f"\n✅ Đang lưu kết quả tổng hợp vào {output_path}...")
    cv2.imwrite(output_path, vis_img)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy PercepTree để đo đường kính cây trong ảnh.')
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn đến ảnh đầu vào.')
    parser.add_argument('--output', type=str, required=True, help='Đường dẫn để lưu ảnh kết quả.')
    args = parser.parse_args()

    run_analysis(args.input, HUMAN_HEIGHT_CM, args.output)