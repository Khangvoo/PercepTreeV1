import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from ultralytics import YOLO, YOLOWorld

# Matplotlib phải dùng backend không GUI khi chạy server
matplotlib.use("Agg")


def _find_class_ids(names, target_keywords):
    """
    Tìm ID lớp dựa trên từ khóa. Hỗ trợ list từ khóa.
    """
    if not isinstance(target_keywords, list):
        target_keywords = [target_keywords]

    found_ids = []
    items = names.items() if isinstance(names, dict) else enumerate(names)

    for k, v in items:
        name_lower = str(v).lower()
        for kw in target_keywords:
            if kw.lower() in name_lower:
                found_ids.append(int(k))
                break
    return found_ids


def _pick_best_box(results, class_ids, mode="area"):
    """Chọn bounding box tốt nhất dựa trên diện tích hoặc chiều cao."""
    best_box = None
    best_score = -1
    fallback = None

    if not results:
        return None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            b = box.xyxy[0].cpu().numpy().astype(int)
            w_b = b[2] - b[0]
            h_b = b[3] - b[1]

            score = h_b if mode == "height" else w_b * h_b

            if class_ids and cls_id not in class_ids:
                if fallback is None or score > (fallback[3] - fallback[1]) * (fallback[2] - fallback[0]):
                    fallback = b
                continue

            if score > best_score:
                best_score = score
                best_box = b

    if best_box is not None:
        return best_box
    return fallback


@dataclass
class GreenMeterResult:
    dbh_cm: float
    tree_height_m: float
    scale_cm: float
    used_human_height_cm: float
    person_box: List[int]
    tree_box: Optional[List[int]]
    trunk_box: List[int]
    result_image: Optional[str]
    messages: List[str]

    def to_dict(self):
        data = asdict(self)
        return data


class GreenMeterAnalyzer:
    """
    Bao đóng toàn bộ logic inference để có thể tái sử dụng cho CLI & API.
    """

    def __init__(
        self,
        tree_model_weights: str = "inference_pretrained/best.pt",
        trunk_model_weights: str = "output/X-101_RGB_60k.pth",
        person_model_weights: str = "yolov8s-worldv2.pt",
        human_height_cm: float = 170.0,
        device: Optional[str] = None,
    ):
        self.tree_model_weights = tree_model_weights
        self.trunk_model_weights = trunk_model_weights
        self.person_model_weights = person_model_weights
        self.human_height_cm = human_height_cm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._messages: List[str] = []

        self.person_model = None
        self.tree_bbox_model = None
        self.predictor = None
        self.tree_metadata = None

        self._load_models()

    def _log(self, msg: str):
        self._messages.append(msg)

    def _load_models(self):
        self._messages = []
        self._log(f"Thiết bị: {self.device.upper()}")

        try:
            self.person_model = YOLOWorld(self.person_model_weights)
            self.person_model.set_classes(["person"])
            self.person_model.to(self.device)
            self._log("Đã load YOLO-World (Người).")
        except Exception as e:
            raise RuntimeError(f"Lỗi load YOLO-World: {e}") from e

        try:
            self.tree_bbox_model = YOLO(self.tree_model_weights)
            self.tree_bbox_model.to(self.device)
            self._log("Đã load YOLO Custom (Cây).")
        except Exception as e:
            raise RuntimeError(f"Lỗi load YOLO Tree: {e}") from e

        if not os.path.exists(self.trunk_model_weights):
            raise FileNotFoundError(f"Không tìm thấy trunk model: {self.trunk_model_weights}")

        try:
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.WEIGHTS = self.trunk_model_weights
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = self.device
            self.predictor = DefaultPredictor(cfg)

            self.tree_metadata = MetadataCatalog.get("my_tree_dataset_infer")
            self.tree_metadata.thing_classes = ["trunk"]
            self._log("Đã load Detectron2 (Thân).")
        except Exception as e:
            raise RuntimeError(f"Lỗi load Detectron2: {e}") from e

    def analyze(
        self,
        image_path: Optional[str] = None,
        image_bgr: Optional[np.ndarray] = None,
        image_name: Optional[str] = None,
        save_visual: bool = True,
        output_dir: str = ".",
        person_height_cm: Optional[float] = None,
    ) -> GreenMeterResult:
        """
        Chạy phân tích trên ảnh đầu vào.
        """
        self._messages = []
        if image_bgr is None:
            if image_path is None:
                raise ValueError("Cần truyền image_path hoặc image_bgr.")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Không đọc được ảnh: {image_path}")
            image_name = image_name or os.path.basename(image_path)
        else:
            if image_name is None:
                image_name = "uploaded.jpg"

        img = image_bgr
        h_img, w_img, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Người ---
        human_height_cm = (
            float(person_height_cm)
            if person_height_cm is not None and person_height_cm > 0
            else self.human_height_cm
        )
        self._log(f"Chiều cao tham chiếu: {human_height_cm:.1f} cm")
        self._log("Tìm người (scale).")
        person_ids = _find_class_ids(self.person_model.names, "person")
        person_results = self.person_model.predict(
            img_rgb, conf=0.15, verbose=False, device=self.device, classes=person_ids
        )
        person_box = _pick_best_box(person_results, person_ids, mode="height")

        if person_box is None:
            self._log("Không thấy người -> dùng giá trị mặc định.")
            best_person_h = h_img / 4
            person_box = [10, h_img - int(best_person_h), 50, h_img]
        else:
            self._log("Đã tìm thấy người.")

        best_person_h = person_box[3] - person_box[1]
        scale_cm = human_height_cm / best_person_h
        self._log(f"Tỷ lệ: 1px = {scale_cm:.2f} cm")

        # --- Cây ---
        self._log("Tìm BBox cây.")
        tree_class_ids = _find_class_ids(self.tree_bbox_model.names, ["tree", "cay"])
        if not tree_class_ids and len(self.tree_bbox_model.names) == 1:
            tree_class_ids = [0]

        tree_results = self.tree_bbox_model.predict(
            img_rgb, conf=0.20, verbose=False, device=self.device, classes=tree_class_ids
        )
        tree_box_yolo = _pick_best_box(tree_results, tree_class_ids, mode="area")

        # --- Detectron2 ---
        self._log("Phân đoạn thân cây.")
        crop_offset_x, crop_offset_y = 0, 0
        detectron_input = img

        if tree_box_yolo is not None:
            x1, y1, x2, y2 = tree_box_yolo
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop_offset_x, crop_offset_y = x1, y1
            detectron_input = img[y1:y2, x1:x2]
            self._log("Đã crop ảnh theo bbox cây.")
        else:
            self._log("Không tìm thấy bbox cây -> dùng toàn ảnh.")

        outputs = self.predictor(detectron_input)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            raise RuntimeError("Không tìm thấy thân cây.")

        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_masks = instances.pred_masks.numpy()
        pred_boxes[:, [0, 2]] += crop_offset_x
        pred_boxes[:, [1, 3]] += crop_offset_y

        full_masks = []
        for m in pred_masks:
            mask_full = np.zeros((h_img, w_img), dtype=bool)
            h_m, w_m = m.shape
            mask_full[crop_offset_y : crop_offset_y + h_m, crop_offset_x : crop_offset_x + w_m] = m
            full_masks.append(mask_full)
        pred_masks = np.stack(full_masks, axis=0)

        center_target_x = (
            (tree_box_yolo[0] + tree_box_yolo[2]) / 2 if tree_box_yolo is not None else w_img // 2
        )
        ground_ref_y = person_box[3]

        best_idx = -1
        min_dist = float("inf")

        for i, box in enumerate(pred_boxes):
            if abs(box[3] - ground_ref_y) < (h_img * 0.3):
                center_trunk_x = (box[0] + box[2]) / 2
                dist = abs(center_trunk_x - center_target_x)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

        if best_idx == -1:
            areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            best_idx = int(np.argmax(areas))
            self._log("Chọn thân cây theo diện tích (fallback).")
        else:
            self._log("Chọn thân cây gần trung tâm nhất.")

        tree_mask = pred_masks[best_idx]
        box = pred_boxes[best_idx]
        tree_root_y = int(box[3])

        pixel_1m3 = 130.0 / scale_cm
        dbh_y = int(tree_root_y - pixel_1m3)
        if dbh_y < box[1]:
            dbh_y = int((box[1] + tree_root_y) / 2)
        if dbh_y < 0:
            dbh_y = 0

        y_slice_start = max(0, dbh_y - 5)
        y_slice_end = min(h_img, dbh_y + 5)
        mask_slice = tree_mask[y_slice_start:y_slice_end, :]
        row_widths = np.sum(mask_slice, axis=1)
        valid_widths = row_widths[row_widths > 0]

        avg_width_px = np.median(valid_widths) if len(valid_widths) > 0 else (box[2] - box[0]) * 0.8
        dbh_cm = float(avg_width_px * scale_cm)

        if tree_box_yolo is not None:
            h_px = tree_box_yolo[3] - tree_box_yolo[1]
            tree_height_m = float((h_px * scale_cm) / 100)
        else:
            h_px = box[3] - box[1]
            tree_height_m = float((h_px * scale_cm) / 100)

        result_path = None
        if save_visual:
            os.makedirs(output_dir, exist_ok=True)
            result_filename = f"result_{os.path.splitext(image_name)[0].replace('.', '_')}.png"
            result_path = os.path.join(output_dir, result_filename)
            self._save_visualization(
                img,
                person_box,
                tree_box_yolo,
                tree_mask,
                dbh_y,
                dbh_cm,
                pred_boxes,
                pred_masks,
                best_idx,
                instances,
                result_path,
                height=tree_height_m,
            )

        result = GreenMeterResult(
            dbh_cm=dbh_cm,
            tree_height_m=tree_height_m,
            scale_cm=float(scale_cm),
            used_human_height_cm=float(human_height_cm),
            person_box=list(map(int, person_box)),
            tree_box=list(map(int, tree_box_yolo)) if tree_box_yolo is not None else None,
            trunk_box=list(map(int, pred_boxes[best_idx])),
            result_image=result_path,
            messages=list(self._messages),
        )
        return result

    def _save_visualization(
        self,
        img,
        person_box,
        tree_box_yolo,
        tree_mask,
        dbh_y,
        dbh_cm,
        pred_boxes,
        pred_masks,
        best_idx,
        instances,
        output_path,
        height,
    ):
        full_instances = Instances(image_size=img.shape[:2])
        full_instances.pred_boxes = Boxes(torch.tensor(pred_boxes[[best_idx]]))
        full_instances.pred_masks = torch.tensor(pred_masks[[best_idx]])
        full_instances.scores = torch.tensor(instances.scores.numpy()[[best_idx]])
        full_instances.pred_classes = torch.tensor(instances.pred_classes.numpy()[[best_idx]])

        v = Visualizer(img[:, :, ::-1], metadata=self.tree_metadata, scale=1.0)
        out = v.draw_instance_predictions(full_instances)
        vis_img = out.get_image()[:, :, ::-1]
        vis_img = np.ascontiguousarray(vis_img, dtype=np.uint8)

        cv2.rectangle(
            vis_img,
            (int(person_box[0]), int(person_box[1])),
            (int(person_box[2]), int(person_box[3])),
            (255, 0, 0),
            2,
        )
        cv2.putText(vis_img, "Ref Person", (int(person_box[0]), int(person_box[1]) - 5), 0, 0.6, (255, 0, 0), 2)

        if tree_box_yolo is not None:
            cv2.rectangle(
                vis_img,
                (int(tree_box_yolo[0]), int(tree_box_yolo[1])),
                (int(tree_box_yolo[2]), int(tree_box_yolo[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis_img, "Tree Area", (int(tree_box_yolo[0]), int(tree_box_yolo[1]) - 5), 0, 0.6, (0, 255, 0), 2
            )

        idx_dbh = np.where(tree_mask[dbh_y, :])[0]
        if len(idx_dbh) > 0:
            cv2.line(vis_img, (int(idx_dbh[0]), int(dbh_y)), (int(idx_dbh[-1]), int(dbh_y)), (0, 0, 255), 3)
            cv2.putText(
                vis_img, f"D={dbh_cm:.1f}cm", (int(idx_dbh[0]) - 60, int(dbh_y)), 0, 0.7, (0, 0, 255), 2
            )

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"H: {height:.1f}m | D: {dbh_cm:.1f}cm")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
