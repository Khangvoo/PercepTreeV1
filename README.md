# PercepTreeV1

Official code repository for the papers:

<div align="left">
  <img width="100%" alt="DINO illustration" src=".github/figure6.png">
</div>

- [Tree Detection and Diameter Estimation Based on Deep Learning](https://academic.oup.com/forestry/advance-article-abstract/doi/10.1093/forestry/cpac043/6779886?utm_source=advanceaccess&utm_campaign=forestry&utm_medium=email), published in *Forestry: An International Journal Of Forest Research*. Preprint version (soon).

<div align="left">
  <img width="100%" alt="DINO illustration" src=".github/detection_synth.jpg">
</div>

- [Training Deep Learning Algorithms on Synthetic Forest Images for Tree Detection](http://arxiv.org/abs/2210.04104), presented at *ICRA 2022 IFRRIA Workshop*. The video presentation is [available](https://www.youtube.com/watch?v=8KT97ZFMC0g&list=PLbiomSAe-K8896UHcLVkNWP66DaFpS7j5&index=2).

---

## Hai luồng sử dụng chính
- **Chạy nhanh với model có sẵn**: tải weight, chạy demo ảnh/video.
- **Tạo mô hình cho thiết bị di động**: huấn luyện lại rồi xuất sang định dạng thân thiện mobile (TorchScript/ONNX).

## Cấu trúc thư mục (chính)
- `inference_pretrained/`: script demo ảnh/video và mẫu `sample_images/`.
- `mobile_training/`: script huấn luyện `train_synth_RGB.py`.
- `configs/`: file cấu hình Detectron2.
- `output/`: nơi đặt/tạo weight `.pth`, kết quả export.

---

## 1. Chạy bằng model có sẵn
Yêu cầu: Python 3.8+, PyTorch, Detectron2, OpenCV (`pip install opencv-python tqdm albumentations`).

### Bước 1: tải weight
Pre-trained model tương thích với Detectron2 config.

#### Mask R-CNN (SynthTree43k)
<table>
  <tr>
    <th>Backbone</th>
    <th>Modality</th>
    <th>box AP50</th>
    <th>mask AP50</th>
    <th colspan="6">Download</th>
  </tr>
  <tr>
    <td>R-50-FPN</td>
    <td>RGB</td>
    <td>87.74</td>
    <td>69.36</td>
    <td><a href="https://drive.google.com/file/d/1pnJZ3Vc0SVTn_J8l_pwR4w1LMYnFHzhV/view?usp=sharing">model</a></td>
  <tr>
    <td>R-101-FPN</td>
    <td>RGB</td>
    <td>88.51</td>
    <td>70.53</td>
    <td><a href="https://drive.google.com/file/d/1ApKm914PuKm24kPl0sP7-XgG_Ottx5tJ/view?usp=sharing">model</a></td>
  <tr>
    <td>X-101-FPN</td>
    <td>RGB</td>
    <td>88.91</td>
    <td>71.07</td>
    <td><a href="https://drive.google.com/file/d/1Q5KV5beWVZXK_vlIED1jgpf4XJgN71ky/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>R-50-FPN</td>
    <td>Depth</td>
    <td>89.67</td>
    <td>70.66</td>
    <td><a href="https://drive.google.com/file/d/1bnH7ZSXWoOJx5AkbNeHf_McV46qiKIkY/view?usp=sharing">model</a></td>
  <tr>
    <td>R-101-FPN</td>
    <td>Depth</td>
    <td>89.89</td>
    <td>71.65</td>
    <td><a href="https://drive.google.com/file/d/1DgMscnTIGty7y9-VNcq1zERrevfT3b_L/view?usp=sharing">model</a></td>
  <tr>
    <td>X-101-FPN</td>
    <td>Depth</td>
    <td>87.41</td>
    <td>68.19</td>
    <td><a href="https://drive.google.com/file/d/1rsCbLSvFf2I47FJK4vhhv0du5uCV6zjO/view?usp=sharing">model</a></td>
  </tr>
</table>

#### Mask R-CNN (CanaTree100 finetune)
<table>
  <tr>
    <th>Backbone</th>
    <th>Description</th>
    <th colspan="6">Download</th>
  </tr>
  <tr>
    <td>X-101-FPN</td>
    <td>Trained on fold 01, good for inference.</td>
    <td><a href="https://drive.google.com/file/d/108tORWyD2BFFfO5kYim9jP0wIVNcw0OJ/view?usp=sharing">model</a></td>
  </tr>
</table>

Đặt file `.pth` vào thư mục `output/`.

### Bước 2: chạy demo
- **Ảnh**: chỉnh `model_name` và `image_path` trong `inference_pretrained/demo_single_frame.py`, rồi chạy:
  ```bash
  python inference_pretrained/demo_single_frame.py
  ```
- **Video**: chỉnh `model_name` và `video_path` trong `inference_pretrained/demo_video.py`, rồi chạy:
  ```bash
  python inference_pretrained/demo_video.py
  ```

### Đánh giá YOLO-World vs. model Roboflow (mAP/AP person/tree)
- Chuẩn bị `data.yaml` (định nghĩa train/val/test + names).  
- Tạo môi trường và cài gói:
  ```bash
  python3 -m venv .venv_eval
  source .venv_eval/bin/activate
  pip install -r requirements_eval.txt
  ```
- Chạy đánh giá (ví dụ):
  ```bash
  python inference_pretrained/eval_yolo_vs_roboflow.py \
    --data /path/to/data.yaml \
    --split test \
    --imgsz 640 \
    --conf 0.25 \
    --yolo-world-weights inference_pretrained/yolov8s-worldv2.pt \
    --roboflow-weights /path/to/roboflow.pt \
    --out output/eval/results.json
  ```
Kết quả in ra console và lưu JSON (mAP@50-95, mAP@50, AP từng lớp person/tree cho cả hai mô hình).

<div align="left">
  <img width="70%" alt="DINO illustration" src=".github/trailer_0.gif">
</div>

---

## 2. Tạo mô hình cho thiết bị di động
Quy trình: chuẩn bị dữ liệu ➜ huấn luyện ➜ xuất sang TorchScript/ONNX ➜ tối ưu trên thiết bị.

### 2.1. Dữ liệu huấn luyện
<table>
  <tr>
    <th>Dataset name</th>
    <th>Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td>SynthTree43k</td>
    <td>43k ảnh tổng hợp, 190k cây đã gán nhãn. Bao gồm train/val/test. (84.6 GB) 
    <a href="https://drive.google.com/drive/folders/1sdJtmQ4H8aHzYZ9TWz8xpm06R9mQMd34?usp=sharing">annos</a>
    </td>
    <td><a href="http://norlab.s3.valeria.science/SynthTree43k.zip?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2274019241&Signature=KfOgwrHX8WHejopspqQ8XMwlMJE%3D">S3 storage</a></td>
  <tr>
  <tr>
    <td>SynthTree43k</td>
    <td>Depth images.</td>
    <td><a href="https://ulavaldti-my.sharepoint.com/:u:/g/personal/vigro7_ulaval_ca/EfglPMp555FGvwKGDEp9eRwBn_jXK-7vMPfYxDAVHbzTgg?e=l9HFd4">OneDrive </a></td>
  <tr>
  <tr>
    <td>CanaTree100</td>
    <td>100 ảnh thực, 920 cây đã gán nhãn, có sẵn 5 fold train/val/test.</td>
    <td><a href="http://norlab.s3.valeria.science/neats/CanaTree100.zip?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2339251391&Signature=6beuqoLRQfCTaSpoC7ZKELhJwhY%3D">S3 storage </a></td>
  <tr>
</table>

Annotation đầy đủ đã kèm trong link tải. Nếu cần annotation toàn bộ cây:
<a href="https://drive.google.com/file/d/1AZUtdrNJGPWgqEwUrRin6OKwE_KGavZq/view?usp=sharing">train_RGB_entire_tree.json</a>,
<a href="https://drive.google.com/file/d/1doTRoLvQ1pGaNb75mx-SOr5aEVBLNnZe/view?usp=sharing">val_RGB_entire_tree.json</a>,
<a href="https://drive.google.com/file/d/1ZMYqFylSrx2KDHR-2TSoXFq-_uoyb6Qp/view?usp=share_link">test_RGB_entire_tree.json</a>.

### 2.2. Huấn luyện
Script chính: `mobile_training/train_synth_RGB.py`.
- Đặt đường dẫn ảnh trong biến `img_dir`.
- Đặt đường dẫn annotation COCO trong `./output/train_RGB.json`, `val_RGB.json`, `test_RGB.json` (hoặc chỉnh lại biến tương ứng).
- Chọn backbone/config Detectron2 trong phần `cfg.merge_from_file(...)`.
- Chạy:
  ```bash
  python mobile_training/train_synth_RGB.py
  ```
Checkpoint sẽ nằm trong `output/model_final.pth`.

### 2.3. Xuất cho thiết bị di động
Xuất sang TorchScript (PyTorch Mobile) hoặc ONNX để chuyển tiếp sang CoreML/TFLite.

TorchScript (quick): dùng Detectron2 TracingAdapter.
```bash
python - <<'PY'
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # hoặc dùng config tùy biến bạn đã train
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.DEVICE = "cpu"
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.freeze()

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

sample_inputs = [{"image": torch.randn(3, 720, 720)}]  # chỉnh kích thước phù hợp với dữ liệu của bạn
traceable_model = TracingAdapter(model, sample_inputs)
ts_model = torch.jit.trace(traceable_model, (sample_inputs,))
ts_model.save("output/model_mobile.ts")
print("Saved TorchScript to output/model_mobile.ts")
PY
```

ONNX (để chuyển CoreML/TFLite):
```bash
python - <<'PY'
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.DEVICE = "cpu"
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.freeze()

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

sample_inputs = [{"image": torch.randn(3, 720, 720)}]
traceable_model = TracingAdapter(model, sample_inputs)

torch.onnx.export(
    traceable_model,
    (sample_inputs,),
    "output/model_mobile.onnx",
    input_names=["images"],
    output_names=["detections"],
    opset_version=16,
    dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}}
)
print("Saved ONNX to output/model_mobile.onnx")
PY
```

Sau khi xuất:
- PyTorch Mobile: optimize bằng `torch.utils.mobile_optimizer.optimize_for_mobile` trước khi đóng gói vào app Android/iOS.
- ONNX: dùng `onnxsim` để simplify, rồi chuyển sang CoreML (`coremltools`) hoặc TFLite (`onnx-tflite` hoặc `tf` converter).

---

# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```bibtex
@article{grondin2022tree,
    author = {Grondin, Vincent and Fortin, Jean-Michel and Pomerleau, François and Giguère, Philippe},
    title = {Tree detection and diameter estimation based on deep learning},
    journal = {Forestry: An International Journal of Forest Research},
    year = {2022},
    month = {10},
}

@inproceedings{grondin2022training,
  title={Training Deep Learning Algorithms on Synthetic Forest Images for Tree Detection},
  author={Grondin, Vincent and Pomerleau, Fran{\c{c}}ois and Gigu{\`e}re, Philippe},
  booktitle={ICRA 2022 Workshop in Innovation in Forestry Robotics: Research and Industry Adoption},
  year={2022}
}
```
