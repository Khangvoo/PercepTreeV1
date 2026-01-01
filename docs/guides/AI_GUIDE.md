# Hướng dẫn cho AI Agent: PercepTreeV1

Tài liệu này cung cấp hướng dẫn để các AI agent có thể hiểu, cài đặt và vận hành dự án PercepTreeV1.

## 1. Tổng quan dự án

PercepTreeV1 là một dự án thị giác máy tính được thiết kế để nhận thức và phân tích cây cối. Các chức năng chính của nó bao gồm:
- Phát hiện đối tượng (có thể là cây, cành cây, v.v.) bằng các mô hình YOLO.
- Chức năng "Green Meter" để định lượng mức độ "xanh" trong một hình ảnh, có thể chạy dưới dạng script hoặc thông qua API.
- Các bản demo để xử lý hình ảnh đơn lẻ và video.

Phần cốt lõi của dự án nằm trong thư mục `inference_pretrained`.

## 2. Cấu trúc dự án

-   `inference_pretrained/`: Thư mục chính chứa các script để chạy suy luận (inference) và các mô hình chính.
    -   `run_green_meter.py`: Script chính để tính toán độ "xanh" của một hình ảnh.
    -   `green_meter_core.py`: Logic cốt lõi cho tính năng Green Meter.
    -   `api_green_meter.py`: Một API (dựa trên Flask/FastAPI) để cung cấp chức năng Green Meter.
    -   `demo_single_frame.py`: Script để chạy phát hiện đối tượng trên một ảnh đơn.
    -   `demo_video.py`: Script để chạy phát hiện đối tượng trên một video.
    -   `yolov8s-worldv2.pt`, `best.pt`: Các trọng số của mô hình YOLO.
    -   `sample_images/`: Chứa các hình ảnh mẫu để thử nghiệm.
-   `output/`: Thư mục mặc định để lưu kết quả xử lý (hình ảnh, video).
-   `configs/`: Chứa các tệp cấu hình cho mô hình (ví dụ: R-CNN).
-   `docs/guides/`: Chứa tài liệu dự án, bao gồm cả hướng dẫn này.
-   `venv/`: Môi trường ảo Python chính cho dự án này.

## 3. Cài đặt

1.  **Kích hoạt môi trường ảo:**
    ```bash
    # Trên Windows
    .\venv\Scripts\activate
    ```
2.  **Cài đặt các thư viện cần thiết:** Các thư viện của dự án không được liệt kê rõ ràng trong một tệp `requirements.txt` duy nhất. Bạn có thể cần kiểm tra các câu lệnh `import` trong các script Python (như `run_green_meter.py`, v.v.) để xác định các thư viện cần thiết. Các thư viện chính có thể bao gồm:
    -   `ultralytics` (cho YOLO)
    -   `torch`
    -   `torchvision`
    -   `opencv-python`
    -   `numpy`
    -   `flask` hoặc `fastapi` (cho API)

    Một điểm khởi đầu tốt là phân tích tệp `requirements_eval.txt` và cài đặt thêm các gói còn thiếu.

## 4. Cách chạy

Tất cả các lệnh nên được chạy từ thư mục gốc của dự án (`E:\PercepTreeV1`).

### 4.1. Chạy Green Meter

Để chạy phân tích Green Meter trên một hình ảnh:
```bash
python inference_pretrained/run_green_meter.py --image_path "đường/dẫn/tới/ảnh.jpg"
```
*(Lưu ý: Các đối số dòng lệnh như `--image_path` được suy luận và có thể cần được xác minh bằng cách kiểm tra phần `argparse` của script.)*

### 4.2. Phát hiện trên ảnh đơn

Để chạy phát hiện đối tượng trên một khung hình duy nhất:
```bash
python inference_pretrained/demo_single_frame.py --input "đường/dẫn/tới/ảnh.jpg" --output "output/ket_qua.jpg"
```
*(Lưu ý: Các đối số dòng lệnh được suy luận.)*

### 4.3. Phát hiện trên video

Để chạy phát hiện đối tượng trên một video:
```bash
python inference_pretrained/demo_video.py --input "đường/dẫn/tới/video.mp4" --output "output/ket_qua.mp4"
```
*(Lưu ý: Các đối số dòng lệnh được suy luận.)*
