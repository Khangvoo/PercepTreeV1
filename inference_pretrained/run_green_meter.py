from inference_pretrained.green_meter_core import GreenMeterAnalyzer

# ================= CẤU HÌNH INPUT =================
IMAGE_PATH = "inference_pretrained/7.jpg"
TREE_MODEL_WEIGHTS = "inference_pretrained/best.pt"
TRUNK_MODEL_WEIGHTS = "output/X-101_RGB_60k.pth"
PERSON_MODEL_WEIGHTS = "yolov8s-worldv2.pt"
HUMAN_HEIGHT_CM = 170.0
# ==================================================


def analyze_biomass(image_path: str):
    analyzer = GreenMeterAnalyzer(
        tree_model_weights=TREE_MODEL_WEIGHTS,
        trunk_model_weights=TRUNK_MODEL_WEIGHTS,
        person_model_weights=PERSON_MODEL_WEIGHTS,
        human_height_cm=HUMAN_HEIGHT_CM,
    )
    result = analyzer.analyze(image_path=image_path, save_visual=True, output_dir=".")

    print("\n====== KẾT QUẢ ======")
    print(f"🎯 DBH: {result.dbh_cm:.2f} cm")
    print(f"🌲 Height: {result.tree_height_m:.2f} m")
    print(f"🖼 Ảnh kết quả: {result.result_image}")
    print("=====================")


if __name__ == "__main__":
    analyze_biomass(IMAGE_PATH)
