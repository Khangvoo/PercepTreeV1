import argparse
import json
import os
import time
from pathlib import Path

import yaml
from ultralytics import YOLO, YOLOWorld
import torch
import torch.serialization

# Torch 2.6+ defaults weights_only=True; allow YOLO-World pickled classes/modules
try:
    from ultralytics.nn.tasks import WorldModel
    from ultralytics.nn.modules.conv import Conv, DWConv
    from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, C3, C3x
    from ultralytics.nn.modules.head import Detect, Classify, Segment, Pose, WorldDetect
    torch.serialization.add_safe_globals(
        [
            WorldModel,
            WorldDetect,
            Detect,
            Classify,
            Segment,
            Pose,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.container.ModuleDict,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.linear.Linear,
            Conv,
            DWConv,
            C2f,
            SPPF,
            Bottleneck,
            C3,
            C3x,
        ]
    )
except Exception:
    pass


def load_class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names") or data.get("class_names")
    if isinstance(names, dict):
        # Convert {0: "person", 1: "tree"} to ["person", "tree"]
        names = [names[k] for k in sorted(names.keys())]
    if not names:
        raise ValueError("Dataset yaml must contain `names`.")
    return names


def summarize_results(res, class_names):
    summary = {
        "map50_95": float(res.box.map),
        "map50": float(res.box.map50),
        "per_class_ap50_95": {},
    }
    for cls_name, ap in zip(class_names, res.box.maps):
        summary["per_class_ap50_95"][cls_name] = float(ap)
    return summary


def eval_yolo_world(weights, data_yaml, class_names, imgsz, conf, split):
    model = YOLOWorld(weights)
    model.set_classes(class_names)
    res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        split=split,
        verbose=False,
    )
    return summarize_results(res, class_names)


def eval_yolov8(weights, data_yaml, imgsz, conf, split, class_names):
    model = YOLO(weights)
    res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        split=split,
        verbose=False,
    )
    return summarize_results(res, class_names)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-World and Roboflow YOLOv8 model on the same dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml (train/val/test and names defined here).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--yolo-world-weights",
        default="inference_pretrained/yolov8s-worldv2.pt",
        help="Path to YOLO-World weights.",
    )
    parser.add_argument(
        "--roboflow-weights",
        required=True,
        help="Path to Roboflow-exported YOLOv8 weights (.pt).",
    )
    parser.add_argument(
        "--out",
        default="output/eval/results.json",
        help="Path to save metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = args.data
    class_names = load_class_names(data_yaml)

    os.makedirs(Path(args.out).parent, exist_ok=True)

    print(f"🧪 Evaluating split='{args.split}' on {data_yaml}")
    print(f"Classes: {class_names}")
    print("\n1) YOLO-World:")
    yolo_world_metrics = eval_yolo_world(
        args.yolo_world_weights,
        data_yaml,
        class_names,
        imgsz=args.imgsz,
        conf=args.conf,
        split=args.split,
    )
    print(json.dumps(yolo_world_metrics, indent=2))

    print("\n2) Roboflow YOLOv8:")
    roboflow_metrics = eval_yolov8(
        args.roboflow_weights,
        data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        split=args.split,
        class_names=class_names,
    )
    print(json.dumps(roboflow_metrics, indent=2))

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_yaml": data_yaml,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "yolo_world_weights": args.yolo_world_weights,
        "roboflow_weights": args.roboflow_weights,
        "yolo_world": yolo_world_metrics,
        "roboflow": roboflow_metrics,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✅ Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
