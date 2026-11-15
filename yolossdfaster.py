"""
Q2: Compare YOLO26, SSD300 and Faster R-CNN on the SAME labelled detection set.

Dataset format: standard Ultralytics/YOLO detection dataset with COCO class names.
Default dataset is coco8.yaml (automatically downloaded by Ultralytics).
For a meaningful experiment, use a larger labelled validation set.

Run:
  python programs/q2_compare_yolo_ssd_fasterrcnn.py --data coco8.yaml

Optional:
  python programs/q2_compare_yolo_ssd_fasterrcnn.py --data path/to/data.yaml \
      --max-images 100 --conf 0.001

Metrics:
  - Mean matched bounding-box IoU at IoU 0.50
  - mAP@0.50
  - mAP@0.50:0.95 (COCO-style mean AP)
  - Box Dice coefficient at IoU 0.50

Important:
  Dice here is calculated for bounding boxes, not pixel masks. True mask Dice
  requires ground-truth segmentation masks and instance-segmentation outputs.
"""
from pathlib import Path
import argparse
import csv
import yaml
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    ssd300_vgg16,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)

def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=float)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)

def box_dice_from_iou(iou):
    # Equivalent to 2*intersection/(area_pred + area_gt), derived from IoU.
    return (2.0 * iou) / (1.0 + iou + 1e-9)

def compute_ap(recall, precision):
    # COCO-style 101-point interpolated AP.
    if len(recall) == 0:
        return 0.0
    recalls = np.linspace(0, 1, 101)
    p_interp = []
    for r in recalls:
        p_interp.append(np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0)
    return float(np.mean(p_interp))

def evaluate_class_ap(gt_by_image, detections, class_name, threshold):
    gt_count = sum(len(v) for v in gt_by_image.values())
    if gt_count == 0:
        return None
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_image.items()}
    tp, fp = [], []
    matched_ious = []
    for det in dets:
        gts = gt_by_image.get(det["image_id"], np.zeros((0, 4), dtype=float))
        if len(gts) == 0:
            tp.append(0); fp.append(1)
            continue
        ious = box_iou_one_to_many(det["box"], gts)
        best = int(np.argmax(ious))
        best_iou = float(ious[best])
        if best_iou >= threshold and not matched[det["image_id"]][best]:
            tp.append(1); fp.append(0)
            matched[det["image_id"]][best] = True
            matched_ious.append(best_iou)
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(gt_count, 1)
    precision = tp / np.maximum(tp + fp, 1e-9)
    return compute_ap(recall, precision), matched_ious

def calculate_metrics(ground_truth, predictions, class_names):
    aps_by_threshold = []
    ap50_values = []
    ious50 = []
    for threshold in IOU_THRESHOLDS:
        class_aps = []
        for class_name in class_names:
            gt_cls = {img: arr for img, arr in ground_truth[class_name].items()}
            det_cls = [d for d in predictions if d["class"] == class_name]
            result = evaluate_class_ap(gt_cls, det_cls, class_name, threshold)
            if result is not None:
                ap, matched_ious = result
                class_aps.append(ap)
                if abs(threshold - 0.5) < 1e-6:
                    ap50_values.append(ap)
                    ious50.extend(matched_ious)
        if class_aps:
            aps_by_threshold.append(float(np.mean(class_aps)))
    map_5095 = float(np.mean(aps_by_threshold)) if aps_by_threshold else 0.0
    map50 = float(np.mean(ap50_values)) if ap50_values else 0.0
    mean_iou = float(np.mean(ious50)) if ious50 else 0.0
    mean_dice = float(np.mean([box_dice_from_iou(x) for x in ious50])) if ious50 else 0.0
    return {
        "mean_matched_box_iou_at_0.50": mean_iou,
        "mAP50": map50,
        "mAP50_95": map_5095,
        "box_dice_at_0.50": mean_dice
    }

def image_to_label_path(image_path):
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.parent.parent / "labels" / p.with_suffix(".txt").name

def load_yolo_ground_truth(image_path, names):
    label_path = image_to_label_path(image_path)
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing annotation file {label_path}. Every validation image needs a YOLO .txt label."
        )
    with Image.open(image_path) as im:
        w, h = im.size
    boxes_by_class = {}
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes_by_class
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        name = names[cls]
        boxes_by_class.setdefault(name, []).append([x1, y1, x2, y2])
    return {k: np.asarray(v, dtype=float) for k, v in boxes_by_class.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="coco8.yaml", help="Ultralytics-format detection YAML")
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all validation images")
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold for AP calculation")
    parser.add_argument("--output", default="results_q2")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics resolves/downloads the selected dataset and returns its paths.
    data = check_det_dataset(args.data)
    val_path = Path(data["val"])
    if val_path.is_dir():
        image_paths = sorted([p for p in val_path.rglob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        image_paths = [Path(x.strip()) for x in val_path.read_text().splitlines() if x.strip()]
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise RuntimeError(f"No validation images found at {val_path}")

    names_obj = data["names"]
    if isinstance(names_obj, dict):
        names = [names_obj[i] for i in sorted(names_obj)]
    else:
        names = list(names_obj)
    class_names = list(dict.fromkeys(names))

    print(f"Evaluating {len(image_paths)} images on {device}...")
    ground_truth = {name: {} for name in class_names}
    for image_id, image_path in enumerate(image_paths):
        gt = load_yolo_ground_truth(image_path, names)
        for class_name in class_names:
            ground_truth[class_name][image_id] = gt.get(class_name, np.zeros((0, 4), dtype=float))

    predictions_by_model = {}

    # 1) YOLO26 detector
    print("Loading YOLO26...")
    yolo = YOLO("yolo26n.pt")
    yolo_predictions = []
    for image_id, image_path in enumerate(image_paths):
        result = yolo.predict(str(image_path), conf=args.conf, verbose=False)[0]
        if result.boxes is None:
            continue
        for b in result.boxes:
            cls_id = int(b.cls.item())
            cls_name = str(result.names[cls_id])
            if cls_name not in class_names:
                continue
            yolo_predictions.append({
                "image_id": image_id, "class": cls_name,
                "score": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy().astype(float)
            })
    predictions_by_model["YOLO26n"] = yolo_predictions

    # 2) Original SSD300 detector with VGG16 backbone
    print("Loading SSD300 VGG16...")
    ssd_weights = SSD300_VGG16_Weights.DEFAULT
    ssd = ssd300_vgg16(weights=ssd_weights).to(device).eval()
    ssd_categories = ssd_weights.meta["categories"]
    ssd_predictions = []
    with torch.no_grad():
        for image_id, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            tensor = to_tensor(image).to(device)
            pred = ssd([tensor])[0]
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if float(score) < args.conf:
                    continue
                cls_name = ssd_categories[int(label)]
                if cls_name not in class_names:
                    continue
                ssd_predictions.append({
                    "image_id": image_id, "class": cls_name,
                    "score": float(score),
                    "box": box.cpu().numpy().astype(float)
                })
    predictions_by_model["SSD300"] = ssd_predictions

    # 3) Faster R-CNN ResNet50-FPN
    print("Loading Faster R-CNN ResNet50-FPN...")
    fr_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    faster = fasterrcnn_resnet50_fpn(weights=fr_weights).to(device).eval()
    fr_categories = fr_weights.meta["categories"]
    faster_predictions = []
    with torch.no_grad():
        for image_id, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            tensor = to_tensor(image).to(device)
            pred = faster([tensor])[0]
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if float(score) < args.conf:
                    continue
                cls_name = fr_categories[int(label)]
                if cls_name not in class_names:
                    continue
                faster_predictions.append({
                    "image_id": image_id, "class": cls_name,
                    "score": float(score),
                    "box": box.cpu().numpy().astype(float)
                })
    predictions_by_model["Faster R-CNN"] = faster_predictions

    rows = []
    for model_name, preds in predictions_by_model.items():
        metrics = calculate_metrics(ground_truth, preds, class_names)
        rows.append({"model": model_name, "images_evaluated": len(image_paths),
                     "detections": len(preds), **metrics})
        print(model_name, metrics)

    csv_file = out_dir / "model_comparison_metrics.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison saved to: {csv_file}")
    print("Use the same dataset, images, labels and confidence threshold for a fair comparison.")

if __name__ == "__main__":
    main()
