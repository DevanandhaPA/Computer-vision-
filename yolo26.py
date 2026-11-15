"""
Q1: YOLO26 object detection, pixel-size measurement and instance segmentation.

Run:
  python programs/q1_yolo26_measure_segment.py --source input_images

Optional calibrated measurement:
  python programs/q1_yolo26_measure_segment.py --source input_images \
      --reference-pixels 200 --reference-cm 10

The reference object and measured object must be approximately in the same
plane and at a similar camera distance. Without calibration, only pixel
measurements are reported.
"""
from pathlib import Path
import argparse
import csv
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="input_images", help="Image file or folder")
    parser.add_argument("--weights", default="yolo26n-seg.pt",
                        help="YOLO26 segmentation weights")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", default="results_q1")
    parser.add_argument("--reference-pixels", type=float, default=None,
                        help="Pixel width of a known reference object")
    parser.add_argument("--reference-cm", type=float, default=None,
                        help="Real width of that reference object in centimetres")
    args = parser.parse_args()

    source = Path(
    r"C:\Users\Devanandha\OneDrive\Documents\M TECH\CV\Computer_Vision_Three_Questions_Programs\vision_assignment_programs\input_images")
    if not source.exists():
        raise FileNotFoundError(f"Input source not found: {source}")
    if (args.reference_pixels is None) != (args.reference_cm is None):
        raise ValueError("Supply both --reference-pixels and --reference-cm, or neither.")
    if args.reference_pixels is not None and args.reference_pixels <= 0:
        raise ValueError("--reference-pixels must be positive.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # YOLO26 instance-segmentation model (pretrained on COCO).
    model = YOLO(args.weights)
    results = model.predict(source=str(source), conf=args.conf, save=False, verbose=False)

    csv_path = out_dir / "object_measurements.csv"
    rows = []

    for image_index, result in enumerate(results):
        image_path = Path(result.path)
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = original.shape[:2]
        annotated = original.copy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        boxes = result.boxes

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
            box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            mask_area = ""
            if masks is not None and i < len(masks):
                # Resize predicted mask to original image dimensions.
                mask = cv2.resize(masks[i].astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST)
                mask_bool = mask > 0
                mask_area = int(mask_bool.sum())
                # Transparent coloured overlay for the instance mask.
                color = np.array([40, 190, 80], dtype=np.uint8)
                overlay = annotated.copy()
                overlay[mask_bool] = (
                    0.55 * overlay[mask_bool] + 0.45 * color
                ).astype(np.uint8)
                annotated = overlay

            real_w = ""
            real_h = ""
            if args.reference_pixels is not None:
                cm_per_pixel = args.reference_cm / args.reference_pixels
                real_w = round(box_w * cm_per_pixel, 2)
                real_h = round(box_h * cm_per_pixel, 2)

            label = f"{class_name} {confidence:.2f} {box_w}x{box_h}px"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            rows.append({
                "image": image_path.name,
                "class": class_name,
                "confidence": round(confidence, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width_pixels": box_w,
                "height_pixels": box_h,
                "mask_area_pixels": mask_area,
                "estimated_width_cm": real_w,
                "estimated_height_cm": real_h
            })

        save_path = out_dir / f"{image_path.stem}_yolo26_segmented.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"{image_path.name}: {len(boxes)} instance(s) detected")
        print(f"  Saved: {save_path}")

    fieldnames = [
        "image", "class", "confidence", "x1", "y1", "x2", "y2",
        "width_pixels", "height_pixels", "mask_area_pixels",
        "estimated_width_cm", "estimated_height_cm"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV report saved to: {csv_path}")
    if args.reference_pixels is None:
        print("Note: real-world dimensions cannot be inferred from one ordinary image alone.")
        print("Only pixel width, pixel height and mask area are reported.")
    else:
        print("Centimetre values are approximate and assume the reference and object")
        print("are at similar depth and lie approximately in the same plane.")

if __name__ == "__main__":
    main()
