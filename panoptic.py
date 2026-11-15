"""
Q3: Panoptic segmentation using pretrained Mask2Former on COCO panoptic.

Run:
  python programs/q3_panoptic_segmentation.py --image input_images/photo.jpg

The script produces a panoptic ID map, a colour visualisation, and a CSV
listing each segment's category and thing/stuff status.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image")
    parser.add_argument("--model", default="facebook/mask2former-swin-small-coco-panoptic")
    parser.add_argument("--output", default="results_q3")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")

    print("Loading pretrained COCO panoptic Mask2Former model...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[(image.height, image.width)],
        threshold=args.threshold,
        mask_threshold=0.5
    )[0]

    panoptic = result["segmentation"].cpu().numpy().astype(np.int32)
    segments_info = result["segments_info"]

    # Give each segment ID a reproducible colour and build a readable legend.
    rng = np.random.default_rng(7)
    colour_map = {}
    for segment in segments_info:
        sid = int(segment["id"])
        colour_map[sid] = rng.integers(40, 240, size=3, dtype=np.uint8)

    colour_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for sid, colour in colour_map.items():
        colour_image[panoptic == sid] = colour

    # Blend panoptic colours over the original image.
    original = np.asarray(image).astype(np.float32)
    overlay = (0.55 * original + 0.45 * colour_image.astype(np.float32)).clip(0, 255).astype(np.uint8)

    panoptic_path = out_dir / "panoptic_id_map.png"
    colour_path = out_dir / "panoptic_colours.png"
    overlay_path = out_dir / "panoptic_overlay.png"
    Image.fromarray(panoptic.astype(np.uint16), mode="I;16").save(panoptic_path)
    Image.fromarray(colour_image).save(colour_path)
    Image.fromarray(overlay).save(overlay_path)

    csv_path = out_dir / "panoptic_segments.csv"
    fields = ["segment_id", "label_id", "label", "is_thing", "pixel_count", "score"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seg in segments_info:
            sid = int(seg["id"])
            label_id = int(seg["label_id"])
            label = model.config.id2label.get(label_id, str(label_id))
            writer.writerow({
                "segment_id": sid,
                "label_id": label_id,
                "label": label,
                "is_thing": bool(seg.get("was_fused", False)) is False
                            and bool(getattr(model.config, "is_thing_map", {}).get(label_id, False)),
                "pixel_count": int((panoptic == sid).sum()),
                "score": round(float(seg.get("score", 0.0)), 4)
            })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Original image")
    axes[1].imshow(colour_image); axes[1].set_title("Panoptic segments")
    axes[2].imshow(overlay); axes[2].set_title("Panoptic overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "panoptic_visualization.png", dpi=160)
    plt.close()

    print(f"Segments found: {len(segments_info)}")
    print(f"Saved: {panoptic_path}")
    print(f"Saved: {colour_path}")
    print(f"Saved: {overlay_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'panoptic_visualization.png'}")
    print("Model configuration:")
    print(model.config)
    print("Available labels:", model.config.id2label)
    print("Thing map:", getattr(model.config, "is_thing_map", "Not available"))

if __name__ == "__main__":
    main()
