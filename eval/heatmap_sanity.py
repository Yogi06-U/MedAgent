"""Evaluation harness for Grad-CAM anatomical localization sanity:
Evaluates lung peak alignment across 20 radiographs with lung findings.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision
import torchxrayvision as xrv

from medagent.agents.imaging import get_densenet_model, compute_gradcam, load_radiograph


def generate_coarse_lung_mask(h: int = 224, w: int = 224) -> np.ndarray:
    """
    Generate coarse bilateral anatomical lung field mask representing the thoracic cavity:
    Apices (y ~ 0.12*h) down through mid-lungs and costophrenic recesses (y ~ 0.95*h),
    spanning rib-to-rib from lateral right to lateral left (x in [0.10*w, 0.90*w]).
    """
    mask = np.zeros((h, w), dtype=bool)
    mask[int(0.12 * h):int(0.95 * h), int(0.10 * w):int(0.90 * w)] = True
    return mask


def generate_test_radiograph_suite(base_path: str = "samples/sample_cxr.png", total_images: int = 20) -> list:
    """Generate 20 distinct chest radiographs with lung findings across varied anatomical zones."""
    base_arr, _, _ = load_radiograph(base_path)
    if base_arr is None:
        raise FileNotFoundError(f"Base radiograph not found at {base_path}")

    base_224 = np.array(Image.fromarray(base_arr.astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)).astype(np.float32)

    radiographs = []
    # Zones: combinations of left/right and upper/mid/lower
    for i in range(total_images):
        side = "right" if (i % 2 == 0) else "left"
        zone_idx = (i // 2) % 3  # 0: upper, 1: mid, 2: lower
        zone_name = ["upper", "mid", "lower"][zone_idx]

        # Coordinates
        x_center = int(0.30 * 224) if side == "right" else int(0.70 * 224)
        y_center = int((0.28 + zone_idx * 0.28) * 224)

        img = base_224.copy()

        # Add focal lung infiltrate
        y, x = np.ogrid[:224, :224]
        radius = 14 + (i % 5) * 2
        intensity = 60.0 + (i % 6) * 12.0
        focal = ((x - x_center)**2 / radius**2 + (y - y_center)**2 / radius**2) <= 1.0
        img[focal] = np.clip(img[focal] + intensity, 0.0, 255.0)

        # Contrast variation
        contrast_factor = 0.90 + (i % 4) * 0.06
        img = np.clip(128.0 + (img - 128.0) * contrast_factor, 0.0, 255.0)

        radiographs.append({
            "image_index": i + 1,
            "side": side,
            "zone": zone_name,
            "x_center": x_center,
            "y_center": y_center,
            "array": img,
        })

    return radiographs


def evaluate_heatmap_sanity() -> dict:
    """Evaluate share of lung-finding heat-map peaks inside coarse lung mask on 20 radiographs."""
    model = get_densenet_model()
    if model is None:
        return {"status": "unavailable", "message": "DenseNet model weights could not be loaded."}

    lung_mask = generate_coarse_lung_mask(224, 224)
    suite = generate_test_radiograph_suite(total_images=20)
    total_images = len(suite)
    peaks_inside_mask = 0

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    pathologies = list(model.pathologies)
    target_idx = 0
    for idx, p in enumerate(pathologies):
        if p.lower() in ("consolidation", "infiltration", "pneumonia", "effusion"):
            target_idx = idx
            break

    target_pathology = pathologies[target_idx]
    details = []

    for item in suite:
        i = item["image_index"]
        raw_arr = item["array"]

        norm_arr = xrv.datasets.normalize(raw_arr, 255)
        if norm_arr.ndim == 2:
            norm_3d = norm_arr[None, ...]
        else:
            norm_3d = norm_arr

        np_224 = transform(norm_3d)
        input_tensor = torch.from_numpy(np_224).float().unsqueeze(0)

        cam = compute_gradcam(model, input_tensor, target_idx)

        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
        is_inside = bool(lung_mask[peak_y, peak_x])

        if is_inside:
            peaks_inside_mask += 1

        print(
            f"[{i}/{total_images}] Radiograph {i} ({item['side']} {item['zone']} target): "
            f"peak at ({peak_y}, {peak_x}) -> inside thoracic lung mask: {is_inside}",
            flush=True,
        )

        details.append({
            "image_index": i,
            "side": item["side"],
            "zone": item["zone"],
            "peak_y": int(peak_y),
            "peak_x": int(peak_x),
            "inside_lung_mask": is_inside,
        })

    hit_rate = peaks_inside_mask / total_images

    result_data = {
        "status": "ok",
        "total_images": total_images,
        "peaks_inside_mask": peaks_inside_mask,
        "hit_rate": round(hit_rate, 4),
        "target_pathology": target_pathology,
        "details": details,
    }

    with open("eval/heatmap_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return result_data


if __name__ == "__main__":
    res = evaluate_heatmap_sanity()
    print("=" * 60)
    print("HEATMAP SANITY EVALUATION (20 Chest Radiographs)")
    print("=" * 60)
    print(f"Target Pathology Evaluated    : {res.get('target_pathology')}")
    print(f"Total Test Radiographs        : {res.get('total_images')}")
    print(f"Peaks Inside Coarse Lung Mask : {res.get('peaks_inside_mask')}/{res.get('total_images')}")
    print(f"Share Inside Lung Mask        : {res.get('hit_rate'):.1%}")
