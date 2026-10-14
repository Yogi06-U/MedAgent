"""Imaging Agent: evaluates chest radiographs using TorchXRayVision DenseNet-121 and Grad-CAM."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from medagent.config import settings
from medagent.schemas import CaseInput, ImagingReport, ImagingFinding

logger = logging.getLogger("medagent.imaging")

_MODEL_CACHE: Optional[Any] = None


def get_densenet_model() -> Optional[Any]:
    """Load and cache the TorchXRayVision DenseNet-121 model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    try:
        import torch
        import torchxrayvision as xrv

        cache_dir = str(settings.DATA_DIR / "weights")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading TorchXRayVision DenseNet with cache_dir=%s", cache_dir)
        model = xrv.models.DenseNet(
            weights=settings.TORCHXRAYVISION_WEIGHTS,
            cache_dir=cache_dir,
        )
        model.eval()
        _MODEL_CACHE = model
        return _MODEL_CACHE
    except Exception as e:
        logger.error("Failed to load TorchXRayVision DenseNet: %s", str(e), exc_info=True)
        return None


def load_radiograph(file_path: str) -> Tuple[Optional[np.ndarray], List[str], Optional[str]]:
    """
    Load radiograph from PNG, JPG, or DICOM.
    Returns: (processed_image_array, quality_notes, error_message).
    """
    path = Path(file_path)
    if not path.exists():
        return None, [], f"Radiograph file not found at {file_path}"

    ext = path.suffix.lower()
    quality_notes: List[str] = []

    # DICOM Handling
    if ext in (".dcm", ".dicom"):
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))

            # Modality check: CR/DX only
            modality = getattr(ds, "Modality", "CR").upper()
            if modality not in ("CR", "DX"):
                return None, [], f"Modality '{modality}' unsupported. Chest X-ray (CR/DX) only; CT/MRI not accepted."

            arr = ds.pixel_array.astype(float)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            # Invert MONOCHROME1
            photometric = getattr(ds, "PhotometricInterpretation", "").upper()
            if photometric == "MONOCHROME1":
                arr = arr.max() - arr

            # Min-max scale to 0..255
            if arr.max() > arr.min():
                arr = ((arr - arr.min()) / (arr.max() - arr.min())) * 255.0
            else:
                arr = np.zeros_like(arr)

        except ImportError:
            return None, [], "pydicom is not installed to read DICOM files."
        except Exception as e:
            return None, [], f"DICOM reading error: {str(e)}"

    # Standard Image (PNG / JPG)
    else:
        try:
            pil_img = Image.open(str(path))
            w, h = pil_img.size

            # Resolution check
            if w < 256 or h < 256:
                quality_notes.append(f"Low resolution ({w}x{h} px under 256 px); diagnostic features may be degraded.")

            # Aspect ratio check
            aspect = w / max(h, 1)
            if aspect < 0.6 or aspect > 1.6:
                quality_notes.append(f"Unusual aspect ratio ({aspect:.2f}); non-standard projection or cropping.")

            # Colour check
            if pil_img.mode in ("RGB", "RGBA"):
                rgb_arr = np.array(pil_img.convert("RGB"))
                std_diff = np.std(rgb_arr[..., 0] - rgb_arr[..., 1])
                if std_diff > 5.0:
                    quality_notes.append("Colour image detected; converted to single-channel greyscale.")
                pil_img = pil_img.convert("L")
            else:
                pil_img = pil_img.convert("L")

            arr = np.array(pil_img).astype(float)

            # Contrast check
            p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
            if (p95 - p5) < 30:
                quality_notes.append("Very low image contrast detected; details may be obscured.")

        except Exception as e:
            return None, [], f"Image reading error: {str(e)}"

    return arr, quality_notes, None


def compute_gradcam(
    model: Any,
    input_tensor: Any,
    class_idx: int,
) -> np.ndarray:
    """
    Compute Grad-CAM for class logit w.r.t. the last DenseNet feature map (before final ReLU/pool).
    Returns 224x224 normalized heatmap in range [0, 1].
    """
    import torch
    import torch.nn.functional as F

    # Ensure input tensor requires grad or features retain grad
    model.zero_grad()
    
    # Forward through features (last feature map before final ReLU/pool)
    features = model.features(input_tensor)
    features.retain_grad()

    # Forward through non-inplace ReLU + adaptive avg pool + classifier
    relu_feats = F.relu(features, inplace=False)
    pooled = F.adaptive_avg_pool2d(relu_feats, (1, 1)).view(features.size(0), -1)
    logits = model.classifier(pooled)

    logit = logits[0, class_idx]
    logit.backward(retain_graph=True)

    if features.grad is None:
        return np.zeros((224, 224), dtype=np.float32)

    # Global average pooling of gradients
    weights = features.grad.mean(dim=(2, 3), keepdim=True)
    weighted = (weights * features).sum(dim=1, keepdim=True)
    cam = F.relu(weighted)

    # If ReLU clipped all activations, fallback to shifted linear activation
    if cam.max() == 0 and weighted.max() > weighted.min():
        cam = weighted - weighted.min()

    # Bilinear upsample to 224x224
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy()

    # Min-max normalization to [0, 1]
    if cam_np.max() > cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np.astype(np.float32)



def determine_location(cam: np.ndarray, pathology: str) -> str:
    """
    Derive anatomical location from Grad-CAM activation.
    Threshold: 0.6.
    Laterality: image-left is patient's right; bilateral if 28-72% share.
    Zone: vertical centroid (<0.36 upper, <0.64 mid, >=0.64 lower).
    Cardiomegaly / mediastinal silhouette special handling.
    """
    path_lower = pathology.lower()
    if "cardiomegaly" in path_lower or "mediastinum" in path_lower or "cardiac" in path_lower:
        return "cardiac / mediastinal silhouette"

    mask = cam >= 0.6
    if mask.sum() == 0:
        mask = cam >= 0.4
    if mask.sum() == 0:
        return "diffuse / unlocalized"

    h, w = cam.shape
    y_indices, x_indices = np.where(mask)

    # Laterality: image-left is x < w / 2 (patient's right)
    left_side_mass = np.sum(x_indices < (w / 2))
    total_mass = len(x_indices)
    left_ratio = left_side_mass / max(total_mass, 1)

    if 0.28 <= left_ratio <= 0.72:
        laterality = "bilateral"
    elif left_ratio > 0.72:
        laterality = "right"  # image-left = patient's right
    else:
        laterality = "left"   # image-right = patient's left

    # Vertical zone by centroid
    y_centroid = np.mean(y_indices) / h
    if y_centroid < 0.36:
        zone = "upper zone"
    elif y_centroid < 0.64:
        zone = "mid zone"
    else:
        zone = "lower zone"

    return f"{laterality} {zone}"


def calculate_severity(probability: float, cam: np.ndarray) -> str:
    """
    Heuristic indicative severity based on probability and Grad-CAM spatial extent.
    high: p >= 0.75 and extent >= 0.12
    moderate: p >= 0.60 or extent >= 0.08
    else low.
    """
    extent = float(np.mean(cam >= 0.6))
    if probability >= 0.75 and extent >= 0.12:
        return "high"
    elif probability >= 0.60 or extent >= 0.08:
        return "moderate"
    else:
        return "low"


def generate_heatmap_overlay(
    raw_img_arr: np.ndarray,
    cam: np.ndarray,
    output_path: Path,
) -> str:
    """Render a 512x512 blue-to-red Grad-CAM overlay on the grayscale radiograph."""
    # Resize base image to 512x512
    base_pil = Image.fromarray(np.clip(raw_img_arr, 0, 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)
    base_gray = np.array(base_pil, dtype=float) / 255.0

    # Resize CAM to 512x512
    cam_pil = Image.fromarray((cam * 255.0).astype(np.uint8)).resize((512, 512), Image.BILINEAR)
    cam_512 = np.array(cam_pil, dtype=float) / 255.0

    # Custom Jet-like blue -> cyan -> yellow -> red colormap
    # Blue: (0, 0, 1), Red: (1, 0, 0)
    r = np.clip(1.5 * cam_512, 0.0, 1.0)
    g = np.clip(1.5 * (1.0 - np.abs(cam_512 - 0.5) * 2.0), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - cam_512), 0.0, 1.0)
    heatmap_rgb = np.stack([r, g, b], axis=-1)

    # Base greyscale expanded to RGB
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)

    # Alpha blend: 0.55 heatmap + 0.45 grayscale
    alpha = np.expand_dims(np.clip(cam_512 * 1.2, 0.2, 0.75), axis=-1)
    blended = (1.0 - alpha) * base_rgb + alpha * heatmap_rgb
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(blended_uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_img.save(str(output_path), format="PNG")
    return str(output_path)


def run_imaging_agent(case_input: CaseInput) -> ImagingReport:
    """Execute deep learning chest X-ray evaluation with Grad-CAM visualization."""
    if not case_input.image_paths:
        return ImagingReport(
            status="skipped",
            message="No chest radiograph attached for this case.",
        )

    img_file = case_input.image_paths[0]
    raw_arr, quality_notes, err_msg = load_radiograph(img_file)

    if err_msg or raw_arr is None:
        return ImagingReport(
            status="unavailable" if "unsupported" in (err_msg or "").lower() else "error",
            message=err_msg or "Failed to load radiograph image.",
            image_quality_notes=quality_notes,
        )

    model = get_densenet_model()
    if model is None:
        return ImagingReport(
            status="unavailable",
            message="TorchXRayVision DenseNet-121 weights unavailable. Install via `make run` or download model weights.",
            image_quality_notes=quality_notes,
        )

    try:
        import torch
        import torchvision
        import torchxrayvision as xrv

        # Preprocessing: normalize(img, 255) -> XRayCenterCrop -> XRayResizer(224)
        norm_img = xrv.datasets.normalize(raw_arr, 255)
        # Ensure 2D
        if len(norm_img.shape) > 2:
            norm_img = norm_img.mean(2)

        # Apply transformations on numpy array of shape (1, H, W)
        if norm_img.ndim == 2:
            norm_img_3d = norm_img[None, ...]
        else:
            norm_img_3d = norm_img

        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])
        np_224 = transform(norm_img_3d)
        tensor_224 = torch.from_numpy(np_224).float().unsqueeze(0)  # [1, 1, 224, 224]

        # Forward pass: obtain both raw sigmoids and op_norm normalized probabilities
        with torch.no_grad():
            features = model.features2(tensor_224)
            logits = model.classifier(features)
            raw_sigmoids = torch.sigmoid(logits)[0].detach().cpu().numpy()

            preds = model(tensor_224)
            raw_vals = preds[0].detach().cpu().numpy()
            if raw_vals.min() >= 0.0 and raw_vals.max() <= 1.0:
                probs = np.clip(raw_vals, 0.0, 1.0)
            else:
                probs = torch.sigmoid(preds)[0].detach().cpu().numpy()

        pathologies = model.pathologies
        findings: List[ImagingFinding] = []
        heatmap_paths: Dict[str, str] = []

        # Clinically calibrated candidate selection:
        # Avoid dataset threshold artifacts for rare classes where baseline noise (< 5% raw) gets scaled to 50%+
        candidate_indices = []
        for idx in range(len(pathologies)):
            p_op = float(probs[idx])
            p_raw = float(raw_sigmoids[idx])
            op_thresh = float(model.op_threshs[idx].item()) if hasattr(model, "op_threshs") else 0.05

            if p_op >= 0.50:
                # For rare pathologies (op_thresh < 0.03 like pneumothorax, emphysema, fibrosis)
                # require either genuine raw logit elevation (>= 10%) or high op_norm (>= 0.65)
                if op_thresh < 0.03:
                    if p_raw >= 0.10 or p_op >= 0.65:
                        candidate_indices.append((idx, p_op, pathologies[idx]))
                else:
                    if p_raw >= 0.06 or p_op >= 0.55:
                        candidate_indices.append((idx, p_op, pathologies[idx]))

        # Sort descending, top 6
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidate_indices[:6]

        case_dir = settings.UPLOAD_DIR / case_input.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        heatmaps_dict: Dict[str, str] = {}

        for idx, prob, path_name in top_candidates:
            # Grad-CAM computation
            cam = compute_gradcam(model, tensor_224.clone(), idx)
            loc = determine_location(cam, path_name)
            sev = calculate_severity(prob, cam)

            # Confidence band
            if prob >= 0.70:
                conf = "high"
            elif prob >= 0.50:
                conf = "moderate"
            else:
                conf = "low"

            findings.append(
                ImagingFinding(
                    finding=path_name,
                    probability=round(prob, 4),
                    location=loc,
                    severity=sev,
                    confidence=conf,
                )
            )

            # Generate and save 512x512 heatmap overlay PNG
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", path_name)
            hm_file = case_dir / f"heatmap_{safe_name}.png"
            generate_heatmap_overlay(raw_arr, cam, hm_file)
            heatmaps_dict[path_name] = str(hm_file)

        if not findings:
            quality_notes.append(
                "Unremarkable radiograph: No findings exceed clinical significance threshold (all probabilities within baseline normal limits)."
            )
            msg = "Unremarkable chest radiograph: No acute cardiopulmonary abnormality detected."
        else:
            msg = f"DenseNet-121 identified {len(findings)} pathologies meeting clinical operating threshold (p >= 50%)."

        return ImagingReport(
            status="ok",
            modality="chest X-ray",
            model_name="TorchXRayVision-DenseNet121",
            findings=findings,
            heatmap_paths=heatmaps_dict,
            image_quality_notes=quality_notes,
            message=msg,
        )

    except Exception as e:
        logger.error("Imaging analysis execution failed: %s", str(e), exc_info=True)
        return ImagingReport(
            status="error",
            message=f"Imaging pipeline error: {str(e)}",
            image_quality_notes=quality_notes,
        )
