from __future__ import annotations

import base64
import io
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pydicom
import requests
from PIL import Image, ImageDraw
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, vuetify

try:
    import nibabel as nib
except Exception:
    nib = None

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller
state.trame__title = "Atlas Slicer Web"

RUNTIME = {
    "volumes": {},  # study_id -> np.ndarray [z, h, w], uint8
    "overlays": {},  # study_id -> np.ndarray [z, h, w], uint8 {0,1}
    "upload_dirs": [],  # temp folders created from browser uploads
}


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def png_data_uri(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def add_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.log_lines = (state.log_lines + [f"[{timestamp}] {message}"])[-20:]
    state.status_log = message


def parse_hex_color(color: str) -> tuple[int, int, int]:
    clean = (color or "#ff5c70").strip().lstrip("#")
    if len(clean) != 6:
        return (255, 92, 112)
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    low, high = np.percentile(arr_f, [1, 99])
    if high <= low:
        low, high = float(arr_f.min()), float(arr_f.max() or 1.0)
    arr_f = np.clip(arr_f, low, high)
    arr_f = (arr_f - low) / max(high - low, 1e-6)
    return np.uint8(arr_f * 255.0)


def apply_window(slice_u8: np.ndarray, window_level: int) -> np.ndarray:
    # window_level from 0..100 controls contrast around mid-gray
    contrast = 0.55 + (window_level / 100.0) * 1.45
    centered = (slice_u8.astype(np.float32) - 127.0) * contrast + 127.0
    return np.uint8(np.clip(centered, 0, 255))


def apply_zoom(image: np.ndarray, zoom_level: int) -> np.ndarray:
    if zoom_level == 100:
        return image
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    if zoom_level > 100:
        factor = zoom_level / 100.0
        crop_w = int(w / factor)
        crop_h = int(h / factor)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        cropped = pil.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        return np.array(cropped.resize((w, h), Image.Resampling.BICUBIC))

    factor = zoom_level / 100.0
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    scaled = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("L" if image.ndim == 2 else "RGB", (w, h), 0)
    canvas.paste(scaled, ((w - new_w) // 2, (h - new_h) // 2))
    return np.array(canvas)


def render_slice_image(
    slice_u8: np.ndarray,
    mask_u8: np.ndarray | None,
    model_color: str,
    window_level: int,
    zoom_level: int,
) -> Image.Image:
    base = apply_window(slice_u8, window_level)
    base = apply_zoom(base, zoom_level)
    rgb = np.stack([base, base, base], axis=-1)
    if mask_u8 is not None:
        mask = apply_zoom(mask_u8 * 255, zoom_level) > 0
        r, g, b = parse_hex_color(model_color)
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = r
        overlay[..., 1] = g
        overlay[..., 2] = b
        alpha = 0.45
        rgb = np.where(mask[..., None], (rgb * (1 - alpha) + overlay * alpha).astype(np.uint8), rgb)
    return Image.fromarray(rgb, mode="RGB")


def render_volume_image(volume: np.ndarray, overlay: np.ndarray | None, model_color: str) -> Image.Image:
    projection = np.max(volume, axis=0)
    projection = normalize_to_uint8(projection)
    proj_rgb = np.stack([projection, projection, projection], axis=-1)
    if overlay is not None and overlay.size:
        overlay_proj = np.max(overlay, axis=0) > 0
        r, g, b = parse_hex_color(model_color)
        alpha = 0.4
        proj_rgb[overlay_proj] = (
            proj_rgb[overlay_proj] * (1 - alpha) + np.array([r, g, b]) * alpha
        ).astype(np.uint8)

    card = Image.new("RGB", (960, 420), (8, 17, 29))
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, 960, 420), fill=(10, 19, 34))

    panel = Image.fromarray(proj_rgb, mode="RGB").resize((460, 340), Image.Resampling.BICUBIC)
    card.paste(panel, (260, 40))
    draw.rectangle((260, 40, 720, 380), outline=(130, 160, 220), width=2)
    draw.text((30, 30), "Volume rendering (MIP)", fill=(235, 245, 255))
    draw.text((30, 60), f"Volume size: {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}", fill=(165, 185, 215))
    return card


def fake_segmentation(slice_u8: np.ndarray, modality: str) -> np.ndarray:
    threshold = 155 if modality.upper() == "CT" else 140
    return (slice_u8 > threshold).astype(np.uint8)


def decode_mask_base64(mask_b64: str, expected_shape: tuple[int, int]) -> np.ndarray | None:
    try:
        decoded = base64.b64decode(mask_b64)
        image = Image.open(io.BytesIO(decoded)).convert("L")
        arr = np.array(image)
        if arr.shape != expected_shape:
            image = image.resize((expected_shape[1], expected_shape[0]), Image.Resampling.NEAREST)
            arr = np.array(image)
        return (arr > 127).astype(np.uint8)
    except Exception:
        return None


def infer_remote_mask(model: dict, slice_u8: np.ndarray, modality: str) -> np.ndarray | None:
    endpoint = (model.get("endpoint") or "").strip()
    if not endpoint or not endpoint.startswith(("http://", "https://")):
        return None

    try:
        image_uri = png_data_uri(Image.fromarray(slice_u8, mode="L"))
        payload = {
            "image": image_uri,
            "modality": modality,
            "classes": model.get("classes", []),
            "model_name": model.get("name", "custom"),
        }
        response = requests.post(endpoint, json=payload, timeout=float(state.inference_timeout_s))
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        add_log(f"Inference endpoint failed: {exc}")
        return None

    if isinstance(body, dict):
        if isinstance(body.get("mask"), list):
            arr = np.array(body["mask"], dtype=np.uint8)
            if arr.shape == slice_u8.shape:
                return (arr > 0).astype(np.uint8)
        if isinstance(body.get("mask_base64"), str):
            mask = decode_mask_base64(body["mask_base64"], slice_u8.shape)
            if mask is not None:
                return mask
    return None


def dicom_sort_key(ds: pydicom.Dataset, fallback_index: int) -> tuple[float, float]:
    instance = float(getattr(ds, "InstanceNumber", fallback_index))
    ipp = getattr(ds, "ImagePositionPatient", None)
    z_pos = float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 else instance
    return (z_pos, instance)


def load_dicom_volume(path_value: str) -> tuple[dict, np.ndarray]:
    source = Path(path_value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {source}")

    if source.is_file():
        files = [source]
    else:
        files = sorted([p for p in source.rglob("*") if p.is_file()])

    dicom_rows: list[tuple[tuple[float, float], np.ndarray, pydicom.Dataset]] = []
    for idx, file_path in enumerate(files):
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            pixels = ds.pixel_array
            if pixels.ndim == 2:
                dicom_rows.append((dicom_sort_key(ds, idx), pixels, ds))
            elif pixels.ndim == 3:
                for frame_idx in range(pixels.shape[0]):
                    key = dicom_sort_key(ds, idx + frame_idx / 1000.0)
                    dicom_rows.append((key, pixels[frame_idx], ds))
        except Exception:
            continue

    if not dicom_rows:
        raise RuntimeError("No readable DICOM pixel data was found at this path.")

    dicom_rows.sort(key=lambda item: item[0])
    first_ds = dicom_rows[0][2]
    modality = str(getattr(first_ds, "Modality", "Unknown"))
    title = str(getattr(first_ds, "StudyDescription", source.name or "Imported Study"))

    volume_slices: list[np.ndarray] = []
    for _, arr, ds in dicom_rows:
        arr_f = arr.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr_f = arr_f * slope + intercept
        volume_slices.append(normalize_to_uint8(arr_f))

    volume = np.stack(volume_slices, axis=0)
    study_id = f"study-{int(datetime.now().timestamp())}"
    frames = [{"index": i, "title": f"Slice {i + 1}"} for i in range(volume.shape[0])]
    study = {"id": study_id, "title": title, "modality": modality, "frames": frames}
    return study, volume


def selected_study() -> dict | None:
    return next((study for study in state.studies if study["id"] == state.selected_study_id), None)


def selected_model() -> dict | None:
    return next((model for model in state.models if model["id"] == state.selected_model_id), None)


def refresh_catalog() -> None:
    state.study_items = [{"label": s["title"], "value": s["id"]} for s in state.studies]
    state.model_items = [{"label": m["name"], "value": m["id"]} for m in state.models]
    state.study_count = len(state.studies)
    state.model_count = len(state.models)


def update_viewer(*_args, **_kwargs) -> None:
    study = selected_study()
    model = selected_model()
    if not study:
        empty_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#08111d"/>
  <text x="30" y="45" fill="white" font-size="24" font-family="Segoe UI, Arial">Atlas Slicer Web</text>
  <text x="30" y="80" fill="#9fb2cb" font-size="14" font-family="Segoe UI, Arial">Load a DICOM path to begin</text>
</svg>
"""
        state.viewer_src = svg_data_uri(empty_svg.strip())
        state.volume_src = state.viewer_src
        state.status_viewport = "Idle"
        state.status_inference = "Waiting for study"
        state.status_modality = "Unknown"
        state.status_overlay = "Disabled"
        state.slice_max = 1
        return

    volume = RUNTIME["volumes"].get(study["id"])
    overlay_volume = RUNTIME["overlays"].get(study["id"])
    if volume is None:
        return

    state.slice_max = int(volume.shape[0])
    state.slice_number = max(1, min(int(state.slice_number), int(volume.shape[0])))
    z = state.slice_number - 1
    current_slice = volume[z]
    current_mask = None
    if state.overlay_enabled and overlay_volume is not None and z < overlay_volume.shape[0]:
        current_mask = overlay_volume[z]

    model_color = model["color"] if model else "#ff5c70"
    slice_img = render_slice_image(
        current_slice,
        current_mask,
        model_color,
        int(state.window_level),
        int(state.zoom_level),
    )
    volume_img = render_volume_image(volume, overlay_volume if state.overlay_enabled else None, model_color)

    state.viewer_src = png_data_uri(slice_img)
    state.volume_src = png_data_uri(volume_img)
    state.status_viewport = f"Slice {state.slice_number}/{volume.shape[0]}"
    state.status_modality = study["modality"]
    state.status_overlay = "Enabled" if state.overlay_enabled else "Disabled"
    state.status_inference = f"{model['name']} selected" if model else "No model selected"


def register_default_models() -> None:
    state.models = [
        {
            "id": "brain-tumor",
            "name": "Brain Tumor Segmenter",
            "modality": "MR",
            "endpoint": "",
            "classes": ["tumor", "edema", "necrosis"],
            "color": "#4d7cff",
            "kind": "Local",
        },
        {
            "id": "liver-lesion",
            "name": "Liver Lesion Segmenter",
            "modality": "CT",
            "endpoint": "",
            "classes": ["liver", "lesion"],
            "color": "#19c6a5",
            "kind": "Local",
        },
    ]
    state.selected_model_id = state.models[0]["id"]


def load_demo() -> None:
    z = 20
    y = 256
    x = 256
    yy, xx = np.mgrid[0:y, 0:x]
    volume = np.zeros((z, y, x), dtype=np.uint8)
    for i in range(z):
        r = 40 + i * 4
        cx = 128 + int(np.sin(i / 3.5) * 14)
        cy = 128 + int(np.cos(i / 4.0) * 10)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        volume[i][mask] = np.clip(140 + i * 4, 0, 255)
        volume[i] = np.maximum(volume[i], np.uint8((xx + yy) / 4))

    study_id = f"demo-{int(datetime.now().timestamp())}"
    study = {
        "id": study_id,
        "title": "Demo Brain MRI",
        "modality": "MR",
        "frames": [{"index": i, "title": f"Slice {i + 1}"} for i in range(z)],
    }
    RUNTIME["volumes"][study_id] = volume
    RUNTIME["overlays"][study_id] = np.zeros_like(volume, dtype=np.uint8)
    state.studies = [study]
    state.selected_study_id = study_id
    state.slice_number = 1
    state.overlay_enabled = True
    refresh_catalog()
    update_viewer()
    add_log("Demo volume loaded.")


def load_study_from_path() -> None:
    path_value = (state.new_study_path or "").strip()
    if not path_value:
        add_log("Please enter a local DICOM folder or file path.")
        return

    try:
        study, volume = load_dicom_volume(path_value)
    except Exception as exc:
        add_log(f"Study load failed: {exc}")
        return

    RUNTIME["volumes"][study["id"]] = volume
    RUNTIME["overlays"][study["id"]] = np.zeros_like(volume, dtype=np.uint8)
    state.studies = [study, *state.studies]
    state.selected_study_id = study["id"]
    state.slice_number = 1
    refresh_catalog()
    update_viewer()
    add_log(f"Loaded study '{study['title']}' with {volume.shape[0]} slices.")


def _decode_data_url(content: str) -> bytes:
    if "," in content and content.startswith("data:"):
        return base64.b64decode(content.split(",", 1)[1])
    return base64.b64decode(content)


def _prepare_upload_dir(upload_payload: list[dict]) -> Path:
    upload_dir = Path(tempfile.mkdtemp(prefix="atlas_upload_"))
    for item in upload_payload:
        name = Path(str(item.get("name", "file.bin"))).name
        content = item.get("content")
        if not isinstance(content, str):
            continue
        data = _decode_data_url(content)
        (upload_dir / name).write_bytes(data)
    RUNTIME["upload_dirs"].append(str(upload_dir))
    return upload_dir


def load_study_from_upload() -> None:
    files = state.uploaded_files or []
    if not isinstance(files, list) or not files:
        add_log("Upload import skipped: choose one or more files first.")
        return

    if not isinstance(files[0], dict):
        add_log("Upload payload format not supported by this browser/session.")
        return

    try:
        upload_dir = _prepare_upload_dir(files)
    except Exception as exc:
        add_log(f"Upload decode failed: {exc}")
        return

    # If user uploaded a zip, extract and use extracted folder.
    zip_files = list(upload_dir.glob("*.zip"))
    source_path = upload_dir
    if zip_files:
        extract_dir = upload_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                zf.extractall(extract_dir)
            source_path = extract_dir
        except Exception as exc:
            add_log(f"ZIP extraction failed: {exc}")
            return

    try:
        study, volume = load_dicom_volume(str(source_path))
    except Exception as exc:
        add_log(f"Upload import failed: {exc}")
        return

    RUNTIME["volumes"][study["id"]] = volume
    RUNTIME["overlays"][study["id"]] = np.zeros_like(volume, dtype=np.uint8)
    state.studies = [study, *state.studies]
    state.selected_study_id = study["id"]
    state.slice_number = 1
    refresh_catalog()
    update_viewer()
    add_log(f"Uploaded study '{study['title']}' with {volume.shape[0]} slices.")


def register_model() -> None:
    name = (state.new_model_name or "").strip()
    if not name:
        add_log("Model registration skipped: name is required.")
        return

    classes = [c.strip() for c in (state.new_model_classes or "").split(",") if c.strip()]
    model = {
        "id": f"model-{int(datetime.now().timestamp())}",
        "name": name,
        "modality": (state.new_model_modality or "Mixed").strip() or "Mixed",
        "endpoint": (state.new_model_endpoint or "").strip(),
        "classes": classes,
        "color": (state.new_model_color or "#ff5c70").strip(),
        "kind": "Custom",
    }
    state.models = [model, *state.models]
    state.selected_model_id = model["id"]
    state.new_model_name = ""
    state.new_model_modality = ""
    state.new_model_endpoint = ""
    state.new_model_classes = ""
    refresh_catalog()
    update_viewer()
    add_log(f"Registered model '{model['name']}'.")


def run_segmentation() -> None:
    study = selected_study()
    model = selected_model()
    if not study or not model:
        add_log("Segmentation skipped: select a study and model first.")
        return

    volume = RUNTIME["volumes"].get(study["id"])
    if volume is None:
        add_log("Segmentation skipped: study volume is missing in runtime cache.")
        return

    z = state.slice_number - 1
    slice_u8 = volume[z]
    remote_mask = infer_remote_mask(model, slice_u8, study["modality"])
    mask = remote_mask if remote_mask is not None else fake_segmentation(slice_u8, study["modality"])

    overlay_volume = RUNTIME["overlays"].get(study["id"])
    if overlay_volume is None or overlay_volume.shape != volume.shape:
        overlay_volume = np.zeros_like(volume, dtype=np.uint8)
    overlay_volume[z] = mask
    RUNTIME["overlays"][study["id"]] = overlay_volume

    state.overlay_enabled = True
    update_viewer()
    source = "remote endpoint" if remote_mask is not None else "local fallback"
    add_log(f"Segmentation complete on slice {state.slice_number} using {source}.")


def export_masks() -> None:
    study = selected_study()
    if not study:
        add_log("Export skipped: no active study.")
        return

    overlay = RUNTIME["overlays"].get(study["id"])
    if overlay is None:
        add_log("Export skipped: no overlay volume available.")
        return

    export_dir = Path((state.export_dir or ".").strip()).expanduser().resolve()
    export_name = (state.export_name or "segmentation").strip()
    if not export_name:
        export_name = "segmentation"
    export_dir.mkdir(parents=True, exist_ok=True)

    npy_path = export_dir / f"{export_name}.npy"
    np.save(str(npy_path), overlay.astype(np.uint8))

    if nib is not None:
        # Internal storage is [z, y, x]. NIfTI is typically saved as [x, y, z].
        nii_data = np.transpose(overlay.astype(np.uint8), (2, 1, 0))
        nii_image = nib.Nifti1Image(nii_data, np.eye(4))
        nii_path = export_dir / f"{export_name}.nii.gz"
        nib.save(nii_image, str(nii_path))
        state.status_export = f"Saved: {nii_path.name}, {npy_path.name}"
    else:
        state.status_export = f"Saved: {npy_path.name} (nibabel unavailable for NIfTI)"
    add_log(f"Exported masks to {export_dir}")


@state.change("selected_study_id", "selected_model_id", "slice_number", "window_level", "zoom_level", "overlay_enabled")
def on_controls_change(**_kwargs) -> None:
    update_viewer()


def build_ui() -> None:
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Atlas Slicer Web")

        html.Style(
            """
            :root {
              color-scheme: dark;
              --panel: rgba(14, 25, 41, 0.92);
              --line: rgba(147, 184, 255, 0.14);
              --text: #edf5ff;
              --muted: #9fb2cb;
              --shadow: 0 24px 72px rgba(0, 0, 0, 0.38);
            }
            .app-shell { display:grid; gap:18px; max-width:1680px; margin:0 auto; width:100%; padding:18px; }
            .app-bar {
              display:flex; justify-content:space-between; gap:18px; align-items:flex-end; padding:20px 22px;
              border:1px solid var(--line); border-radius:26px; background:linear-gradient(180deg, rgba(18, 29, 46, 0.96), rgba(13, 22, 37, 0.92)); box-shadow: var(--shadow);
            }
            .eyebrow { margin:0 0 6px; color:#89aaff; text-transform:uppercase; letter-spacing:0.15em; font-size:0.74rem; font-weight:700; }
            .lede { margin: 12px 0 0; max-width: 760px; color: var(--muted); line-height: 1.6; }
            .workspace-grid { display:grid; grid-template-columns:300px minmax(0,1fr) 330px; gap:18px; align-items:start; }
            .panel, .viewer-panel, .info-panel { background:var(--panel); border:1px solid var(--line); border-radius:26px; box-shadow:var(--shadow); padding:16px; }
            .panel-section { display:grid; gap:12px; margin-bottom:14px; }
            .section-head, .viewer-head { display:flex; justify-content:space-between; align-items:center; gap:10px; }
            .status-row, .scene-row {
              display:flex; justify-content:space-between; gap:12px; align-items:center;
              padding:10px 12px; border-radius:12px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.05);
            }
            .canvas-shell {
              position:relative; overflow:hidden; border-radius:22px; min-height:420px;
              background:radial-gradient(circle at 50% 20%, rgba(77, 124, 255, 0.24), transparent 30%), linear-gradient(145deg, #121d30 0%, #070d16 100%);
              border:1px solid rgba(255,255,255,0.05);
            }
            .canvas-shell img { width:100%; height:100%; display:block; object-fit:contain; }
            .control-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }
            .bottom-grid { display:grid; grid-template-columns:1fr; gap:12px; }
            .summary-box { white-space:pre-wrap; margin:0; padding:14px; border-radius:18px; background:#07111d; color:#d8e5ff; }
            .log-panel { min-height:190px; max-height:260px; overflow:auto; }
            @media (max-width: 1320px) { .workspace-grid { grid-template-columns:1fr; } }
            @media (max-width: 900px) { .app-bar, .workspace-grid, .control-grid { display:grid; grid-template-columns:1fr; } }
            """
        )

        with layout.toolbar:
            vuetify.VBtn("Load Demo", click=ctrl.load_demo, color="primary", variant="flat")
            vuetify.VSpacer()
            vuetify.VSwitch(v_model=("overlay_enabled", True), label="Overlay", hide_details=True, density="compact")

        with layout.content:
            with html.Div(classes="app-shell"):
                with html.Div(classes="app-bar"):
                    with html.Div():
                        html.Div("trame-slicer inspired", classes="eyebrow")
                        html.H1("Atlas Slicer Web")
                        html.Div(
                            "Real DICOM loader, real volume preview, and pluggable segmentation backend hooks.",
                            classes="lede",
                        )
                    with html.Div(style="display:flex; gap: 10px; align-items:center; flex-wrap: wrap;"):
                        vuetify.VTextField(
                            v_model=("new_study_path", ""),
                            label="DICOM folder/file path",
                            density="compact",
                            hide_details=True,
                            style="min-width: 360px;",
                        )
                        vuetify.VFileInput(
                            v_model=("uploaded_files", []),
                            label="Upload DICOM files or ZIP",
                            multiple=True,
                            chips=True,
                            show_size=True,
                            accept=".dcm,.zip",
                            density="compact",
                            hide_details=True,
                            style="min-width: 320px;",
                        )
                        vuetify.VBtn("Load Path", click=ctrl.load_study_from_path, color="secondary", variant="flat")
                        vuetify.VBtn("Import Upload", click=ctrl.load_study_from_upload, color="secondary", variant="flat")
                        vuetify.VBtn("Run Segmentation", click=ctrl.run_segmentation, color="secondary", variant="flat")

                with html.Div(classes="workspace-grid"):
                    with vuetify.VCard(classes="panel"):
                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Data")
                                html.Span("{{ study_count }} studies")
                            with vuetify.VBtnToggle(v_model=("selected_study_id", ""), mandatory=True, density="compact", style="flex-wrap: wrap; gap: 8px;"):
                                with vuetify.VBtn(v_for="study in study_items", value=("study.value",), variant="outlined"):
                                    html.Span("{{ study.label }}")

                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Modules")
                                html.Span("{{ model_count }} models")
                            with vuetify.VBtnToggle(v_model=("selected_model_id", ""), mandatory=True, density="compact", style="flex-wrap: wrap; gap: 8px;"):
                                with vuetify.VBtn(v_for="model in model_items", value=("model.value",), variant="outlined"):
                                    html.Span("{{ model.label }}")

                    with html.Div():
                        with vuetify.VCard(classes="viewer-panel"):
                            with html.Div(classes="viewer-head"):
                                html.H2("2D Slice Viewer")
                                html.Span("{{ status_viewport }}")
                            with html.Div(classes="canvas-shell"):
                                html.Img(src=("viewer_src", ""), alt="Current slice")
                            with html.Div(classes="control-grid"):
                                vuetify.VSlider(label="Slice", v_model=("slice_number", 1), min=1, max=("slice_max", 1), step=1, hide_details=True, density="compact")
                                vuetify.VSlider(label="Window", v_model=("window_level", 55), min=0, max=100, step=1, hide_details=True, density="compact")
                                vuetify.VSlider(label="Zoom", v_model=("zoom_level", 100), min=70, max=180, step=1, hide_details=True, density="compact")

                        with vuetify.VCard(classes="viewer-panel", style="margin-top: 14px;"):
                            with html.Div(classes="viewer-head"):
                                html.H2("3D Volume Preview")
                                html.Span("{{ status_modality }}")
                            with html.Div(classes="canvas-shell"):
                                html.Img(src=("volume_src", ""), alt="Volume preview")

                    with vuetify.VCard(classes="panel"):
                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Add Model")
                            vuetify.VTextField(v_model=("new_model_name", ""), label="Name", density="compact", hide_details=True)
                            vuetify.VTextField(v_model=("new_model_modality", ""), label="Modality", density="compact", hide_details=True)
                            vuetify.VTextField(v_model=("new_model_endpoint", ""), label="Endpoint (http://... optional)", density="compact", hide_details=True)
                            vuetify.VTextField(v_model=("new_model_classes", ""), label="Classes (comma separated)", density="compact", hide_details=True)
                            vuetify.VTextField(v_model=("new_model_color", "#ff5c70"), label="Color (#RRGGBB)", density="compact", hide_details=True)
                            vuetify.VBtn("Register Module", click=ctrl.register_model, color="primary", variant="flat")

                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Inference")
                            vuetify.VTextField(v_model=("inference_timeout_s", "15"), label="Timeout (seconds)", density="compact", hide_details=True)
                            with html.Div(classes="status-row"):
                                html.Span("Status")
                                html.Strong("{{ status_inference }}")
                            with html.Div(classes="status-row"):
                                html.Span("Overlay")
                                html.Strong("{{ status_overlay }}")

                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Export")
                            vuetify.VTextField(v_model=("export_dir", "."), label="Export directory", density="compact", hide_details=True)
                            vuetify.VTextField(v_model=("export_name", "segmentation"), label="Export base name", density="compact", hide_details=True)
                            vuetify.VBtn("Export Masks (.nii.gz + .npy)", click=ctrl.export_masks, color="primary", variant="flat")
                            with html.Div(classes="status-row"):
                                html.Span("Last export")
                                html.Strong("{{ status_export }}")

                        with html.Div(classes="panel-section"):
                            with html.Div(classes="section-head"):
                                html.H2("Console")
                            with html.Div(classes="log-panel"):
                                html.Pre("{{ log_lines.join('\\n') || 'No events yet.' }}", classes="summary-box")


ctrl.load_demo = load_demo
ctrl.load_study_from_path = load_study_from_path
ctrl.load_study_from_upload = load_study_from_upload
ctrl.register_model = register_model
ctrl.run_segmentation = run_segmentation
ctrl.export_masks = export_masks

state.studies = []
state.models = []
state.study_items = []
state.model_items = []
state.study_count = 0
state.model_count = 0
state.selected_study_id = ""
state.selected_model_id = ""
state.new_study_path = ""
state.uploaded_files = []
state.new_model_name = ""
state.new_model_modality = ""
state.new_model_endpoint = ""
state.new_model_classes = ""
state.new_model_color = "#ff5c70"
state.inference_timeout_s = "15"
state.slice_number = 1
state.slice_max = 1
state.window_level = 55
state.zoom_level = 100
state.overlay_enabled = True
state.viewer_src = ""
state.volume_src = ""
state.status_viewport = "Idle"
state.status_inference = "Waiting for study"
state.status_modality = "Unknown"
state.status_overlay = "Disabled"
state.status_export = "Not exported"
state.status_log = "Ready"
state.log_lines = []
state.export_dir = "."
state.export_name = "segmentation"

register_default_models()
refresh_catalog()
load_demo()
build_ui()
update_viewer()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    server.start(host=host, port=port, open_browser=False)
