# Atlas Slicer Web

`Atlas Slicer Web` is a trame-based DICOM viewer inspired by `trame-slicer`.
It now includes:

- real DICOM loading from a local folder/file path (`pydicom`)
- real slice rendering and volume preview from voxel data (`numpy` + `Pillow`)
- segmentation backend hooks for user-provided model endpoints (`requests`)
- local fallback segmentation when no endpoint is configured
- mask export to NIfTI (`.nii.gz`) and NumPy (`.npy`)

## Setup

```bash
pip install -r requirements.txt
python app.py
```

The app listens on `HOST`/`PORT` env vars for cloud deployment.

## Workflow

1. Start with **Load Demo** or paste a local DICOM path in **DICOM folder/file path**.
2. Click **Load Path** to ingest a real study.
3. Or upload `.dcm` files / a `.zip` and click **Import Upload** (recommended on deployed web).
4. Register custom models in the right panel.
5. Optionally set model endpoint to `http://...` or `https://...`.
6. Click **Run Segmentation**.
7. Use **Export Masks** to save the current overlay volume.

## Segmentation API contract

The app sends:

```json
{
  "image": "data:image/png;base64,...",
  "modality": "CT",
  "classes": ["liver", "lesion"],
  "model_name": "Liver Segmenter"
}
```

Accepted response formats:

```json
{ "mask": [[0,1,1,...], ...] }
```

or

```json
{ "mask_base64": "iVBORw0..." }
```

`mask_base64` should decode to a grayscale mask image.

## Notes

- DICOM sorting uses `ImagePositionPatient`/`InstanceNumber` when present.
- Overlay is stored in server memory per loaded study.
- For production, add authentication, study persistence, and full 3D interactivity.

## Mask export

- `Export directory`: output folder
- `Export base name`: filename prefix
- Output files:
- `<name>.npy` (raw mask array in `[z, y, x]`)
- `<name>.nii.gz` (NIfTI mask volume when `nibabel` is available)

## Deploy (Render example)

1. Push this project to GitHub.
2. In Render, create a new **Web Service** from the repo.
3. Render will detect `render.yaml` (or use `build: pip install -r requirements.txt`, `start: python app.py`).
4. Deploy and open the generated URL.
