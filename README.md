# Coconut Defect Detection — Hybrid YOLO + Tensor (Notebook Only)

This repo is **one Jupyter Notebook (.ipynb)** that trains and tests a **Hybrid YOLO + Tensor (TTConv) setup** for coconut defect detection.

## Requirements
- Python 3.10+
- GPU recommended (Colab T4/P100 works)
- Install inside the notebook:
  - `ultralytics`, `torch`, `opencv-python`, `numpy`, `tqdm`

## How to Run (Local Jupyter)
1. Open the `.ipynb` in Jupyter/VSCode.
2. Put your dataset zip where the notebook expects it (same folder is easiest).
3. Run cells **top → bottom** (don’t skip the dataset-prep cells).
4. Training outputs + weights will be saved in the notebook’s `runs/` (Ultralytics default) or the output folder defined in the cells.

## How to Run (Google Colab)
1. Upload the `.ipynb` to Colab.
2. Upload the dataset `.zip` (or mount Drive).
3. Update the dataset path variable in the notebook to match Colab path, e.g.:
   - `/content/<your_zip>.zip` or `/content/drive/MyDrive/<your_zip>.zip`
4. Runtime → Change runtime type → **GPU**
5. Run all cells.

## Dataset Expectation (as used by the notebook)
- The notebook **extracts the zip** and creates the working dataset structure automatically.
- It generates a `dataset.yaml` for YOLO training inside the working directory it sets.
- Keep class folder names and dataset layout consistent with what the notebook cells assume.

## Training
- The notebook trains a YOLO model with Tensor/TTConv hybrid modifications as defined in the model cells.
- Key knobs (edit inside notebook):
  - `imgsz`, `epochs`, `batch`, `model/weights path`, `data.yaml path`

## Inference / Testing
- The notebook runs prediction on test images using the trained weights.
- Predictions are saved in the default Ultralytics output folder:
  - `runs/detect/predict*` (unless the notebook overrides it)

## Output Artifacts
- Trained weights:
  - `runs/detect/train*/weights/best.pt` and `last.pt`
- Logs + metrics:
  - `runs/detect/train*/results.csv` and plots
- Predictions:
  - `runs/detect/predict*/`

## Quick Troubleshooting
- **No GPU / slow** → use Colab GPU.
- **FileNotFoundError** → you didn’t update the zip path or the notebook renamed files during preprocessing.
- **Bad metrics** → check that your dataset labels/classes match what the notebook assumes.

## Notes
This project is intentionally notebook-only (hackathon-style). If you want it as a clean repo (scripts + modules), you’ll need to refactor the cells into `/models` + `/scripts`.
