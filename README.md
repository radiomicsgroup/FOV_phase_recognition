# FOV_phase_recognition
Pipeline to identify the anatomical Field-of-View (FOV) for 3D medical images and contrast phase for CT images.

**Overview**
- **Purpose:** 
    - Detect anatomical FOV from 3D NIfTI images and save per-image results: thorax, abdomen, thorax_abdomen, whole_body, scout, unknown, pelvis (only MRI) or spine (only MRI).
    - Detect the contrast phase from 3D CT NIfTI images and save per-image results: native, portal_venus, arterial or delay.

**Requirements**
- **Python:** 3.8+ recommended.
- **Core Python packages:** `pandas`, `tqdm`, `nibabel`, `numpy`.
- **Segmentation backend:** `totalsegmentator` is required. Follow totalsegmentator installation instructions (may require PyTorch and appropriate CUDA drivers for GPU acceleration).

**Usage**
- Basic run (processes both FOV and phase by default):

```bash
python main.py path/to/input.csv MRI
```

- Arguments:
	- `input_file`: path to the input CSV (must contain `im_path` column).
	- `im_type`: either `MRI` or `CT` (affects which segmentation model and rules are used).

- Optional flags:
	- `--output`: directory path where segmentation masks will be saved. If not provided, segmentation masks are not saved.
	- `--no_fov`: skip FOV recognition.
	- `--no_phase`: skip phase recognition.

- Example with output directory:

```bash
python main.py tests/1001prostate_MRI_T2_first100.csv MRI --output outputs/
```

**Output**
- A CSV file is written/appended in the same folder as the input CSV with the suffix `_fov_phase_processed.csv`. Columns include `im_path` and, unless disabled, `fov` and `phase`.
- For each processed image, the pipeline also writes small text files next to the image path named `<image>_fov.txt` and `<image>_phase.txt` containing the detected label.

**Notes**
- The script will skip images already present in an existing output CSV and will resume processing remaining entries.
- Small 3D volumes with fewer than ~10 non-blank slices are labeled `scout` by the FOV routine.
