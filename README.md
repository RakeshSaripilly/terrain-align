# LunarX: Lunar Image Correspondence

Lunar image registration pipeline for matching the same terrain under different
Sun angles and sensor conditions. The current implementation uses multi-scale
Local Contrast Normalization (LCN), LoFTR feature matching, spatially uniform
sampling, subpixel Lucas-Kanade refinement, and USAC MAGSAC homography fitting.

The pipeline is designed for LRO NAC and Chandrayaan-2 imagery such as OHRC,
TMC, and IIRS products.

## Project Structure

The main implementation is inside `SunAngle/`:

```text
SunAngle/
├── Run-After-Lro.py       # Command-line entry point
├── web_app.py             # Interactive Streamlit upload interface
├── pipeline.py            # End-to-end orchestration
├── Task2pipeline.py       # Task 2 runner with ISRO deliverables
├── isro_metric_evaluator.py # Importable ISRO metrics and deliverables API
├── Batch-Isro-Evaluator.py # Batch .1/.2 pair evaluator
├── batch_isro_evaluator.py # Importable batch evaluator API
├── preprocessing.py       # Loading, normalization, multi-scale LCN, resizing
├── matching_engine.py     # LoFTR matching
├── subpixel_uniformity.py # Spatial filtering and subpixel refinement
├── geometry_warping.py    # MAGSAC homography, RMSE, warping
├── metrics_and_vis.py     # Metrics and visual artifacts
├── evaluation.py          # SIFT baseline and evaluation helpers
├── data/sample/           # Sample low-sun and high-sun images
├── outputs/               # Generated results
└── tests/                 # Unit and pipeline tests
```

## Installation

Open PowerShell in the project root:

```powershell
cd C:\College\SIH2026\Codes
```

Create and activate a virtual environment if one does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

The web interface is included in the requirements and uses Streamlit:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

## Run the Command-Line Pipeline

From `C:\College\SIH2026\Codes`:

```powershell
.\.venv\Scripts\python.exe .\SunAngle\Run-After-Lro.py `
	--low .\SunAngle\data\sample\nac_low.png `
	--high .\SunAngle\data\sample\nac_high.png `
	--out .\SunAngle\outputs
```

For your own image pair, replace the values of `--low` and `--high` with the
low-sun/reference and high-sun/target image paths.

The processing order is:

1. Run the SIFT baseline.
2. Load and normalize both images.
3. Apply multi-scale LCN to reduce illumination differences.
4. Generate LoFTR correspondences.
5. Enforce spatially distributed matches.
6. Refine valid coordinates to subpixel precision.
7. Estimate a robust homography with USAC MAGSAC.
8. Calculate inlier ratio, reprojection RMSE, spatial metrics, NCC, mutual information, and PSNR.
9. Save visual results and `metrics.json`.

## Run the ISRO Task 2 Pipeline

The Task 2 runner adds the ISRO metric evaluator and writes the required
registered product, match points, homography, and metric files:

```powershell
.\.venv\Scripts\python.exe -m SunAngle.Task2pipeline `
	--low .\SunAngle\data\sample\nac_low.png `
	--high .\SunAngle\data\sample\nac_high.png `
	--out .\SunAngle\outputs_task2
```

The same functionality is available from Python through
`SunAngle.run_task2`, `SunAngle.compute_isro_metrics`, and
`SunAngle.save_isro_deliverables`.

Task 2 outputs include `registered_product.jpg`, `match_points.csv`,
`H.npy`, `metrics.json`, `isro_metrics.json`, and the visual artifacts listed
below.

## Run Batch ISRO Evaluation

Batch mode evaluates multiple image pairs with one LoFTR engine instance.
The evaluator supports either of these layouts:

**One folder:** matching files use `<image_id>.1.png` and
`<image_id>.2.png`. The `.1` image is treated as the high-sun target and `.2`
as the low-sun reference:

```text
quickmap/
├── 1.1.png
├── 1.2.png
├── 2.1.png
└── 2.2.png
```

```powershell
.\.venv\Scripts\python.exe -m SunAngle.batch_isro_evaluator `
	--pair_dir .\SunAngle\data\quickmap `
	--out .\SunAngle\batch_outputs
```

**Two folders:** use `--low_dir` for `.2` images and `--high_dir` for `.1`
images. Files are paired by their shared image ID:

```powershell
.\.venv\Scripts\python.exe -m SunAngle.batch_isro_evaluator `
	--low_dir .\SunAngle\data\low_2 `
	--high_dir .\SunAngle\data\high_1 `
	--out .\SunAngle\batch_outputs
```

For a quick validation run, limit the number of pairs:

```powershell
.\.venv\Scripts\python.exe -m SunAngle.batch_isro_evaluator `
	--pair_dir .\SunAngle\data\quickmap `
	--out .\SunAngle\batch_outputs `
	--max_pairs 5
```

Useful options are `--resize` for the maximum image dimension, `--conf` for
the LoFTR confidence threshold, `--ransac` for the geometric reprojection
threshold, and `--no_vis` to skip registered images and visual artifacts.
Run `python -m SunAngle.batch_isro_evaluator --help` for the complete list.

Batch outputs have this structure:

```text
batch_outputs/
├── 001_1/
│   ├── metrics.json
│   ├── isro_metrics.json
│   ├── match_points.csv
│   └── registered_product.jpg
├── batch_metrics.csv
├── batch_summary.json
└── plots/rmse_histogram.png
```

The per-pair folders also contain homography arrays and visual artifacts when
visualization is enabled. `batch_metrics.csv` contains one row per evaluated
pair. `batch_summary.json` contains total pairs, ISRO pass count and rate,
mean and median RMSE, mean 3x3 coverage, and timing information. The plot is
written when Matplotlib is available.

The Streamlit interface exposes the one-folder layout through **Batch folder**:
choose a local directory with **Browse**, confirm the detected pair count, and
select **Run batch evaluation**. The web interface downloads
`batch_metrics.csv` and `batch_summary.json` after completion.

## Generated Outputs

The default output directory is `SunAngle/outputs/`:

| File | Description |
|---|---|
| `metrics.json` | Numeric evaluation results |
| `match_lines.jpg` | Green inlier and red outlier correspondence lines |
| `spatial_density.jpg` | Spatial distribution of matches |
| `checkerboard_overlay.jpg` | Reference/registered image alignment view |
| `difference_heatmap.jpg` | Registration residual heatmap |
| `registered_target.jpg` | Target image warped into the reference frame |
| `lcn_ref.jpg` | Normalized reference image |
| `lcn_target.jpg` | Normalized target image |

Important metrics include:

- `raw_matches`: LoFTR matches after confidence filtering.
- `uniform_matches`: matches retained after spatial filtering.
- `inlier_count`: matches accepted by the homography model.
- `inlier_ratio`: `inlier_count / number of processed matches`.
- `rmse_pixels`: inlier reprojection error in pixels.
- `spatial_coverage` and `spatial_entropy`: distribution across the image.

## Run Tests

Install the test runner if necessary:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
```

Run the SunAngle test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest .\SunAngle\tests\test_pipeline.py -q
```

## Run the Interactive Web Interface

From `c:\College\SIH2026\SIH26166\Codes\LunarX`:

```powershell
..\.venv\Scripts\python.exe -m streamlit run web_app.py
```

Streamlit opens the interface in a browser at `http://localhost:8501`.

### Supported Web UI Modes:
1. **Single Pair (Multi-Scale Alignment)**:
   - Upload Reference Image (Low-Sun / Coarse / Base) and Target Image (High-Sun / Fine / Target).
   - Select or auto-detect sensor/GSD (OHRC 0.25m, LRO NAC 0.5m, TMC-2 5.0m, SELENE 10.0m, IIRS 80.0m, or Custom GSD).
   - Live scale ratio banner indicating MTF anti-aliasing filter and downsampling factor.
   - Computes sub-pixel tie points (Gruen LSM / LK fallback), MAGSAC++ homography, and displays:
     - ISRO metrics (RMSE, 80/20 Checkpoint RMSE in px & ground meters, SDI 8x8, 3x3 Grid Coverage, Scale consistency).
     - Interactive tabs for Inlier Correspondences, Diagnostic Checkerboards, Difference Heatmaps, Spatial Densities, and Match Points table.
     - Single-click downloads for `registered_product.jpg`, `match_points.csv`, `isro_metrics.json`, and complete ZIP bundle.
2. **Group of Images (Multi-Scale Correspondence)**:
   - Upload 1 Anchor Reference Image and multiple Target Images of varying scales (e.g. TMC-2 vs. OHRC, NAC, IIRS).
   - Real-time progress bar across all target images.
   - Interactive Group Summary Table (Scale ratios, Inliers, Inlier Ratios, Checkpoint RMSE px & m, SDI 8x8, Scale checks, ISRO status).
   - Dropdown inspector to view individual registration artifacts for any image in the group.
   - Download `group_summary.csv`, `group_summary.json`, and full Group ZIP bundle.
3. **Batch Folder (Local Directory)**:
   - Evaluates a local folder containing paired images with summary statistics and CSV export.

## Hardware

The matcher automatically uses CUDA when available and otherwise runs on CPU.
CPU execution is supported but can be slower for large images. Input images are
resized to a maximum long dimension of 1024 pixels by default; this can be
changed through the pipeline API or by editing the runner configuration.
