# TerrainAlign: Robust Multi-Illumination Surface Image Registration

A deep feature matching and geometric alignment pipeline for registering complex surface terrain imagery captured under extreme illumination differences, severe shadow variations, and diverse optical sensors.

The pipeline integrates multi-scale Local Contrast Normalization (LCN), LoFTR (Local Feature Transformer) detector-free correspondences, spatial uniformity filtering, subpixel Lucas-Kanade refinement, and robust USAC MAGSAC homography estimation.

Optimized for high-resolution aerial, satellite, and orbital optical survey datasets with high incidence angle variance and challenging shadow transitions.

## Project Structure

Core components and execution scripts:

```text
├── Run-After-Lro.py       # Command-line entry point for dual-image registration
├── web_app.py             # Interactive Streamlit evaluation interface
├── pipeline.py            # End-to-end alignment orchestration
├── Task2pipeline.py       # Extended evaluation pipeline with metric export
├── isro_metric_evaluator.py # Deliverables and validation metrics module
├── Batch-Isro-Evaluator.py # Batch image pair evaluator
├── batch_isro_evaluator.py # Importable batch evaluation API
├── preprocessing.py       # Normalization, multi-scale LCN, adaptive resizing
├── matching_engine.py     # LoFTR dense matching engine
├── subpixel_uniformity.py # Spatial bucket filtering and subpixel LK refinement
├── geometry_warping.py    # MAGSAC homography fitting, RMSE, image warping
├── metrics_and_vis.py     # Alignment metrics and visual diagnostic outputs
├── evaluation.py          # Classical baseline (SIFT) and benchmark helpers
├── data/sample/           # Sample high/low illumination test pairs
├── outputs/               # Generated evaluation artifacts
└── tests/                 # Unit tests and regression suite
```

## Installation

1. Clone or download the repository and navigate to the project root:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate a virtual environment:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

## Running the Command-Line Pipeline

To align an image pair with significant illumination or shadow differences:

```bash
python Run-After-Lro.py \
    --low ./data/sample/nac_low.png \
    --high ./data/sample/nac_high.png \
    --out ./outputs
```

Replace `--low` and `--high` with your respective reference (e.g. low-angle / high-shadow) and target (e.g. direct-angle / low-shadow) image paths.

### Pipeline Workflow:

1. **Baseline Comparison**: Executes standard keypoint matching baseline (SIFT).
2. **Preprocessing**: Normalizes dynamic range and adjusts input dimensions.
3. **Multi-Scale LCN**: Suppresses harsh cast shadows and enhances low-frequency gradient contrast.
4. **LoFTR Matching**: Produces dense, detector-free point correspondences.
5. **Spatial Uniformity Filtering**: Distributes match points evenly across a spatial grid.
6. **Subpixel LK Refinement**: Refines point coordinates to subpixel accuracy.
7. **Robust Estimation**: Fits a projective homography using USAC MAGSAC.
8. **Diagnostic Evaluation**: Computes inlier ratio, reprojection RMSE, spatial coverage, entropy, NCC, and PSNR.
9. **Artifact Export**: Saves warped images, diagnostic maps, and `metrics.json`.

## Running the Extended Assessment Pipeline (Task 2)

To run the extended validation pipeline that exports standard registered products, tie points, homography matrices, and assessment deliverables:

```bash
python -m Task2pipeline \
    --low ./data/sample/nac_low.png \
    --high ./data/sample/nac_high.png \
    --out ./outputs_task2
```

Programmatic access is available via `Task2pipeline.run_task2`, `compute_isro_metrics`, and `save_isro_deliverables`.

Exported deliverables include `registered_product.jpg`, `match_points.csv`, `H.npy`, `metrics.json`, and `isro_metrics.json`.

## Batch Evaluation Mode

The batch module evaluates multiple image pairs concurrently using a shared model session.

### Folder Formats:

**Option A: Single folder containing pairs** (e.g. `<id>.1.png` as target and `<id>.2.png` as reference):

```text
pairs_folder/
├── 1.1.png
├── 1.2.png
├── 2.1.png
└── 2.2.png
```

```bash
python -m batch_isro_evaluator \
    --pair_dir ./data/pairs_folder \
    --out ./batch_outputs
```

**Option B: Separate reference and target folders**:

```bash
python -m batch_isro_evaluator \
    --low_dir ./data/reference_images \
    --high_dir ./data/target_images \
    --out ./batch_outputs
```

### Useful Parameters:
- `--max_pairs N`: Limit execution to first N pairs for validation.
- `--resize 1024`: Maximum image dimension (rescaled maintaining aspect ratio).
- `--conf 0.2`: LoFTR match confidence threshold.
- `--ransac 3.0`: Geometric inlier reprojection threshold (in pixels).
- `--no_vis`: Skip generation of heavy visual maps for faster throughput.

### Batch Outputs:

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

## Generated Outputs & Metrics

| Output File | Description |
|---|---|
| `metrics.json` | Comprehensive numeric evaluation results |
| `match_lines.jpg` | Correspondence tie lines (green = inliers, red = outliers) |
| `spatial_density.jpg` | Spatial density distribution across the scene |
| `checkerboard_overlay.jpg` | Checkerboard alignment verification |
| `difference_heatmap.jpg` | Pixel-wise residual difference map |
| `registered_target.jpg` | Warped target image registered to the reference frame |
| `lcn_ref.jpg` | Contrast-normalized reference image |
| `lcn_target.jpg` | Contrast-normalized target image |

### Key Metrics Explained:
- `raw_matches`: Initial correspondence count passing confidence threshold.
- `uniform_matches`: Correspondences retained following spatial bucket suppression.
- `inlier_count`: Matches fitting the computed homography.
- `inlier_ratio`: Ratio of inliers to spatial matches (`inlier_count / uniform_matches`).
- `rmse_pixels`: Root mean squared reprojection error of inliers (subpixel goal < 1.0 px).
- `spatial_coverage` & `spatial_entropy`: Geometric distribution score across grid quadrants.

## Running Tests

Run the test suite with pytest:

```bash
pytest ./tests/test_pipeline.py -q
```

## Interactive Web Interface

Launch the interactive Streamlit application:

```bash
streamlit run web_app.py
```

Features:
- Single pair alignment with interactive visual comparison (side-by-side, slider, checkerboard, difference heatmap).
- Real-time parameter tuning (confidence thresholds, RANSAC margins, spatial grid sizes).
- Batch folder processing with automated metrics export (`batch_metrics.csv`, `batch_summary.json`).

## Hardware & Acceleration

- **GPU**: Automatically leverages CUDA when available for high-throughput tensor operations.
- **CPU**: Full fallback support for standard CPU environments.
- **Image Sizing**: Images are automatically rescaled to 1024px maximum dimension for optimal balance between spatial accuracy and GPU memory consumption.
