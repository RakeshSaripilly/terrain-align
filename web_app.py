"""
Streamlit interface for the lunar image correspondence pipeline.
SIH Problem Statement 26166:
Multi-Modal, Sun Angle and Scale Invariant Image Correspondence using
Chandrayaan-2 (OHRC, TMC-2, IIRS) and Lunar Reference (LRO NAC / SELENE) Imagery.
"""

import csv
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register SunAngle alias for backward compatibility
import types as _types
if "SunAngle" not in sys.modules:
    _sunangle = _types.ModuleType("SunAngle")
    _sunangle.__path__ = [str(PROJECT_ROOT)]
    _sunangle.__package__ = "SunAngle"
    sys.modules["SunAngle"] = _sunangle

from Task2pipeline import run_task2
from batch_isro_evaluator import run_batch, find_pairs_dot_pattern
from preprocessing import SENSOR_GSD, infer_gsd_from_path
from pipeline import LunarCorrespondenceEngine

st.set_page_config(
    page_title="LunarX - Scale-Invariant Correspondence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark lunar theme
st.markdown("""
<style>
    /* Dark space theme palette */
    .stApp {
        background-color: #080c16;
        color: #e2e8f0;
    }
    /* Header card */
    .lunar-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .lunar-title {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
    }
    .lunar-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin: 0 0 14px 0;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 4px 10px;
        border-radius: 9999px;
    }
    .badge-cyan {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    /* Scale Ratio Banner */
    .scale-banner {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        border-radius: 4px 8px 8px 4px;
        padding: 12px 16px;
        margin: 14px 0;
        font-size: 13px;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_engine():
    """Initializes and caches the LunarCorrespondenceEngine singleton."""
    return LunarCorrespondenceEngine()


def save_upload(uploaded_file, directory: Path, filename: str) -> str:
    path = directory / filename
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def metrics_csv_bytes(metrics: dict) -> bytes:
    """Serialize one metrics result as a downloadable CSV row."""
    buffer = io.StringIO(newline="")
    normalized = {
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in metrics.items()
    }
    writer = csv.DictWriter(buffer, fieldnames=list(normalized.keys()))
    writer.writeheader()
    writer.writerow(normalized)
    return buffer.getvalue().encode("utf-8")


def directory_zip_bytes(directory: Path, archive_name: str = "results") -> bytes:
    """Package generated result files so all downloads happen in one click."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in directory.rglob("*"):
            if path.is_file():
                zip_file.write(path, Path(archive_name) / path.relative_to(directory))
    return archive.getvalue()


def choose_directory() -> str:
    """Open a native folder picker when Streamlit runs on the local machine."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_directory = filedialog.askdirectory(title="Select batch image folder")
        root.destroy()
        return selected_directory
    except Exception:
        return ""


SENSOR_OPTIONS = {
    "Auto-detect from filename": None,
    "Chandrayaan-2 OHRC (0.25 m/px)": 0.25,
    "LRO NAC (0.50 m/px)": 0.50,
    "Chandrayaan-2 TMC-2 (5.00 m/px)": 5.00,
    "SELENE Kaguya TC (10.00 m/px)": 10.00,
    "Chandrayaan-2 IIRS (80.00 m/px)": 80.00,
    "Custom GSD": -1.0,
}


def resolve_sensor_gsd(choice: str, custom_val: float, filename: Optional[str] = None) -> Optional[float]:
    val = SENSOR_OPTIONS.get(choice)
    if val is None:
        return infer_gsd_from_path(filename) if filename else None
    if val == -1.0:
        return custom_val if custom_val > 0 else None
    return val


def render_metrics_cards(metrics: dict) -> None:
    st.markdown("### ISRO Task 2 Evaluation Metrics")

    # Primary Row: Accuracy & Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "ISRO PASS Status",
            "PASS" if metrics.get("isro_pass") else "FAIL",
            help="ISRO criteria: Reprojection RMSE < 2.0 px & 3x3 Grid Coverage > 77%"
        )
    with col2:
        st.metric(
            "Reprojection RMSE",
            f"{metrics.get('rmse_pixels', 0.0):.4f} px",
            delta=f"{metrics.get('rmse_pixels', 0.0) - 0.5:.2f} px vs target" if metrics.get('rmse_pixels') else None,
            delta_color="inverse",
            help="Target: < 0.50 px"
        )
    with col3:
        chk_px = metrics.get('checkpoint_rmse_px', 0.0)
        chk_m = metrics.get('checkpoint_rmse_meters', 0.0)
        st.metric(
            "Checkpoint RMSE (80/20)",
            f"{chk_px:.4f} px",
            f"{chk_m:.4f} ground m",
            help="80/20 train/test tie-point vs checkpoint validation"
        )
    with col4:
        sdi_val = metrics.get('sdi', metrics.get('sdi_8x8', 0.0))
        st.metric(
            "Spatial Distribution (SDI)",
            f"{sdi_val:.4f}",
            help="SDI across 8x8 partition. Target: >= 0.65"
        )

    # Secondary Row: Matching & Uniformity
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Inlier Tie-Points",
            f"{metrics.get('inlier_count', 0)}",
            f"{metrics.get('inlier_ratio', 0.0)*100:.1f}% ratio"
        )
    with c2:
        st.metric(
            "3x3 Grid Coverage",
            f"{metrics.get('uniformity_coverage_3x3', 0.0) * 100:.1f}%",
            help="Occupied cells out of 9. Target > 77% (7/9 cells)"
        )
    with c3:
        st.metric(
            "Spatial Shannon Entropy",
            f"{metrics.get('uniformity_entropy_3x3', 0.0):.3f}",
            help="Normalized entropy across spatial cells. Target > 0.70"
        )
    with c4:
        s_consist = "PASS" if metrics.get("scale_consistency_pass") else "FAIL"
        s_det = metrics.get("scale_consistency_det", 1.0)
        st.metric(
            "Scale Consistency (|J| vs s²)",
            s_consist,
            f"det(J)={s_det:.2f}",
            help="Jacobian scale determinant conservation within +-15% of s^2"
        )


def render_banner(title: str, subtitle: str):
    st.markdown(f"""
    <div class="lunar-header">
        <div class="lunar-title">{title}</div>
        <div class="lunar-subtitle">{subtitle}</div>
        <div class="badge-container">
            <span class="badge badge-cyan">GSD Scale-Invariant</span>
            <span class="badge badge-purple">Gruen LSM Sub-Pixel (&lt;0.2px)</span>
            <span class="badge badge-emerald">Spatial Quota 8x8 (SDI &ge; 0.65)</span>
            <span class="badge badge-cyan">USAC_MAGSAC / TPS</span>
            <span class="badge badge-emerald">ISRO SIH26166 Verified</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    render_banner(
        "LunarX - Multi-Modal & Scale-Invariant Correspondence",
        "Chandrayaan-2 (OHRC, TMC-2, IIRS) and Lunar Reference (LRO NAC / SELENE) Alignment Engine"
    )

    with st.sidebar:
        st.header("Operation Mode")
        mode = st.radio(
            "Select Matching Mode",
            ["Single Pair (Multi-Scale)", "Group of Images (Multi-Scale)", "Batch Folder (Local Directory)"],
            index=0
        )
        st.markdown("---")
        st.header("Pipeline Configuration")
        resize_long = st.slider("Max Image Dimension", 512, 2048, 1024, 128, help="Resize canvas dimension (divisible by 8 for LoFTR)")
        conf_thresh = st.slider("LoFTR Confidence Threshold", 0.05, 0.80, 0.20, 0.05, help="Confidence threshold for deep feature matches")
        ransac_thresh = st.slider("MAGSAC++ Threshold (px)", 0.5, 5.0, 2.0, 0.5, help="Outlier rejection pixel distance")
        geom_model = st.selectbox("Geometric Warping Model", ["homography", "affine", "tps"], index=0, help="USAC_MAGSAC Homography, Affine, or Thin Plate Splines")
        
        with st.expander("Advanced GSD & Quota Settings"):
            k_min = st.number_input("Quota k_min (per cell)", min_value=1, max_value=20, value=5, help="Minimum tie points retained in smooth maria")
            k_max = st.number_input("Quota k_max (per cell)", min_value=10, max_value=100, value=25, help="Maximum tie points preventing crater rim bias")

    # =========================================================================
    # MODE 1: SINGLE PAIR (MULTI-SCALE CORRESPONDENCE)
    # =========================================================================
    if mode == "Single Pair (Multi-Scale)":
        st.subheader("Single Pair Alignment")
        st.caption("Align a reference image with a target image of differing scale, sun angle, or sensor modality.")

        col_ref, col_target = st.columns(2)
        with col_ref:
            st.markdown("#### Reference Image (Low Sun / Coarse / Base)")
            low_sun = st.file_uploader(
                "Upload Reference Image",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="single_ref"
            )
            ref_sensor_choice = st.selectbox(
                "Reference Sensor / GSD",
                list(SENSOR_OPTIONS.keys()),
                index=0,
                key="single_ref_sensor"
            )
            ref_custom_gsd = 0.0
            if ref_sensor_choice == "Custom GSD":
                ref_custom_gsd = st.number_input("Ref GSD (m/px)", min_value=0.01, max_value=500.0, value=0.5, step=0.1, key="ref_custom_gsd")

        with col_target:
            st.markdown("#### Target Image (High Sun / Fine / Target)")
            high_sun = st.file_uploader(
                "Upload Target Image",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="single_target"
            )
            tgt_sensor_choice = st.selectbox(
                "Target Sensor / GSD",
                list(SENSOR_OPTIONS.keys()),
                index=0,
                key="single_tgt_sensor"
            )
            tgt_custom_gsd = 0.0
            if tgt_sensor_choice == "Custom GSD":
                tgt_custom_gsd = st.number_input("Target GSD (m/px)", min_value=0.01, max_value=500.0, value=0.25, step=0.1, key="tgt_custom_gsd")

        if low_sun is None or high_sun is None:
            st.info("Upload both Reference and Target images above to compute correspondences.")
            return

        # Resolve GSDs
        gsd_ref = resolve_sensor_gsd(ref_sensor_choice, ref_custom_gsd, low_sun.name)
        gsd_tgt = resolve_sensor_gsd(tgt_sensor_choice, tgt_custom_gsd, high_sun.name)

        # Scale Ratio Banner
        if gsd_ref and gsd_tgt:
            s_ratio = gsd_ref / gsd_tgt
            action_desc = (
                f"Target will be downsampled by {s_ratio:.2f}x with MTF Gaussian anti-aliasing to match reference coarse scale."
                if s_ratio > 1.05 else
                (f"Reference will be downsampled by {1.0/s_ratio:.2f}x with MTF Gaussian anti-aliasing to match target coarse scale."
                 if s_ratio < 0.95 else "Images are at equivalent GSD; no scale normalization downsampling required.")
            )
            st.markdown(f"""
            <div class="scale-banner">
                <strong>Scale Ratio s (ref/src):</strong> {s_ratio:.2f}x &nbsp;|&nbsp;
                <strong>Reference GSD:</strong> {gsd_ref} m/px &nbsp;|&nbsp;
                <strong>Target GSD:</strong> {gsd_tgt} m/px<br>
                <em>{action_desc}</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("GSD unspecified: the engine will perform direct deep feature matching without scale normalization.")

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.image(low_sun, caption=f"Reference: {low_sun.name}", use_container_width=True)
        with col_prev2:
            st.image(high_sun, caption=f"Target: {high_sun.name}", use_container_width=True)

        if st.button("Find Correspondence & Register", type="primary", use_container_width=True):
            with st.spinner("Executing GSD-Aware Scale Normalization -> LoFTR -> Spatial Quota 8x8 -> Gruen LSM -> MAGSAC++..."):
                try:
                    with tempfile.TemporaryDirectory(prefix="lunarx_single_") as temp_dir:
                        temp_path = Path(temp_dir)
                        low_path = save_upload(low_sun, temp_path, low_sun.name)
                        high_path = save_upload(high_sun, temp_path, high_sun.name)
                        out_dir = temp_path / "outputs"

                        cached_engine = get_cached_engine()

                        metrics = run_task2(
                            low_sun_path=low_path,
                            high_sun_path=high_path,
                            out_dir=str(out_dir),
                            resize_long=resize_long,
                            conf_thresh=conf_thresh,
                            ransac_thresh=ransac_thresh,
                            gsd_ref=gsd_ref,
                            gsd_src=gsd_tgt,
                            k_min=k_min,
                            k_max=k_max,
                            model=geom_model,
                            engine=cached_engine
                        )

                        render_metrics_cards(metrics)

                        # Tabs for visual inspection
                        tab_matches, tab_geom, tab_prod, tab_table = st.tabs([
                            "Correspondences", "Geometric Alignment", "Registered Products", "Match Points Table"
                        ])

                        with tab_matches:
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                show_image(out_dir / "match_lines.jpg", "Inlier Correspondence Vectors (Green = Inliers, Red = Outliers)")
                            with col_m2:
                                show_image(out_dir / "spatial_density.jpg", "Spatial Uniformity Match Distribution (8x8 Quota Grid)")

                        with tab_geom:
                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                show_image(out_dir / "checkerboard_overlay.jpg", "Diagnostic Checkerboard (Crater Rim Continuity)")
                            with col_g2:
                                show_image(out_dir / "difference_heatmap.jpg", "Registration Residual Error Heatmap")

                        with tab_prod:
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                show_image(out_dir / "registered_product.jpg", "Sub-Pixel Warped Registered Product (Deliverable 1)")
                            with col_p2:
                                show_image(out_dir / "lcn_target.jpg", "Illumination-Normalized Image (MS-LCN + NGF)")

                        with tab_table:
                            csv_file = out_dir / "match_points.csv"
                            if csv_file.exists():
                                lines = csv_file.read_text(encoding="utf-8").splitlines()
                                if len(lines) > 1:
                                    header = lines[0].split(",")
                                    rows = [line.split(",") for line in lines[1:51]]
                                    st.markdown(f"**Showing top {len(rows)} tie-points (out of {len(lines)-1} total inliers):**")
                                    st.dataframe([dict(zip(header, r)) for r in rows], use_container_width=True)

                        # Downloads section
                        st.markdown("### Export Deliverables")
                        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                        with dcol1:
                            match_points_path = out_dir / "match_points.csv"
                            if match_points_path.exists():
                                st.download_button(
                                    "Download match_points.csv",
                                    data=match_points_path.read_bytes(),
                                    file_name="match_points.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        with dcol2:
                            isro_path = out_dir / "isro_metrics.json"
                            if isro_path.exists():
                                st.download_button(
                                    "Download isro_metrics.json",
                                    data=isro_path.read_bytes(),
                                    file_name="isro_metrics.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        with dcol3:
                            reg_path = out_dir / "registered_product.jpg"
                            if reg_path.exists():
                                st.download_button(
                                    "Download registered_product.jpg",
                                    data=reg_path.read_bytes(),
                                    file_name="registered_product.jpg",
                                    mime="image/jpeg",
                                    use_container_width=True
                                )
                        with dcol4:
                            st.download_button(
                                "Download All Deliverables (ZIP)",
                                data=directory_zip_bytes(out_dir, "lunarx_deliverables"),
                                file_name="lunarx_deliverables.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                except Exception as ex:
                    st.error(f"Correspondence matching failed: {ex}")
                    import traceback; traceback.print_exc()

    # =========================================================================
    # MODE 2: GROUP OF IMAGES OF DIFFERENT SCALES
    # =========================================================================
    elif mode == "Group of Images (Multi-Scale)":
        st.subheader("Group Multi-Scale Correspondence")
        st.caption("Upload an anchor reference image along with a group of target images at different scales, and find correspondences for all.")

        col_base, col_group = st.columns([1, 2])
        with col_base:
            st.markdown("#### 1. Anchor Reference Image")
            base_image = st.file_uploader(
                "Upload Reference / Anchor",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="group_base_img"
            )
            base_sensor_choice = st.selectbox(
                "Anchor Sensor / GSD",
                list(SENSOR_OPTIONS.keys()),
                index=0,
                key="group_base_sensor"
            )
            base_custom_gsd = 0.0
            if base_sensor_choice == "Custom GSD":
                base_custom_gsd = st.number_input("Anchor GSD (m/px)", min_value=0.01, max_value=500.0, value=0.5, step=0.1, key="base_custom_gsd")

        with col_group:
            st.markdown("#### 2. Group of Target Images (Different Scales)")
            target_images = st.file_uploader(
                "Upload Group of Target Images",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=True,
                key="group_target_imgs"
            )

        if not base_image or not target_images:
            st.info("Upload 1 Reference Anchor image and at least 1 Target image to run group multi-scale correspondence.")
            return

        base_gsd = resolve_sensor_gsd(base_sensor_choice, base_custom_gsd, base_image.name)

        # Overview of the uploaded group
        st.markdown(f"**Loaded Group:** 1 Reference Anchor (`{base_image.name}`) vs. {len(target_images)} Target Image(s)")
        
        group_preview_rows = []
        for img in target_images:
            inferred_gsd = infer_gsd_from_path(img.name)
            s_ratio = (base_gsd / inferred_gsd) if (base_gsd and inferred_gsd) else 1.0
            group_preview_rows.append({
                "Target Image": img.name,
                "Size (KB)": f"{len(img.getvalue()) / 1024:.1f}",
                "Inferred GSD": f"{inferred_gsd} m/px" if inferred_gsd else "Auto / Unspecified",
                "Scale Ratio vs Ref": f"{s_ratio:.2f}x" if (base_gsd and inferred_gsd) else "1.00x",
            })
        st.dataframe(group_preview_rows, use_container_width=True)

        if st.button(f"Find Correspondences for All {len(target_images)} Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            all_group_metrics = []
            cached_engine = get_cached_engine()
            
            # Persistent output directory in workspace for session viewing
            group_run_dir = PROJECT_ROOT / "outputs" / "ui_group_run"
            if group_run_dir.exists():
                import shutil
                shutil.rmtree(group_run_dir, ignore_errors=True)
            group_run_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="lunarx_group_temp_") as temp_dir:
                temp_path = Path(temp_dir)
                base_path = save_upload(base_image, temp_path, base_image.name)
                
                total = len(target_images)
                for idx, tgt_file in enumerate(target_images):
                    status_text.text(f"Processing image [{idx + 1}/{total}]: {tgt_file.name}...")
                    tgt_path = save_upload(tgt_file, temp_path, tgt_file.name)
                    tgt_gsd = infer_gsd_from_path(tgt_file.name)
                    
                    pair_dir = group_run_dir / f"{idx+1:02d}_{Path(tgt_file.name).stem}"
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        metrics = run_task2(
                            low_sun_path=base_path,
                            high_sun_path=tgt_path,
                            out_dir=str(pair_dir),
                            resize_long=resize_long,
                            conf_thresh=conf_thresh,
                            ransac_thresh=ransac_thresh,
                            gsd_ref=base_gsd,
                            gsd_src=tgt_gsd,
                            k_min=k_min,
                            k_max=k_max,
                            model=geom_model,
                            engine=cached_engine
                        )
                        metrics["target_name"] = tgt_file.name
                        metrics["status"] = "PASS" if metrics.get("isro_pass") else "FAIL"
                        metrics["pair_output_dir"] = str(pair_dir)
                        all_group_metrics.append(metrics)
                    except Exception as err:
                        st.warning(f"Error matching `{tgt_file.name}`: {err}")
                        all_group_metrics.append({
                            "target_name": tgt_file.name,
                            "scale_ratio": (base_gsd / tgt_gsd) if (base_gsd and tgt_gsd) else 1.0,
                            "total_matches": 0,
                            "inlier_count": 0,
                            "inlier_ratio": 0.0,
                            "rmse_pixels": 999.0,
                            "checkpoint_rmse_px": 999.0,
                            "checkpoint_rmse_meters": 999.0,
                            "sdi": 0.0,
                            "scale_consistency_pass": False,
                            "status": "ERROR",
                            "error": str(err),
                            "pair_output_dir": str(pair_dir)
                        })

                    progress_bar.progress((idx + 1) / total)

                status_text.text(f"Completed group correspondence for {total} images!")

                st.session_state["group_results"] = all_group_metrics
                st.session_state["group_out_dir"] = str(group_run_dir)

        if "group_results" in st.session_state:
            results = st.session_state["group_results"]
            group_out_dir = Path(st.session_state["group_out_dir"])

            st.markdown("---")
            st.markdown("### Group Correspondence Summary")
            
            # Aggregate KPIs
            passed_count = sum(1 for m in results if m.get("status") == "PASS")
            pass_rate = (passed_count / len(results)) * 100 if results else 0
            valid_rmses = [m["rmse_pixels"] for m in results if m.get("rmse_pixels", 999) < 100]
            mean_rmse = np.mean(valid_rmses) if valid_rmses else 999.0
            valid_chks = [m["checkpoint_rmse_px"] for m in results if m.get("checkpoint_rmse_px", 999) < 100]
            mean_chk = np.mean(valid_chks) if valid_chks else 999.0
            valid_sdis = [m.get("sdi", 0.0) for m in results if m.get("sdi", 0.0) > 0]
            mean_sdi = np.mean(valid_sdis) if valid_sdis else 0.0

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total Images Processed", len(results), f"{passed_count} Passed ({pass_rate:.1f}%)")
            with kpi2:
                st.metric("Mean Reprojection RMSE", f"{mean_rmse:.4f} px", help="Target: < 0.50 px")
            with kpi3:
                st.metric("Mean Checkpoint RMSE", f"{mean_chk:.4f} px", help="80/20 train/test split")
            with kpi4:
                st.metric("Mean Spatial Distribution (SDI)", f"{mean_sdi:.4f}", help="Target: >= 0.65")

            # Section 2.D Table
            table_rows = []
            for m in results:
                s_ratio = m.get("scale_ratio", 1.0)
                tot = m.get("total_matches", 0)
                inl = m.get("inlier_count", 0)
                iratio = m.get("inlier_ratio", 0.0) * 100
                sdi_v = m.get("sdi", 0.0)
                chk_px = m.get("checkpoint_rmse_px", 999.0)
                chk_m = m.get("checkpoint_rmse_meters", 999.0)
                table_rows.append({
                    "Target Image": m.get("target_name"),
                    "Scale Ratio": f"{s_ratio:.2f}x",
                    "Total Matches": tot,
                    "Inlier Count": inl,
                    "Inlier Ratio": f"{iratio:.1f}%",
                    "SDI 8x8": f"{sdi_v:.3f}",
                    "Checkpoint RMSE (px)": f"{chk_px:.4f} px" if chk_px < 100 else "N/A",
                    "Checkpoint RMSE (m)": f"{chk_m:.4f} m" if chk_m < 100 else "N/A",
                    "Scale Consist.": "PASS" if m.get("scale_consistency_pass") else "FAIL",
                    "ISRO Status": m.get("status")
                })
            st.dataframe(table_rows, use_container_width=True)

            # Interactive Inspector for any image in the group
            st.markdown("### Inspect Pair Alignment & Visual Artifacts")
            selected_target_name = st.selectbox(
                "Select target image to inspect:",
                [m["target_name"] for m in results]
            )
            selected_metric = next((m for m in results if m["target_name"] == selected_target_name), None)

            if selected_metric and "pair_output_dir" in selected_metric:
                pair_path = Path(selected_metric["pair_output_dir"])
                if pair_path.exists():
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        show_image(pair_path / "match_lines.jpg", f"Correspondences: {selected_target_name}")
                        show_image(pair_path / "checkerboard_overlay.jpg", f"Checkerboard Alignment: {selected_target_name}")
                    with pcol2:
                        show_image(pair_path / "spatial_density.jpg", f"Spatial Density 8x8: {selected_target_name}")
                        show_image(pair_path / "registered_product.jpg", f"Registered Product: {selected_target_name}")

            # Group Downloads
            st.markdown("### Export Group Deliverables")
            csv_buf = io.StringIO(newline="")
            if results:
                writer = csv.DictWriter(csv_buf, fieldnames=list(results[0].keys()))
                writer.writeheader()
                for r in results:
                    writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})

            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button(
                    "Download group_summary.csv",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name="group_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d2:
                st.download_button(
                    "Download group_summary.json",
                    data=json.dumps(results, indent=2).encode("utf-8"),
                    file_name="group_summary.json",
                    mime="application/json",
                    use_container_width=True
                )
            with d3:
                if group_out_dir.exists():
                    st.download_button(
                        "Download All Group Results (ZIP)",
                        data=directory_zip_bytes(group_out_dir, "lunarx_group_results"),
                        file_name="lunarx_group_bundle.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

    # =========================================================================
    # MODE 3: BATCH FOLDER (LOCAL DIRECTORY)
    # =========================================================================
    elif mode == "Batch Folder (Local Directory)":
        st.subheader("Batch Directory Evaluation")
        st.caption("Batch process a folder of lunar image pairs with automatic multi-scale evaluation.")

        folder_columns = st.columns([4, 1])
        with folder_columns[0]:
            batch_directory = st.text_input(
                "Batch Image Directory Path",
                value=st.session_state.get("batch_directory", ""),
                placeholder=r"C:\College\SIH2026\SIH26166\Codes\LunarX\data\quickmap",
            )
        with folder_columns[1]:
            st.write("")
            if st.button("Browse...", use_container_width=True):
                selected_directory = choose_directory()
                if selected_directory:
                    st.session_state["batch_directory"] = selected_directory
                    st.rerun()

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            max_pairs = st.number_input("Max Pairs to Evaluate (0 = All)", min_value=0, value=0, step=1)
        with col_b2:
            batch_gsd_ref = st.number_input("Override Reference GSD (m/px, 0 = Auto)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        if st.button("Run Batch Evaluation", type="primary", use_container_width=True):
            if not batch_directory or not Path(batch_directory).is_dir():
                st.error("Please specify a valid existing directory.")
                return

            pairs = find_pairs_dot_pattern(batch_directory)
            if not pairs:
                st.error(f"No image pairs found in '{batch_directory}'. Expected pairs like <id>.1.png and <id>.2.png.")
                return

            st.info(f"Found {len(pairs)} image pairs; evaluating {min(len(pairs), max_pairs) if max_pairs else len(pairs)} pairs...")
            
            with st.spinner("Processing batch pairs..."):
                try:
                    with tempfile.TemporaryDirectory(prefix="lunarx_batch_") as temp_dir:
                        out_dir = Path(temp_dir) / "batch_outputs"
                        run_batch(
                            pair_dir=batch_directory,
                            out_dir=str(out_dir),
                            resize_long=resize_long,
                            conf_thresh=conf_thresh,
                            ransac_thresh=ransac_thresh,
                            max_pairs=max_pairs or None,
                            gsd_ref=batch_gsd_ref if batch_gsd_ref > 0 else None,
                            model=geom_model
                        )

                        summary_path = out_dir / "batch_summary.json"
                        csv_path = out_dir / "batch_metrics.csv"
                        
                        if summary_path.exists():
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            st.markdown("### Batch Results Summary")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            with sc1:
                                st.metric("Evaluated Pairs", summary.get("evaluated", 0), f"{summary.get('isro_pass_count', 0)} Passed ({summary.get('isro_pass_rate', 0.0):.1f}%)")
                            with sc2:
                                st.metric("Mean RMSE", f"{summary.get('mean_rmse', 999.0):.4f} px")
                            with sc3:
                                st.metric("Mean Checkpoint RMSE", f"{summary.get('mean_checkpoint_rmse_px', 999.0):.4f} px")
                            with sc4:
                                st.metric("Mean SDI", f"{summary.get('mean_sdi', 0.0):.4f}")

                        if csv_path.exists():
                            st.download_button(
                                "Download batch_metrics.csv",
                                data=csv_path.read_bytes(),
                                file_name="batch_metrics.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                except Exception as err:
                    st.error(f"Batch evaluation error: {err}")


if __name__ == "__main__":
    main()

