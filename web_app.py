"""Streamlit interface for the lunar image correspondence pipeline."""

import csv
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register SunAngle as an alias so internal `from SunAngle.X import ...` works
import importlib, types as _types
_sunangle = _types.ModuleType("SunAngle")
_sunangle.__path__ = [str(PROJECT_ROOT)]
_sunangle.__package__ = "SunAngle"
sys.modules["SunAngle"] = _sunangle

from Task2pipeline import run_task2
from batch_isro_evaluator import run_batch, find_pairs_dot_pattern


st.set_page_config(
    page_title="LunarX Correspondence",
    page_icon="🌙",
    layout="wide",
)


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


def files_zip_bytes(files: list[Path]) -> bytes:
    """Package only the selected result files into a ZIP archive."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, path.name)
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


def render_metrics(metrics: dict) -> None:
    st.subheader("ISRO Task 2 metrics")
    metric_columns = st.columns(4)
    cards = [
        ("Raw matches", metrics.get("raw_matches", 0)),
        ("Uniform matches", metrics.get("uniform_matches", 0)),
        ("MAGSAC++ inliers", metrics.get("inlier_count", 0)),
        ("Inlier ratio", f"{metrics.get('inlier_ratio', 0.0) * 100:.2f}%"),
        ("Inliers <2 px", metrics.get("inlier_count_2px", 0)),
        ("RMSE", f"{metrics.get('rmse_pixels', 0.0):.4f} px"),
        ("3x3 coverage", f"{metrics.get('uniformity_coverage_3x3', 0.0) * 100:.1f}%"),
        ("3x3 entropy", f"{metrics.get('uniformity_entropy_3x3', 0.0):.3f}"),
        ("ISRO status", "PASS" if metrics.get("isro_pass") else "FAIL"),
    ]
    for index, (label, value) in enumerate(cards):
        with metric_columns[index % 4]:
            st.metric(label, value)


def render_evaluation_metrics(metrics: dict) -> None:
    st.subheader("Classical evaluation metrics")
    st.metric("SIFT baseline matches", metrics.get("sift_baseline_matches", 0))


def render_batch_results(output_dir: Path) -> None:
    summary_path = output_dir / "batch_summary.json"
    csv_path = output_dir / "batch_metrics.csv"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.subheader("Batch ISRO summary")
        columns = st.columns(4)
        batch_cards = [
            ("Pairs evaluated", summary.get("evaluated", 0)),
            ("ISRO passes", summary.get("isro_pass_count", 0)),
            ("Pass rate", f"{summary.get('isro_pass_rate', 0.0):.1f}%"),
            ("Mean RMSE", f"{summary.get('mean_rmse', 999.0):.4f} px"),
            ("Mean inlier ratio", f"{summary.get('mean_inlier_ratio', 0.0):.1f}%"),
            ("Mean uniformity", f"{summary.get('mean_uniformity_coverage_3x3', summary.get('mean_coverage_3x3', 0.0)):.1f}%"),
        ]
        for index, (label, value) in enumerate(batch_cards):
            with columns[index]:
                st.metric(label, value)

    if csv_path.exists():
        st.download_button(
            "Download batch_metrics.csv",
            data=csv_path.read_bytes(),
            file_name="batch_metrics.csv",
            mime="text/csv",
            on_click="ignore",
        )
    if summary_path.exists():
        st.download_button(
            "Download batch_summary.json",
            data=summary_path.read_bytes(),
            file_name="batch_summary.json",
            mime="application/json",
            on_click="ignore",
        )
    if csv_path.exists() and summary_path.exists():
        st.download_button(
            "Download batch CSV + JSON (ZIP)",
            data=files_zip_bytes([csv_path, summary_path]),
            file_name="batch_metrics_and_summary.zip",
            mime="application/zip",
            on_click="ignore",
        )


def main() -> None:
    st.title("LunarX Correspondence")
    st.write("Evaluate one image pair or a folder of paired lunar images.")

    with st.sidebar:
        st.header("Evaluation mode")
        evaluation_mode = st.radio("Mode", ["Single pair", "Batch folder"])
        resize_long = st.slider("Maximum image dimension", 512, 2048, 1024, 128)

    if evaluation_mode == "Batch folder":
        st.subheader("Batch image folder")
        st.caption("Select a folder containing pairs named like 1.1.png and 1.2.png.")
        folder_columns = st.columns([4, 1])
        with folder_columns[0]:
            batch_directory = st.text_input(
                "Batch folder path",
                value=st.session_state.get("batch_directory", ""),
                placeholder=r"C:\data\quickmap",
            )
        with folder_columns[1]:
            st.write("")
            if st.button("Browse", use_container_width=True):
                selected_directory = choose_directory()
                if selected_directory:
                    st.session_state["batch_directory"] = selected_directory
                    st.rerun()

        max_pairs = st.number_input("Maximum pairs (0 = all)", min_value=0, value=0, step=1)
        run_batch_button = st.button("Run batch evaluation", type="primary", use_container_width=True)
        if not run_batch_button:
            return
        if not batch_directory or not Path(batch_directory).is_dir():
            st.error("Select a valid batch image folder.")
            return

        pairs = find_pairs_dot_pattern(batch_directory)
        if not pairs:
            st.error("No image pairs found. Use names such as 1.1.png and 1.2.png.")
            return
        selected_pair_count = min(len(pairs), max_pairs) if max_pairs else len(pairs)
        st.info(f"Found {len(pairs)} image pairs; evaluating {selected_pair_count}.")

        with st.spinner("Evaluating image pairs and generating ISRO deliverables..."):
            try:
                with tempfile.TemporaryDirectory(prefix="lunarx_batch_") as temp_dir:
                    output_dir = Path(temp_dir) / "batch_outputs"
                    run_batch(
                        pair_dir=batch_directory,
                        out_dir=str(output_dir),
                        resize_long=resize_long,
                        max_pairs=max_pairs or None,
                    )
                    render_batch_results(output_dir)
            except Exception as error:
                st.error(f"The batch evaluation failed: {error}")
        return

    with st.sidebar:
        st.header("Single pair inputs")
        low_sun = st.file_uploader(
            "Low-sun / reference image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="low_sun",
        )
        high_sun = st.file_uploader(
            "High-sun / target image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="high_sun",
        )
        run_matching = st.button("Find correspondence", type="primary", use_container_width=True)

    if low_sun is None or high_sun is None:
        st.info("Upload both images in the sidebar to begin.")
        return

    preview_columns = st.columns(2)
    with preview_columns[0]:
        st.image(low_sun, caption="Low-sun / reference", use_container_width=True)
    with preview_columns[1]:
        st.image(high_sun, caption="High-sun / target", use_container_width=True)

    if not run_matching:
        st.caption("Click 'Find correspondence' to run LoFTR, subpixel refinement, and geometric verification.")
        return

    with st.spinner("Matching images and generating registration artifacts..."):
        try:
            with tempfile.TemporaryDirectory(prefix="lunarx_") as temp_dir:
                temp_path = Path(temp_dir)
                low_path = save_upload(low_sun, temp_path, "low_sun.png")
                high_path = save_upload(high_sun, temp_path, "high_sun.png")
                output_dir = temp_path / "outputs"

                metrics = run_task2(
                    low_path,
                    high_path,
                    out_dir=str(output_dir),
                    resize_long=resize_long,
                )

                render_metrics(metrics)
                render_evaluation_metrics(metrics)
                if metrics.get("isro_pass"):
                    st.success("ISRO Task 2 criteria passed and deliverables were generated.")
                else:
                    st.warning("The matcher completed, but the ISRO Task 2 criteria were not all met.")

                st.subheader("Registration artifacts")
                artifact_columns = st.columns(2)
                artifacts = [
                    ("match_lines.jpg", "Correspondence matches"),
                    ("spatial_density.jpg", "Spatial match distribution"),
                    ("checkerboard_overlay.jpg", "Checkerboard alignment"),
                    ("difference_heatmap.jpg", "Registration difference"),
                    ("registered_product.jpg", "ISRO registered product"),
                    ("registered_target.jpg", "Registered target"),
                    ("lcn_ref.jpg", "Normalized reference"),
                    ("lcn_target.jpg", "Normalized target"),
                ]
                for index, (filename, caption) in enumerate(artifacts):
                    with artifact_columns[index % 2]:
                        show_image(output_dir / filename, caption)

                match_points_path = output_dir / "match_points.csv"
                if match_points_path.exists():
                    st.download_button(
                        "Download match_points.csv",
                        data=match_points_path.read_bytes(),
                        file_name="match_points.csv",
                        mime="text/csv",
                        on_click="ignore",
                    )

                st.download_button(
                    "Download metrics.csv",
                    data=metrics_csv_bytes(metrics),
                    file_name="metrics.csv",
                    mime="text/csv",
                    on_click="ignore",
                )

                st.download_button(
                    "Download metrics.json",
                    data=json.dumps(metrics, indent=2),
                    file_name="metrics.json",
                    mime="application/json",
                    on_click="ignore",
                )

                isro_metrics_path = output_dir / "isro_metrics.json"
                if isro_metrics_path.exists():
                    st.download_button(
                        "Download isro_metrics.json",
                        data=isro_metrics_path.read_bytes(),
                        file_name="isro_metrics.json",
                        mime="application/json",
                        on_click="ignore",
                    )

                st.download_button(
                    "Download all metrics (ZIP)",
                    data=directory_zip_bytes(output_dir, "single_pair_results"),
                    file_name="metrics_bundle.zip",
                    mime="application/zip",
                    on_click="ignore",
                )
        except Exception as error:
            st.error(f"The correspondence run failed: {error}")


if __name__ == "__main__":
    main()
