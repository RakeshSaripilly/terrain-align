"""
SunAngle/api.py
FastAPI REST API Service for Image Correspondence.
Exposes modular endpoints for multi-modal, illumination, and scale-invariant correspondence.
"""

import os
import sys
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SunAngle.pipeline import LunarCorrespondenceEngine

app = FastAPI(
    title="Surface Image Correspondence Engine API",
    description="Multi-Modal, Illumination and Scale Invariant Registration Engine",
    version="1.0.0"
)

# Global engine singleton instance
engine: Optional[LunarCorrespondenceEngine] = None


def get_engine() -> LunarCorrespondenceEngine:
    global engine
    if engine is None:
        engine = LunarCorrespondenceEngine()
    return engine


class MatchPathRequest(BaseModel):
    ref_image_path: str
    target_image_path: str
    resize_long: Optional[int] = 1024
    conf_thresh: Optional[float] = 0.20
    subpixel_refinement: Optional[bool] = True
    enforce_uniformity: Optional[bool] = True
    ransac_thresh: Optional[float] = 2.0


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "pipeline": "TerrainAlign",
        "supported_sensors": ["Optical_HR", "Stereo", "Multispectral"],
        "target_rmse_threshold": "< 0.5 pixels",
        "device": str(get_engine().matcher.device)
    }


@app.post("/api/v1/match")
def match_by_path(req: MatchPathRequest) -> Dict[str, Any]:
    eng = get_engine()
    if not os.path.exists(req.ref_image_path):
        raise HTTPException(status_code=404, detail=f"Reference image not found: {req.ref_image_path}")
    if not os.path.exists(req.target_image_path):
        raise HTTPException(status_code=404, detail=f"Target image not found: {req.target_image_path}")

    res = eng.match(
        req.ref_image_path,
        req.target_image_path,
        resize_long=req.resize_long,
        conf_thresh=req.conf_thresh,
        subpixel_refinement=req.subpixel_refinement,
        enforce_uniformity=req.enforce_uniformity,
        ransac_thresh=req.ransac_thresh
    )

    return {
        "success": res["success"],
        "raw_matches": res["raw_matches"],
        "uniform_matches": res["uniform_matches"],
        "inlier_count": res["inlier_count"],
        "inlier_ratio": res["inlier_ratio"],
        "rmse_pixels": res["rmse_pixels"],
        "subpixel_accurate": res["subpixel_accurate"],
        "spatial_coverage": res["spatial_coverage"],
        "spatial_entropy": res["spatial_entropy"],
        "H_matrix": res["H_scaled"],
        "elapsed_total_sec": res["elapsed_total_sec"],
        "elapsed_match_sec": res["elapsed_match_sec"]
    }


@app.post("/api/v1/register")
def register_images(
    ref_image_path: str = Query(..., description="Path to reference lunar image (e.g. TMC/OHRC)"),
    target_image_path: str = Query(..., description="Path to target lunar image with different sun angle"),
    out_dir: str = Query("SunAngle/outputs", description="Directory to save visual artifacts")
) -> Dict[str, Any]:
    eng = get_engine()
    if not os.path.exists(ref_image_path) or not os.path.exists(target_image_path):
        raise HTTPException(status_code=404, detail="One or both input image paths not found")

    metrics = eng.register_and_visualize(ref_image_path, target_image_path, out_dir=out_dir)
    return {
        "metrics": metrics,
        "artifacts": {
            "match_lines": os.path.join(out_dir, "match_lines.jpg"),
            "spatial_density": os.path.join(out_dir, "spatial_density.jpg"),
            "checkerboard": os.path.join(out_dir, "checkerboard_overlay.jpg"),
            "difference_heatmap": os.path.join(out_dir, "difference_heatmap.jpg"),
            "registered_target": os.path.join(out_dir, "registered_target.jpg")
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("SunAngle.api:app", host="0.0.0.0", port=8000, reload=True)
