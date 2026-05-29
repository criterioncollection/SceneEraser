"""
SceneEraser 통합 서버 — v2 (H.264 transcoding 추가)
"""

import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from pipeline import run_pipeline

ROOT = Path(__file__).parent.resolve()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

LABEL_TO_MODE = {"A": "C", "B": "A", "C": "B"}
MODE_TO_DISPLAY = {"C": "A", "A": "B", "B": "C"}

app = FastAPI(title="SceneEraser")


def transcode_h264(src: Path, dst: Path) -> bool:
    """cv2.VideoWriter 의 mp4v 코덱은 Chrome 이 디코딩 못 함. H.264 로 변환."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-profile:v", "high", "-movflags", "+faststart",
             "-c:a", "copy",
             str(dst)],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[transcode_h264] ffmpeg 실패 ({type(e).__name__}): 원본 사용")
        return False


@app.post("/api/process")
async def api_process(
    video: UploadFile = File(...),
    top_k: int = Form(1),
    composite_mode: str = Form("A"),
    mask_mode: str = Form("sam2"),
    tight_mask: bool = Form(False),
    shadow_off: bool = Form(False),
    shadow_cap: bool = Form(False),
    sam2_ckpt: str = Form("checkpoints/sam2_small.pt"),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    in_path = job_dir / f"input{suffix}"
    with in_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    internal_mode = LABEL_TO_MODE.get(composite_mode.upper().strip(), "C")

    try:
        result = run_pipeline(
            str(in_path),
            top_k=int(top_k),
            mask_mode=mask_mode,
            composite_mode=internal_mode,
            sam2_ckpt=(sam2_ckpt.strip() or None),
            tight_mask=bool(tight_mask),
            shadow_off=bool(shadow_off),
            shadow_cap=bool(shadow_cap),
            debug=False,
        )
    except Exception as e:
        return JSONResponse({"error": f"pipeline 오류: {e}"}, status_code=500)

    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)

    preview_url = None
    if result.get("preview") is not None:
        pv_path = job_dir / "preview.jpg"
        cv2.imwrite(str(pv_path), result["preview"])
        preview_url = f"/outputs/{job_id}/preview.jpg"

    out_src = Path(result["output_video"])
    out_dst = job_dir / "output.mp4"
    if not transcode_h264(out_src, out_dst):
        if out_src.resolve() != out_dst.resolve():
            shutil.copy(out_src, out_dst)
    output_url = f"/outputs/{job_id}/output.mp4"

    display_mode = MODE_TO_DISPLAY.get(result["composite_mode"], result["composite_mode"])

    return {
        "job_id": job_id,
        "output_video": output_url,
        "preview": preview_url,
        "n_groups": result["n_groups"],
        "remove_ids": result["remove_ids"],
        "composite_mode": display_mode,
    }


@app.get("/")
def root():
    return RedirectResponse("/SceneEraser.html")


@app.get("/outputs/{job_id}/{filename}")
def serve_output(job_id: str, filename: str):
    job_dir = (OUTPUTS / job_id).resolve()
    if not str(job_dir).startswith(str(OUTPUTS.resolve())):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    full = job_dir / filename
    if not full.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(full)


@app.get("/{path:path}")
def serve_static(path: str):
    full = (ROOT / path).resolve()
    if not str(full).startswith(str(ROOT)):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if full.is_file():
        return FileResponse(full)
    return JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
