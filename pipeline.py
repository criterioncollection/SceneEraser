"""
v4_local pipeline — fixed-camera inpainting (no-flicker edition)
Local GPU 실행 전용. Colab 의존성 없음.

실행:
    python app.py              # Gradio UI
    python app.py --port 7860  # 포트 지정
"""

from __future__ import annotations
import bisect, tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PERSON_CLASS          = 0
CONF_THRES            = 0.25
TRACKER_CFG           = "bytetrack.yaml"
MAX_W, MAX_H, MAX_SEC = 1920, 1080, 30          # 로컬 GPU용 확장 (Colab: 1280/720/10)
DEFAULT_SAM2_CKPT     = "checkpoints/sam2_small.pt"


# ── helpers ────────────────────────────────────────────────────────────────────

def _sorted_frames(d: Path) -> list[Path]:
    return sorted(d.glob("*.jpg"))

def _row_box(row) -> tuple[int, int, int, int]:
    return int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

def _odd(k: int) -> int:
    return k if k % 2 else k + 1


# ── 1. preprocess ──────────────────────────────────────────────────────────────

def preprocess_video(src: Path, dst: Path) -> dict:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    tw, th = int(w * scale) & ~1, int(h * scale) & ~1
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))
    count = 0
    while count < int(fps * MAX_SEC):
        ret, frame = cap.read()
        if not ret: break
        if scale < 1.0:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        writer.write(frame); count += 1
    cap.release(); writer.release()
    return {"fps": fps, "w": tw, "h": th, "n": count}


# ── 2. extract frames ──────────────────────────────────────────────────────────

def extract_frames(video: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video)); i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(out / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()


# ── 3. detect + track ──────────────────────────────────────────────────────────

def detect_track(video: Path, model_name: str = "yolov8n.pt") -> pd.DataFrame:
    yolo = YOLO(model_name); cap = cv2.VideoCapture(str(video))
    rows, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = yolo.track(source=frame, persist=True, tracker=TRACKER_CFG,
                         classes=[PERSON_CLASS], conf=CONF_THRES, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ids = (res.boxes.id.cpu().numpy().astype(int)
                   if res.boxes.id is not None else [-1] * len(boxes))
            for box, conf, tid in zip(boxes, confs, ids):
                x1, y1, x2, y2 = map(int, box); bw, bh = x2-x1, y2-y1
                rows.append([idx, tid, 0, float(conf),
                              x1, y1, x2, y2, x1+bw/2, y1+bh/2, bw, bh])
        idx += 1
    cap.release()
    return pd.DataFrame(rows, columns=["frame_idx","track_id","class_id","conf",
                                        "x1","y1","x2","y2","cx","cy","w","h"])


# ── 4. merge fragments ─────────────────────────────────────────────────────────

def _track_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx"); f, l = g.iloc[0], g.iloc[-1]
        mw, mh = float(g["w"].mean()), float(g["h"].mean())
        # x velocity (px/frame). 시작과 끝 위치 차이 / 프레임 수.
        # 정적 객체는 절대값 작음, 이동 객체는 큼.
        nframes = max(int(l["frame_idx"]) - int(f["frame_idx"]), 1)
        vel_x = (float(l["cx"]) - float(f["cx"])) / nframes
        rows.append({"track_id": int(tid), "frame_count": len(g),
                     "first_frame": int(f["frame_idx"]), "last_frame": int(l["frame_idx"]),
                     "first_cx": float(f["cx"]), "first_cy": float(f["cy"]),
                     "last_cx": float(l["cx"]),  "last_cy": float(l["cy"]),
                     "mean_w": mw, "mean_h": mh,
                     "mean_diag": float(np.sqrt(mw**2+mh**2)),
                     "vel_x": vel_x})
    return pd.DataFrame(rows).sort_values(["first_frame","track_id"]).reset_index(drop=True)


def merge_fragments(df: pd.DataFrame, min_frames=4, max_gap=24,
                    max_ndist=0.75, max_sratio=0.45,
                    use_velocity: bool = False,
                    static_vel_thresh: float = 2.0) -> pd.DataFrame:
    """
    use_velocity: motion velocity 기반 매칭 검증 사용 여부 (완화 버전).
        False (기본): 거리 + 크기만으로 매칭 (원본 v4 동작).
        True: 두 track 모두 이동 중인 경우에만 velocity 검증.
            정적 track이 끼어 있으면 검증 skip (행인이 멈췄거나 새로 시작 가능성).
            검증 내용:
                1. 방향 모순 차단 (둘 다 이동 + vel_x 부호 반대 → 거부)
                2. 위치 예상 검증 (영상 비례 tolerance = 사람 평균 대각선 1배)
            → 분명한 부작용은 막고, 멈춤이나 detection 흔들림 같은 자연스러운
              케이스는 살려두는 보수적 완화.
    """
    df = df[df["track_id"] >= 0].copy()
    vc = df["track_id"].value_counts()
    df = df[df["track_id"].isin(vc[vc >= min_frames].index)].copy()
    if df.empty: df["group_id"] = pd.Series(dtype=int); return df
    summ = _track_summary(df); rows = summ.to_dict("records")
    ffs = [r["first_frame"] for r in rows]; candidates = []
    for i, a in enumerate(rows):
        lo = bisect.bisect_right(ffs, a["last_frame"])
        hi = bisect.bisect_right(ffs, a["last_frame"] + max_gap)
        for b in rows[lo:hi]:
            dx, dy = b["first_cx"]-a["last_cx"], b["first_cy"]-a["last_cy"]
            nd = np.sqrt(dx*dx+dy*dy) / max((a["mean_diag"]+b["mean_diag"])*0.5, 1.0)
            sw = abs(a["mean_w"]-b["mean_w"]) / max((a["mean_w"]+b["mean_w"])*0.5, 1.0)
            sh = abs(a["mean_h"]-b["mean_h"]) / max((a["mean_h"]+b["mean_h"])*0.5, 1.0)
            if not (nd <= max_ndist and max(sw, sh) <= max_sratio):
                continue
            # ── velocity 검증 (옵션, 완화 버전) ──
            if use_velocity:
                gap = b["first_frame"] - a["last_frame"]
                a_moving = abs(a["vel_x"]) >= static_vel_thresh
                b_moving = abs(b["vel_x"]) >= static_vel_thresh
                # 두 track 모두 이동 중인 경우에만 검증.
                # 한 쪽이 정적이면 멈춤/시작 가능성으로 보고 통과 (T3→T7처럼
                # 행인이 멈춘 케이스에서 정상 매칭이 거부되는 부작용 방지).
                if a_moving and b_moving:
                    # 1. 방향 모순 차단
                    if a["vel_x"] * b["vel_x"] < 0:
                        continue
                    # 2. 위치 예상 검증 (사람 대각선 평균을 tolerance로)
                    expected_cx = a["last_cx"] + a["vel_x"] * gap
                    tol_px = (a["mean_diag"] + b["mean_diag"]) * 0.5
                    if abs(b["first_cx"] - expected_cx) > tol_px:
                        continue
            candidates.append({"src": a["track_id"], "dst": b["track_id"],
                               "gap": b["first_frame"]-a["last_frame"], "nd": float(nd)})
    # 정렬: 거리 우선 → gap 보조 (ID swap 방지).
    candidates.sort(key=lambda x: (x["nd"], x["gap"]))
    succ, pred, valid = {}, {}, set(summ["track_id"])
    for c in candidates:
        s, d = c["src"], c["dst"]
        if s not in valid or d not in valid or s in succ or d in pred: continue
        succ[s] = d; pred[d] = s
    ff_map = dict(zip(summ["track_id"], summ["first_frame"]))
    gmap, gid = {}, 0
    for start in sorted([t for t in valid if t not in pred], key=lambda t: ff_map[t]):
        chain, cur = [start], start
        while cur in succ:
            cur = succ[cur]
            if cur in chain: break
            chain.append(cur)
        for t in chain: gmap[int(t)] = gid
        gid += 1
    for t in sorted(valid):
        if int(t) not in gmap: gmap[int(t)] = gid; gid += 1
    df = df.copy(); df["group_id"] = df["track_id"].map(gmap); return df


# ── 5. score & select ──────────────────────────────────────────────────────────

def _norm(s: pd.Series) -> pd.Series:
    x = s.astype(float).to_numpy(); lo, hi = x.min(), x.max()
    if hi-lo < 1e-8: return pd.Series(np.full_like(x, 0.5, dtype=float), index=s.index)
    return pd.Series((x-lo)/(hi-lo), index=s.index)


def score_select(df: pd.DataFrame, fw: int, fh: int,
                 total: int, top_k: int = 1) -> list[int]:
    cr = 0.4
    cx1, cx2 = fw*(1-cr)/2, fw-fw*(1-cr)/2
    cy1, cy2 = fh*(1-cr)/2, fh-fh*(1-cr)/2
    ex, ey = fw*0.15, fh*0.15; rows = []
    for gid, g in df.groupby("group_id"):
        g = g.sort_values("frame_idx"); f, l = g.iloc[0], g.iloc[-1]
        pts = g[["cx","cy"]].to_numpy(np.float32)
        net = float(np.linalg.norm(pts[-1]-pts[0]))
        mdiag = max(float(np.sqrt(g["w"].mean()**2+g["h"].mean()**2)), 1.0)
        edge = int(f["cx"]<=ex or f["cx"]>=fw-ex or l["cx"]<=ex or l["cx"]>=fw-ex or
                   f["cy"]<=ey or f["cy"]>=fh-ey or l["cy"]<=ey or l["cy"]>=fh-ey)
        in_c = ((g["cx"]>=cx1)&(g["cx"]<=cx2)&(g["cy"]>=cy1)&(g["cy"]<=cy2)).mean()
        rows.append({"group_id": int(gid), "nn": net/mdiag,
                     "xr": abs(l["cx"]-f["cx"])/fw,
                     "cov": len(g)/max(total,1), "cdw": float(in_c),
                     "nf": int(g["track_id"].nunique()), "edge": edge})
    sd = pd.DataFrame(rows)
    sd["score"] = (0.35*_norm(sd["nn"]) + 0.25*_norm(sd["xr"]) + 0.15*sd["edge"]
                   + 0.10*(1-_norm(sd["cov"])) + 0.10*(1-_norm(sd["cdw"]))
                   + 0.05*(1-_norm(sd["nf"])))
    return sd.nlargest(top_k, "score")["group_id"].tolist()


# ── 6. generate masks ──────────────────────────────────────────────────────────

def generate_masks(frames_dir: Path, df: pd.DataFrame, out: Path,
                   mode: str = "auto", sam2_ckpt: str | None = None,
                   dilate: int = 8):
    out.mkdir(parents=True, exist_ok=True)
    fps_list = _sorted_frames(frames_dir)
    first = cv2.imread(str(fps_list[0]))
    fh, fw = first.shape[:2]; n = len(fps_list)

    use_sam2 = (mode == "sam2") or (mode == "auto" and torch.cuda.is_available())
    if use_sam2 and not torch.cuda.is_available():
        raise RuntimeError("SAM2 모드는 CUDA GPU가 필요합니다. bbox 모드를 사용하거나 GPU를 확인하세요.")
    if use_sam2 and sam2_ckpt is None:
        sam2_ckpt = DEFAULT_SAM2_CKPT

    if not use_sam2:
        masks: dict = defaultdict(lambda: np.zeros((fh, fw), np.uint8))
        for _, row in df.iterrows():
            fi = int(row["frame_idx"])
            if fi < n:
                x1, y1, x2, y2 = _row_box(row)
                masks[fi][y1:y2, x1:x2] = 255
        if dilate > 0:
            k = np.ones((dilate, dilate), np.uint8)
            for fi in masks: masks[fi] = cv2.dilate(masks[fi], k)
        for i, fp in enumerate(fps_list):
            cv2.imwrite(str(out / f"{fp.stem}.png"), masks[i])
        return

    from sam2.build_sam import build_sam2_video_predictor
    cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    df2 = df[df["w"]*df["h"] >= 64].copy()
    prompts = [
        g.sort_values(["frame_idx", "conf"], ascending=[True, False]).iloc[0]
        for _, g in df2.groupby("track_id")
    ]
    if not prompts:
        for fp in fps_list:
            cv2.imwrite(str(out / f"{fp.stem}.png"), np.zeros((fh, fw), np.uint8))
        return

    pred = build_sam2_video_predictor(cfg, sam2_ckpt)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(str(frames_dir))
        for row in prompts:
            pred.add_new_points_or_box(
                inference_state=state, frame_idx=int(row["frame_idx"]),
                obj_id=int(row["track_id"]),
                box=np.array([row["x1"], row["y1"], row["x2"], row["y2"]], np.float32))
        segs: dict = {}
        for fi, oids, logits in pred.propagate_in_video(state):
            segs[fi] = {}
            for j, oid in enumerate(oids):
                m = (logits[j] > 0.0).detach().cpu().numpy()
                segs[fi][oid] = np.squeeze(m, 0) if m.ndim == 3 else m

    k = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    for i, fp in enumerate(fps_list):
        m = np.zeros((fh, fw), np.uint8)
        for _, obj_m in segs.get(i, {}).items():
            m[obj_m.astype(bool)] = 255
        if k is not None: m = cv2.dilate(m, k)
        cv2.imwrite(str(out / f"{fp.stem}.png"), m)


# ── 7. refine masks ────────────────────────────────────────────────────────────

def refine_masks(frames_dir: Path, masks_dir: Path, out: Path,
                 dilate=9, close=7, btm_exp=25,
                 win=3, thresh=2) -> list[np.ndarray]:
    out.mkdir(parents=True, exist_ok=True)
    fps_list = _sorted_frames(frames_dir)
    mmap = {p.stem: p for p in masks_dir.glob("*.png")}
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close),)*2)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(dilate),)*2)
    pre = []
    for fp in fps_list:
        m = cv2.imread(str(mmap[fp.stem]), cv2.IMREAD_GRAYSCALE)
        m = (m > 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
        m = cv2.dilate(m, kd)
        if btm_exp > 0:
            ys, xs = np.where(m > 0)
            if len(xs): m[np.clip(ys+btm_exp, 0, m.shape[0]-1), xs] = 1
        pre.append(m)
    radius = win // 2
    result = [(lambda s, e: (sum(pre[j] for j in range(s, e)) >= thresh).astype(np.uint8))(
               max(0, i-radius), min(len(pre), i+radius+1)) for i in range(len(pre))]
    for i, fp in enumerate(fps_list):
        cv2.imwrite(str(out / f"{fp.stem}.png"), result[i] * 255)
    return result


# ── 8. temporal plate ──────────────────────────────────────────────────────────

def build_temporal_plate(frames: list[np.ndarray],
                          masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    h, w = frames[0].shape[:2]
    plate    = np.zeros((h, w, 3), dtype=np.uint8)
    residual = np.zeros((h, w), dtype=bool)
    CHUNK = 64
    for y0 in range(0, h, CHUNK):
        y1 = min(y0 + CHUNK, h)
        chunk = np.stack([f[y0:y1] for f in frames], axis=0).astype(np.float32)
        cmask = np.stack([m[y0:y1] for m in masks], axis=0).astype(bool)
        chunk = np.where(cmask[:, :, :, np.newaxis], np.nan, chunk)
        med = np.nanmedian(chunk, axis=0)
        has_nan = np.isnan(med[:, :, 0])
        med = np.where(has_nan[:, :, np.newaxis], 0.0, med)
        plate[y0:y1]    = med.astype(np.uint8)
        residual[y0:y1] = has_nan
    return plate, (residual * 255).astype(np.uint8)


# ── 9. inpaint plate (once) ────────────────────────────────────────────────────

def inpaint_plate(plate: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if not residual.any():
        return plate
    return cv2.inpaint(plate, residual.astype(np.uint8),
                       inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ── 10. shadow expansion ───────────────────────────────────────────────────────

def expand_shadow_masks(frames: list[np.ndarray],
                        masks: list[np.ndarray],
                        clean_plate: np.ndarray,
                        diff_thresh: int = 25,
                        search_px: int = 60,
                        close_px: int = 9) -> list[np.ndarray]:
    k_search = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(search_px * 2), _odd(search_px * 2)))
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_px),) * 2)
    result = []
    for frame, mask in zip(frames, masks):
        m = (mask > 0).astype(np.uint8) * 255
        if not m.any():
            result.append(m)
            continue
        diff      = cv2.absdiff(frame, clean_plate)
        gray      = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)
        search    = cv2.dilate(m, k_search)
        nearby    = cv2.bitwise_and(motion, search)
        combined  = cv2.bitwise_or(m, nearby)
        combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        result.append(combined)
    return result


# ── 11. composite frames ───────────────────────────────────────────────────────

def composite_frames(frames: list[np.ndarray],
                     masks: list[np.ndarray],
                     clean_plate: np.ndarray,
                     fps: float, out: Path, w: int, h: int) -> None:
    n = len(frames)
    mask_bool = [m.astype(bool) for m in masks]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        if not mask_bool[i].any():
            writer.write(frames[i])
            continue
        fill     = clean_plate.copy()
        unfilled = mask_bool[i].copy()
        for d in range(1, n):
            if not unfilled.any():
                break
            for j in (i - d, i + d):
                if 0 <= j < n and unfilled.any():
                    copyable = unfilled & ~mask_bool[j]
                    if copyable.any():
                        fill[copyable] = frames[j][copyable]
                        unfilled[copyable] = False
        result = frames[i].copy()
        result[mask_bool[i]] = fill[mask_bool[i]]
        writer.write(result)
    writer.release()


# ── 11b. 합성 모드 비교용 함수 ──────────────────────────────────────────────────
# 4가지 모드를 동일한 mask로 처리. 결과 영상은 별도 파일.
# 통계도 같이 반환.

def composite_with_mode(frames: list[np.ndarray],
                        masks: list[np.ndarray],
                        clean_plate: np.ndarray,
                        fps: float, out: Path, w: int, h: int,
                        mode: str = "A",
                        max_search: int = 5) -> dict:
    """
    mode 옵션:
      A: 원본 v4 동작. 모든 시간 거리에서 다른 프레임 우선, plate fallback.
      B: plate-only. mask 영역 = clean_plate. 시간 검색 안 함.
      C: 가까운 프레임만 우선 + plate fallback. d <= max_search 까지만 시도.
      D: A와 B의 50:50 평균.
    Returns: 모드별 통계 dict
    """
    n = len(frames)
    mask_bool = [m.astype(bool) for m in masks]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_mask_px   = 0
    total_plate_px  = 0  # plate fallback으로 채워진 픽셀
    total_other_px  = 0  # 다른 프레임에서 가져온 픽셀

    for i in range(n):
        if not mask_bool[i].any():
            writer.write(frames[i])
            continue

        m_px = int(mask_bool[i].sum())
        total_mask_px += m_px

        if mode == "B":
            # 모드 B: plate만 사용
            result = frames[i].copy()
            result[mask_bool[i]] = clean_plate[mask_bool[i]]
            total_plate_px += m_px
            writer.write(result)
            continue

        if mode == "A":
            search_range = range(1, n)
        elif mode == "C":
            search_range = range(1, max_search + 1)
        elif mode == "D":
            search_range = range(1, n)  # A와 동일하게 채운 후 plate와 평균
        else:
            search_range = range(1, n)

        fill = clean_plate.copy()
        unfilled = mask_bool[i].copy()
        for d in search_range:
            if not unfilled.any():
                break
            for j in (i - d, i + d):
                if 0 <= j < n and unfilled.any():
                    copyable = unfilled & ~mask_bool[j]
                    if copyable.any():
                        fill[copyable] = frames[j][copyable]
                        unfilled[copyable] = False
        # 통계
        plate_used = int(unfilled.sum())   # 못 채워서 plate로 fallback된 픽셀
        other_used = m_px - plate_used     # 다른 프레임에서 가져온 픽셀
        total_plate_px += plate_used
        total_other_px += other_used

        if mode == "D":
            # 가중 평균: A 결과와 plate 50:50
            result_a = frames[i].copy()
            result_a[mask_bool[i]] = fill[mask_bool[i]]
            result_b = frames[i].copy()
            result_b[mask_bool[i]] = clean_plate[mask_bool[i]]
            result = cv2.addWeighted(result_a, 0.5, result_b, 0.5, 0)
        else:
            result = frames[i].copy()
            result[mask_bool[i]] = fill[mask_bool[i]]

        writer.write(result)
    writer.release()

    return {
        "mode": mode,
        "total_mask_px": total_mask_px,
        "total_plate_px": total_plate_px,
        "total_other_px": total_other_px,
        "plate_pct": total_plate_px / max(total_mask_px, 1) * 100,
        "other_pct": total_other_px / max(total_mask_px, 1) * 100,
    }


def make_comparison_grid(in_paths: list[Path], out_path: Path,
                          fps: float, w: int, h: int,
                          labels: list[str],
                          layout: str = "2x2") -> None:
    """
    여러 영상을 그리드로 합치고 라벨 박스 추가.
    layout:
        "2x2": 4개 영상 (좌상/우상/좌하/우하)
        "1x2": 2개 영상 (좌/우)
    출력 해상도는 입력과 동일 (영상 크기는 반으로 줄여 합침).
    """
    caps = [cv2.VideoCapture(str(p)) for p in in_paths]
    if any(not c.isOpened() for c in caps):
        for c in caps: c.release()
        raise RuntimeError("comparison: cannot open input videos")

    if layout == "2x2":
        if len(in_paths) != 4:
            raise ValueError(f"2x2 layout requires 4 videos, got {len(in_paths)}")
        cell_w, cell_h = w // 2, h // 2
    elif layout == "1x2":
        if len(in_paths) != 2:
            raise ValueError(f"1x2 layout requires 2 videos, got {len(in_paths)}")
        cell_w, cell_h = w // 2, h
    else:
        raise ValueError(f"unknown layout: {layout}")

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    while True:
        frames = []
        ok = True
        for c in caps:
            ret, f = c.read()
            if not ret:
                ok = False; break
            frames.append(cv2.resize(f, (cell_w, cell_h)))
        if not ok:
            break
        canvas = np.zeros((h, w, 3), np.uint8)
        if layout == "2x2":
            canvas[0:cell_h,        0:cell_w]      = frames[0]
            canvas[0:cell_h,        cell_w:w]      = frames[1]
            canvas[cell_h:h,        0:cell_w]      = frames[2]
            canvas[cell_h:h,        cell_w:w]      = frames[3]
            cv2.line(canvas, (cell_w, 0), (cell_w, h), (255, 255, 255), 2)
            cv2.line(canvas, (0, cell_h), (w, cell_h), (255, 255, 255), 2)
            positions = [(10, 30), (cell_w + 10, 30),
                         (10, cell_h + 30), (cell_w + 10, cell_h + 30)]
        else:  # 1x2
            canvas[:, 0:cell_w] = frames[0]
            canvas[:, cell_w:w] = frames[1]
            cv2.line(canvas, (cell_w, 0), (cell_w, h), (255, 255, 255), 2)
            positions = [(10, 30), (cell_w + 10, 30)]

        for k, (lbl, pos) in enumerate(zip(labels, positions)):
            (tw_, th_), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (pos[0]-5, pos[1]-th_-5),
                          (pos[0]+tw_+5, pos[1]+5), (0, 0, 0), -1)
            cv2.putText(canvas, lbl, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    for c in caps: c.release()
    writer.release()


def make_quad_comparison(in_paths: list[Path], out_path: Path,
                          fps: float, w: int, h: int,
                          labels: list[str]) -> None:
    """이전 버전 호환용 wrapper. 4개 영상을 2x2로 합침."""
    make_comparison_grid(in_paths, out_path, fps, w, h, labels, layout="2x2")


# ── main entry point ───────────────────────────────────────────────────────────

def run_pipeline(src: str, top_k: int = 1,
                 mask_mode: str = "auto",
                 sam2_ckpt: str | None = None,
                 use_velocity: bool = False,
                 composite_mode: str = "fast") -> dict:
    """
    composite_mode:
        "fast" : A 모드만 합성 (가장 빠름)
        "ac"   : A + C 합성, 1x2 비교 영상
        "abcd" : A + B + C + D 합성, 2x2 비교 영상
    """
    work = Path(tempfile.mkdtemp())
    proc       = work / "proc.mp4"
    frames_dir = work / "frames"
    masks_raw  = work / "masks_raw"
    masks_ref  = work / "masks_ref"
    result_mp4 = work / "result.mp4"

    info   = preprocess_video(Path(src), proc)
    extract_frames(proc, frames_dir)
    tracks = detect_track(proc)

    if tracks.empty:
        return {"error": "영상에서 사람을 감지하지 못했습니다.",
                "output_video": None, "preview": None}

    grouped    = merge_fragments(tracks, use_velocity=use_velocity)
    remove_ids = score_select(grouped, info["w"], info["h"], info["n"], top_k)
    remove_df  = grouped[grouped["group_id"].isin(remove_ids)].copy()

    generate_masks(frames_dir, remove_df, masks_raw, mode=mask_mode, sam2_ckpt=sam2_ckpt)
    refined      = refine_masks(frames_dir, masks_raw, masks_ref)
    fps_list     = _sorted_frames(frames_dir)
    frames       = [cv2.imread(str(fp)) for fp in fps_list]
    plate, res   = build_temporal_plate(frames, refined)
    clean_plate  = inpaint_plate(plate, res)

    # ── 디버그: plate 이미지 저장 ──────────────────────────────────────────
    # plate (inpaint 전): 행인 mask 영역을 NaN으로 두고 픽셀별 중앙값.
    #                     KEEP은 마스킹 안 됐으니 KEEP 모습이 박혀있어야 함.
    # clean_plate (inpaint 후): plate에서 hole(영영 안 보였던 픽셀)을 inpaint로 채운 것.
    plate_dbg       = work / "debug_plate_raw.png"
    clean_plate_dbg = work / "debug_plate_clean.png"
    cv2.imwrite(str(plate_dbg), plate)
    cv2.imwrite(str(clean_plate_dbg), clean_plate)

    shadow_masks = expand_shadow_masks(frames, refined, clean_plate)

    # ── 합성 모드별 처리 ─────────────────────────────────────────────────
    # fast: A만
    # ac:   A + C (1x2 비교)
    # abcd: A + B + C + D (2x2 비교)
    result_A = work / "result_A_original.mp4"
    result_B = work / "result_B_plate_only.mp4"
    result_C = work / "result_C_local_plus_plate.mp4"
    result_D = work / "result_D_blend.mp4"
    composite_stats = {}
    quad_mp4 = None

    # A는 항상 실행 (메인 출력)
    composite_stats["A"] = composite_with_mode(frames, shadow_masks, clean_plate,
                                                info["fps"], result_A,
                                                info["w"], info["h"], mode="A")
    if composite_mode == "ac":
        composite_stats["C"] = composite_with_mode(frames, shadow_masks, clean_plate,
                                                    info["fps"], result_C,
                                                    info["w"], info["h"], mode="C",
                                                    max_search=5)
        # 1x2 비교 영상 (좌:A, 우:C)
        quad_mp4 = work / "result_ac_comparison.mp4"
        make_comparison_grid(
            [result_A, result_C],
            quad_mp4, info["fps"], info["w"], info["h"],
            labels=["A: original (v4)", "C: local+plate (d<=5)"],
            layout="1x2"
        )
        result_B = None
        result_D = None
    elif composite_mode == "abcd":
        composite_stats["B"] = composite_with_mode(frames, shadow_masks, clean_plate,
                                                    info["fps"], result_B,
                                                    info["w"], info["h"], mode="B")
        composite_stats["C"] = composite_with_mode(frames, shadow_masks, clean_plate,
                                                    info["fps"], result_C,
                                                    info["w"], info["h"], mode="C",
                                                    max_search=5)
        composite_stats["D"] = composite_with_mode(frames, shadow_masks, clean_plate,
                                                    info["fps"], result_D,
                                                    info["w"], info["h"], mode="D")
        # 2x2 비교 영상
        quad_mp4 = work / "result_quad_comparison.mp4"
        make_comparison_grid(
            [result_A, result_B, result_C, result_D],
            quad_mp4, info["fps"], info["w"], info["h"],
            labels=["A: original (v4)", "B: plate-only",
                    "C: local+plate (d<=5)", "D: A+plate blend"],
            layout="2x2"
        )
    else:
        # fast: A만
        result_B = None
        result_C = None
        result_D = None

    # 메인 출력은 A
    import shutil as _sh
    _sh.copy(result_A, result_mp4)

    # ── 디버그 정보 (영상 처리에는 영향 없음) ────────────────────────────────
    n_frames = len(frames)
    fw, fh = info["w"], info["h"]
    keep_ids = [int(g) for g in sorted(grouped["group_id"].unique()) if g not in remove_ids]
    refined_bool = [(r > 0) for r in refined]
    shadow_bool  = [(s > 0) for s in shadow_masks]

    # raw mask 다시 로드
    refined_raw_bool = []
    for fp in fps_list:
        rrm = cv2.imread(str(masks_raw / f"{fp.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if rrm is None:
            rrm = np.zeros((fh, fw), np.uint8)
        refined_raw_bool.append(rrm > 0)

    # 그룹 분석
    group_stats = []
    cr = 0.5
    cx1g, cx2g = fw*(1-cr)/2, fw - fw*(1-cr)/2
    cy1g, cy2g = fh*(1-cr)/2, fh - fh*(1-cr)/2
    for gid, g in grouped.groupby("group_id"):
        g = g.sort_values("frame_idx")
        cxs = g["cx"].to_numpy(np.float32)
        cys = g["cy"].to_numpy(np.float32)
        x_travel_ratio = float((cxs.max() - cxs.min()) / max(fw, 1))
        net_disp = float(np.linalg.norm([cxs[-1]-cxs[0], cys[-1]-cys[0]]))
        mdiag = max(float(np.sqrt(g["w"].mean()**2 + g["h"].mean()**2)), 1.0)
        cov = len(g) / max(info["n"], 1)
        cdw = float(((g["cx"] >= cx1g) & (g["cx"] <= cx2g) &
                     (g["cy"] >= cy1g) & (g["cy"] <= cy2g)).mean())
        ex_ = fw * 0.15
        starts_edge = bool(cxs[0] <= ex_ or cxs[0] >= fw - ex_)
        ends_edge   = bool(cxs[-1] <= ex_ or cxs[-1] >= fw - ex_)
        group_stats.append({
            "gid": int(gid), "x_travel_ratio": x_travel_ratio,
            "cov": cov, "cdw": cdw, "nn": net_disp/mdiag,
            "n_frames": int(len(g)),
            "starts_edge": starts_edge, "ends_edge": ends_edge,
        })

    # 디버그 영상 + 로그
    debug_mp4 = work / "debug.mp4"
    debug_log = work / "debug_log.txt"
    log_lines = [
        "=== 디버그 로그 (원본 v4 + 진단만 추가) ===",
        f"영상: {fw}x{fh}, {n_frames} frames, {info['fps']:.1f} fps",
        f"감지된 그룹: {grouped['group_id'].nunique()}",
        f"REMOVE group_ids: {remove_ids}",
        f"KEEP group_ids: {keep_ids}",
        "",
        "=== 그룹 분석 ===",
    ]
    for s in group_stats:
        log_lines.append(
            f"G{s['gid']}: x_travel={s['x_travel_ratio']:.2f}, "
            f"cov={s['cov']:.2f}, cdw={s['cdw']:.2f}, "
            f"net_disp/diag={s['nn']:.2f}, frames={s['n_frames']}, "
            f"starts_edge={s['starts_edge']}, ends_edge={s['ends_edge']}"
        )
    log_lines.append("")

    # Track 단위 분석
    log_lines.append("=== Track 단위 분석 ===")
    log_lines.append("(track_id별 시간 범위, 위치 변화, 소속 group)")
    log_lines.append("")
    track_info = []
    for tid, tg in grouped.groupby("track_id"):
        tg = tg.sort_values("frame_idx")
        gid = int(tg["group_id"].iloc[0])
        cls = "KEEP" if gid in keep_ids else "REMOVE"
        f0 = int(tg["frame_idx"].iloc[0])
        f1 = int(tg["frame_idx"].iloc[-1])
        cx0 = float(tg["cx"].iloc[0])
        cx1_v = float(tg["cx"].iloc[-1])
        cy0 = float(tg["cy"].iloc[0])
        cy1_v = float(tg["cy"].iloc[-1])
        cx_mean = float(tg["cx"].mean())
        cy_mean = float(tg["cy"].mean())
        n_t = len(tg)
        nframes = max(f1 - f0, 1)
        vel_x = (cx1_v - cx0) / nframes
        kind = "정적" if abs(vel_x) < 2.0 else f"이동({'→' if vel_x > 0 else '←'})"
        log_lines.append(
            f"Track {int(tid)} → G{gid} ({cls}): "
            f"frames [{f0:>3}-{f1:>3}] {n_t}f, "
            f"cx {cx0:.0f}→{cx1_v:.0f} (Δ{cx1_v-cx0:+.0f}px), "
            f"cy {cy0:.0f}→{cy1_v:.0f}, "
            f"avg ({cx_mean:.0f},{cy_mean:.0f}), "
            f"vel_x={vel_x:+.2f}px/f [{kind}]"
        )
        track_info.append({
            "tid": int(tid), "gid": gid, "f0": f0, "f1": f1,
            "cx0": cx0, "cx1": cx1_v, "cy0": cy0, "cy1": cy1_v,
            "cx_mean": cx_mean, "cy_mean": cy_mean, "n": n_t, "vel_x": vel_x
        })
    log_lines.append("")

    # Group 구성 + 빈 구간
    log_lines.append("=== Group 구성 ===")
    for gid in sorted(grouped["group_id"].unique()):
        gid = int(gid)
        cls = "KEEP" if gid in keep_ids else "REMOVE"
        gtracks = sorted(grouped[grouped["group_id"] == gid]["track_id"].unique().tolist())
        gframes = sorted(grouped[grouped["group_id"] == gid]["frame_idx"].astype(int).unique().tolist())
        if gframes:
            present = set(gframes)
            missing = [i for i in range(gframes[0], gframes[-1] + 1) if i not in present]
            gaps = []
            if missing:
                start = missing[0]; prev = missing[0]
                for m in missing[1:]:
                    if m == prev + 1:
                        prev = m
                    else:
                        gaps.append((start, prev))
                        start = m; prev = m
                gaps.append((start, prev))
            gap_str = ", ".join(f"[{a}-{b}]" for a, b in gaps[:5])
            if len(gaps) > 5:
                gap_str += f" ... (+{len(gaps)-5} more)"
            log_lines.append(
                f"G{gid} ({cls}): tracks={gtracks}, "
                f"frames=[{gframes[0]}-{gframes[-1]}] ({len(gframes)}f), "
                f"내부 빈 구간={len(gaps)}개"
            )
            if gaps:
                log_lines.append(f"   빈 구간: {gap_str}")
    log_lines.append("")

    # ID swap 의심 케이스
    log_lines.append("=== ID swap / 같은 사람 의심 케이스 ===")
    log_lines.append("(track A가 끝난 직후 track B가 비슷한 위치에서 시작)")
    suspects = []
    for a in track_info:
        for b in track_info:
            if a["tid"] >= b["tid"]:
                continue
            time_gap = b["f0"] - a["f1"]
            if 0 <= time_gap <= 30:
                dist = ((a["cx1"] - b["cx0"])**2 + (a["cy1"] - b["cy0"])**2)**0.5
                if dist < 100:
                    same_group = "✓ 같은 group" if a["gid"] == b["gid"] else "✗ 다른 group"
                    suspects.append((time_gap, dist, a, b, same_group))
    if suspects:
        suspects.sort(key=lambda t: (t[0], t[1]))
        for tg, d, a, b, sg in suspects:
            log_lines.append(
                f"T{a['tid']} (G{a['gid']}) frame {a['f1']} 끝, "
                f"T{b['tid']} (G{b['gid']}) frame {b['f0']} 시작 "
                f"(gap={tg}f, 거리={d:.0f}px) [{sg}]"
            )
    else:
        log_lines.append("의심 케이스 없음")
    log_lines.append("")

    # 시간 흐름 visualization
    log_lines.append("=== Track 시간 흐름 (X = 등장, . = 없음) ===")
    bin_size = max(1, n_frames // 60)
    sorted_tids = sorted({int(t) for t in grouped["track_id"].unique()})
    log_lines.append(f"(각 칸 = {bin_size}프레임)")
    for tid in sorted_tids:
        tg = grouped[grouped["track_id"] == tid]
        tframes = set(tg["frame_idx"].astype(int).tolist())
        gid = int(tg["group_id"].iloc[0])
        cls = "K" if gid in keep_ids else "R"
        line = f"T{tid:>2} {cls}: "
        for i in range(0, n_frames, bin_size):
            if any(j in tframes for j in range(i, min(i + bin_size, n_frames))):
                line += "X"
            else:
                line += "."
        log_lines.append(line)
    log_lines.append("")

    # 프레임별 통계
    log_lines.append(
        f"{'frame':>6} | {'raw_remove':>10} | {'refined':>8} | {'shadow':>7}"
    )
    log_lines.append("-" * 50)

    # 디버그 영상 작성
    dbg_writer = cv2.VideoWriter(str(debug_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info["fps"], (fw, fh))
    for i in range(n_frames):
        dbg = frames[i].copy()
        rr = refined_bool[i]
        sb = shadow_bool[i]
        raw = refined_raw_bool[i]

        # 노랑: raw remove (refine 전)
        if raw.any():
            yellow = np.zeros_like(dbg); yellow[:] = (0, 255, 255)
            dbg[raw] = cv2.addWeighted(dbg, 0.6, yellow, 0.4, 0)[raw]
        # 녹색: refined (refine 후)
        if rr.any():
            green = np.zeros_like(dbg); green[:] = (0, 255, 0)
            dbg[rr] = cv2.addWeighted(dbg, 0.5, green, 0.5, 0)[rr]
        # 빨강: shadow (motion 확장 후 최종)
        if sb.any():
            red = np.zeros_like(dbg); red[:] = (0, 0, 255)
            dbg[sb] = cv2.addWeighted(dbg, 0.6, red, 0.4, 0)[sb]

        # 좌상단 통계
        info_text = [
            f"frame: {i}/{n_frames-1}",
            f"raw_remove : {int(raw.sum()):>7}",
            f"refined    : {int(rr.sum()):>7}",
            f"shadow     : {int(sb.sum()):>7}",
        ]
        overlay = dbg.copy()
        cv2.rectangle(overlay, (0, 0), (300, 115), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, dbg, 0.4, 0, dbg)
        for k, line in enumerate(info_text):
            cv2.putText(dbg, line, (10, 25 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 우상단 색상 범례
        legend = [
            ("YELLOW: raw remove (SAM2)", (0, 255, 255)),
            ("GREEN : refined (after refine)", (0, 255, 0)),
            ("RED   : shadow (after expand)", (0, 0, 255)),
        ]
        lx = fw - 320; ly = 5
        overlay2 = dbg.copy()
        cv2.rectangle(overlay2, (lx, ly), (fw - 5, ly + 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, dbg, 0.4, 0, dbg)
        for k, (label, col) in enumerate(legend):
            cv2.rectangle(dbg, (lx + 5, ly + 8 + k * 25),
                          (lx + 22, ly + 22 + k * 25), col, -1)
            cv2.putText(dbg, label, (lx + 30, ly + 22 + k * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        dbg_writer.write(dbg)
        log_lines.append(
            f"{i:>6} | {int(raw.sum()):>10} | {int(rr.sum()):>8} | {int(sb.sum()):>7}"
        )
    dbg_writer.release()

    log_lines.append("")
    log_lines.append("=== 합성 모드 비교 (mask 영역 채우기 방식) ===")
    log_lines.append(f"{'mode':>6} | {'mask px':>10} | {'plate':>10} | {'plate%':>7} | {'other':>10} | {'other%':>7}")
    log_lines.append("-" * 70)
    # 실제로 실행된 모드만 출력
    for mk in ["A", "B", "C", "D"]:
        if mk not in composite_stats:
            continue
        s = composite_stats[mk]
        log_lines.append(
            f"{mk:>6} | {s['total_mask_px']:>10} | {s['total_plate_px']:>10} "
            f"| {s['plate_pct']:>6.1f}% | {s['total_other_px']:>10} | {s['other_pct']:>6.1f}%"
        )
    log_lines.append("")
    log_lines.append("=== 모드 설명 ===")
    log_lines.append("A: 원본 v4 — 다른 프레임 우선 (시간 무한대), plate fallback")
    if "B" in composite_stats:
        log_lines.append("B: plate-only — mask 영역 = clean_plate (가장 단순)")
    if "C" in composite_stats:
        log_lines.append("C: local+plate — 가까운 5프레임 안에서만 시도, 못 채우면 plate")
    if "D" in composite_stats:
        log_lines.append("D: blend — A 결과와 plate의 50:50 평균")
    log_lines.append("")
    log_lines.append("=== 요약 ===")
    log_lines.append(f"velocity 검증 모드: {'ON' if use_velocity else 'OFF'}")
    log_lines.append(f"합성 모드: {composite_mode}")
    log_lines.append(f"평균 raw_remove: {np.mean([m.sum() for m in refined_raw_bool]):.0f}")
    log_lines.append(f"평균 refined:    {np.mean([m.sum() for m in refined_bool]):.0f}")
    log_lines.append(f"평균 shadow:     {np.mean([m.sum() for m in shadow_bool]):.0f}")

    debug_log.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[디버그 로그 저장] {debug_log}")

    preview = cv2.imread(str(frames_dir / "000000.jpg"))
    if preview is not None:
        fi0 = int(grouped["frame_idx"].min())
        for _, row in grouped[grouped["frame_idx"] == fi0].iterrows():
            gid   = int(row["group_id"])
            color = (0, 0, 255) if gid in remove_ids else (0, 255, 0)
            label = f"REMOVE G{gid}" if gid in remove_ids else f"KEEP G{gid}"
            x1, y1, x2, y2 = _row_box(row)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(preview, label, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return {"output_video": str(result_mp4), "preview": preview,
            "debug_video": str(debug_mp4),
            "debug_log": str(debug_log),
            "debug_plate_raw":   str(plate_dbg),
            "debug_plate_clean": str(clean_plate_dbg),
            "result_A": str(result_A),
            "result_B": str(result_B) if result_B is not None else None,
            "result_C": str(result_C) if result_C is not None else None,
            "result_D": str(result_D) if result_D is not None else None,
            "result_quad": str(quad_mp4) if quad_mp4 is not None else None,
            "remove_ids": remove_ids,
            "n_groups":   int(grouped["group_id"].nunique())}
