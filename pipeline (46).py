"""
SceneEraser pipeline — 시연용 (디버그 코드 제거)

흐름:
  preprocess → extract_frames → detect_track → merge_fragments
  → score_select → generate_masks → refine_masks
  → build_temporal_plate → inpaint_plate → expand_shadow_masks
  → composite_with_mode (A 또는 C)

핵심:
  - merge_fragments는 거리(nd) 우선 정렬로 ID swap 방지
  - 합성 모드 A/C 선택 가능
  - preview 이미지로 감지 결과 시각화
"""

import bisect
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ── 상수 ───────────────────────────────────────────────────────────────────────
PERSON_CLASS = 0
CONF_THRES   = 0.25
TRACKER_CFG  = "bytetrack.yaml"
MAX_W, MAX_H, MAX_SEC = 1920, 1080, 30
DEFAULT_SAM2_CKPT = "checkpoints/sam2_small.pt"


# ── 유틸 ───────────────────────────────────────────────────────────────────────
def _odd(k: int) -> int:
    k = max(int(k), 1)
    return k if k % 2 == 1 else k + 1


def _sorted_frames(d: Path) -> list[Path]:
    return sorted(d.glob("*.jpg"))


# ── 1. 영상 전처리 ──────────────────────────────────────────────────────────────
def preprocess_video(src: Path, dst: Path) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("영상 열기 실패")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    tw, th = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    max_frames = int(fps * MAX_SEC)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if scale < 1.0:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return {"fps": fps, "w": tw, "h": th, "n": count}


# ── 2. 프레임 추출 ──────────────────────────────────────────────────────────────
def extract_frames(video: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), frame)
        i += 1
    cap.release()
    return i


# ── 3. YOLO + ByteTrack ───────────────────────────────────────────────────────
def detect_track(video: Path, model_name: str = "yolov8n.pt") -> pd.DataFrame:
    yolo = YOLO(model_name)
    cap = cv2.VideoCapture(str(video))
    rows = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = yolo.track(source=frame, persist=True, tracker=TRACKER_CFG,
                         conf=CONF_THRES, classes=[PERSON_CLASS], verbose=False)
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            for box, tid, conf in zip(
                res[0].boxes.xyxy.cpu().numpy(),
                res[0].boxes.id.cpu().numpy().astype(int),
                res[0].boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                bw, bh = x2 - x1, y2 - y1
                rows.append({
                    "frame_idx": fi, "track_id": int(tid), "conf": float(conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                    "w": bw, "h": bh,
                })
        fi += 1
    cap.release()
    return pd.DataFrame(rows)


# ── 4. merge_fragments (track → group) ────────────────────────────────────────
def _track_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        f, l = g.iloc[0], g.iloc[-1]
        mw, mh = float(g["w"].mean()), float(g["h"].mean())
        rows.append({
            "track_id": int(tid), "frame_count": len(g),
            "first_frame": int(f["frame_idx"]), "last_frame": int(l["frame_idx"]),
            "first_cx": float(f["cx"]), "first_cy": float(f["cy"]),
            "last_cx": float(l["cx"]),  "last_cy": float(l["cy"]),
            "mean_w": mw, "mean_h": mh,
            "mean_diag": float(np.sqrt(mw**2 + mh**2)),
        })
    return pd.DataFrame(rows).sort_values(["first_frame", "track_id"]).reset_index(drop=True)


def merge_fragments(df: pd.DataFrame, min_frames=4, max_gap=24,
                    max_ndist=0.75, max_sratio=0.45) -> pd.DataFrame:
    """
    ByteTrack이 같은 사람을 다른 track_id로 분리한 경우를 group_id로 묶음.
    거리(nd) 우선 정렬로 occlusion 후 ID swap 방지.
    """
    df = df[df["track_id"] >= 0].copy()
    vc = df["track_id"].value_counts()
    df = df[df["track_id"].isin(vc[vc >= min_frames].index)].copy()
    if df.empty:
        df["group_id"] = pd.Series(dtype=int)
        return df
    summ = _track_summary(df)
    rows = summ.to_dict("records")
    ffs = [r["first_frame"] for r in rows]
    candidates = []
    for i, a in enumerate(rows):
        lo = bisect.bisect_right(ffs, a["last_frame"])
        hi = bisect.bisect_right(ffs, a["last_frame"] + max_gap)
        for b in rows[lo:hi]:
            dx = b["first_cx"] - a["last_cx"]
            dy = b["first_cy"] - a["last_cy"]
            nd = np.sqrt(dx * dx + dy * dy) / max(
                (a["mean_diag"] + b["mean_diag"]) * 0.5, 1.0
            )
            sw = abs(a["mean_w"] - b["mean_w"]) / max(
                (a["mean_w"] + b["mean_w"]) * 0.5, 1.0
            )
            sh = abs(a["mean_h"] - b["mean_h"]) / max(
                (a["mean_h"] + b["mean_h"]) * 0.5, 1.0
            )
            if nd <= max_ndist and max(sw, sh) <= max_sratio:
                candidates.append({
                    "src": a["track_id"], "dst": b["track_id"],
                    "gap": b["first_frame"] - a["last_frame"],
                    "nd": float(nd),
                })
    # 거리 우선 → gap 보조 (ID swap 방지)
    candidates.sort(key=lambda x: (x["nd"], x["gap"]))
    succ, pred, valid = {}, {}, set(summ["track_id"])
    for c in candidates:
        s, d = c["src"], c["dst"]
        if s not in valid or d not in valid or s in succ or d in pred:
            continue
        succ[s] = d
        pred[d] = s
    ff_map = dict(zip(summ["track_id"], summ["first_frame"]))
    gmap, gid = {}, 0
    for start in sorted([t for t in valid if t not in pred], key=lambda t: ff_map[t]):
        chain, cur = [start], start
        while cur in succ:
            cur = succ[cur]
            if cur in chain: break
            chain.append(cur)
        for t in chain:
            gmap[int(t)] = gid
        gid += 1
    for t in sorted(valid):
        if int(t) not in gmap:
            gmap[int(t)] = gid
            gid += 1
    df = df.copy()
    df["group_id"] = df["track_id"].map(gmap)
    # ── 진단 출력 (임시) ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"[merge_fragments 진단]  track {df['track_id'].nunique()}개 "
          f"→ group {df['group_id'].nunique()}개")
    for gid, g in df.groupby("group_id"):
        tids = sorted(g["track_id"].unique().tolist())
        print(f"  group {gid}: track {tids}  "
              f"(frame {int(g['frame_idx'].min())}~{int(g['frame_idx'].max())})")
    print("=" * 78)
    return df


# ── 5. score_select ────────────────────────────────────────────────────────────
def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0


def score_select(grouped: pd.DataFrame, top_k: int, fw: int, fh: int) -> list[int]:
    sd = []
    cr = 0.4
    ex, ey = fw * 0.15, fh * 0.15
    cx_lo, cx_hi = fw * (0.5 - cr / 2), fw * (0.5 + cr / 2)
    cy_lo, cy_hi = fh * (0.5 - cr / 2), fh * (0.5 + cr / 2)
    for gid, g in grouped.groupby("group_id"):
        g = g.sort_values("frame_idx")   # 시작/끝 정확히 잡기 위해 정렬
        cx_first, cy_first = g.iloc[0]["cx"], g.iloc[0]["cy"]
        cx_last,  cy_last  = g.iloc[-1]["cx"], g.iloc[-1]["cy"]
        nn = np.sqrt((cx_last - cx_first) ** 2 + (cy_last - cy_first) ** 2)
        xr = (g["cx"].max() - g["cx"].min()) / fw
        edge = 1.0 if (
            cx_first < ex or cx_first > fw - ex or
            cx_last  < ex or cx_last  > fw - ex
        ) else 0.0
        nframes = g["frame_idx"].nunique()
        cov = nframes / max(grouped["frame_idx"].nunique(), 1)
        cdw = ((g["cx"] > cx_lo) & (g["cx"] < cx_hi) &
               (g["cy"] > cy_lo) & (g["cy"] < cy_hi)).mean()
        sd.append({"gid": int(gid), "nn": nn, "xr": xr, "edge": edge,
                   "cov": cov, "cdw": cdw, "nf": g["track_id"].nunique(),
                   "f_start": int(g["frame_idx"].min()),
                   "f_end":   int(g["frame_idx"].max()),
                   "cx_min":  float(g["cx"].min()),
                   "cx_max":  float(g["cx"].max())})
    sd = pd.DataFrame(sd)
    if sd.empty: return []
    sd["score"] = (
        0.35 * _norm(sd["nn"]) + 0.25 * _norm(sd["xr"]) + 0.15 * sd["edge"]
        + 0.10 * (1 - _norm(sd["cov"])) + 0.10 * (1 - _norm(sd["cdw"]))
        + 0.05 * (1 - _norm(sd["nf"]))
    )
    # ── 진단 출력 (임시) ──────────────────────────────────────────────
    diag = sd.sort_values("score", ascending=False)
    print("\n" + "=" * 78)
    print(f"[score_select 진단]  화면: {fw}x{fh}  그룹수: {len(sd)}  top_k: {top_k}")
    print(f"{'gid':>4} {'score':>7} | {'nn':>7} {'xr':>6} {'edge':>5} {'cov':>6} "
          f"{'cdw':>6} {'nf':>3} | {'frame':>11} {'cx범위':>13}")
    print("-" * 78)
    for _, r in diag.iterrows():
        print(f"{int(r['gid']):>4} {r['score']:>7.3f} | "
              f"{r['nn']:>7.1f} {r['xr']:>6.3f} {r['edge']:>5.1f} "
              f"{r['cov']:>6.3f} {r['cdw']:>6.3f} {int(r['nf']):>3} | "
              f"{int(r['f_start']):>4}~{int(r['f_end']):<4} "
              f"{int(r['cx_min']):>5}~{int(r['cx_max']):<5}")
    selected = diag.head(top_k)["gid"].astype(int).tolist()
    print(f"→ 선택된 제거 대상: {selected}")
    print("=" * 78 + "\n")
    return selected


# ── 6. generate_masks (SAM2 또는 bbox) ─────────────────────────────────────────
def _sam2_available() -> bool:
    try:
        import sam2  # noqa
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_masks(frames_dir: Path, remove_df: pd.DataFrame, masks_dir: Path,
                   mode: str = "auto", sam2_ckpt: str | None = None) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    fps = _sorted_frames(frames_dir)
    if not fps: return
    h, w = cv2.imread(str(fps[0])).shape[:2]

    if mode == "auto":
        mode = "sam2" if _sam2_available() else "bbox"

    if mode == "bbox" or remove_df.empty:
        for fp in fps:
            fi = int(fp.stem)
            m = np.zeros((h, w), np.uint8)
            sub = remove_df[remove_df["frame_idx"] == fi]
            for _, r in sub.iterrows():
                cv2.rectangle(m, (int(r["x1"]), int(r["y1"])),
                              (int(r["x2"]), int(r["y2"])), 255, -1)
            m = cv2.dilate(m, np.ones((9, 9), np.uint8), iterations=1)
            cv2.imwrite(str(masks_dir / f"{fi:06d}.png"), m)
        return

    # SAM2 모드
    from sam2.build_sam import build_sam2_video_predictor
    ckpt = sam2_ckpt or DEFAULT_SAM2_CKPT
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml", ckpt, device="cuda"
    )
    df2 = remove_df.sort_values(["track_id", "frame_idx", "conf"],
                                ascending=[True, True, False])
    prompts = [g.iloc[0] for _, g in df2.groupby("track_id")]
    seg_per_frame: dict[int, np.ndarray] = {}

    # 최신 SAM2(2.1)는 bfloat16 autocast 추론을 전제로 함.
    # autocast 컨텍스트 없이 호출하면 dtype 불일치(BFloat16 vs Float) 발생.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))
        obj_id = 1
        for row in prompts:
            fi = int(row["frame_idx"])
            box = np.array([row["x1"], row["y1"], row["x2"], row["y2"]], np.float32)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=fi, obj_id=obj_id, box=box
            )
            obj_id += 1
        for fi, _ids, logits in predictor.propagate_in_video(state):
            if isinstance(logits, torch.Tensor):
                # bfloat16은 numpy로 직접 변환 불가 → float32 경유
                logits = logits.float().cpu().numpy()
            union = np.zeros((h, w), bool)
            for k in range(logits.shape[0]):
                union |= (logits[k, 0] > 0.0)
            seg_per_frame[int(fi)] = (union.astype(np.uint8) * 255)

    kernel8 = np.ones((9, 9), np.uint8)
    for fp in fps:
        fi = int(fp.stem)
        m = seg_per_frame.get(fi, np.zeros((h, w), np.uint8))
        m = cv2.dilate(m, kernel8, iterations=1)
        cv2.imwrite(str(masks_dir / f"{fi:06d}.png"), m)


# ── 7. refine_masks ───────────────────────────────────────────────────────────
def refine_masks(frames_dir: Path, masks_dir: Path, refined_dir: Path,
                 stab_mode: str = "legacy",
                 tight: bool = False) -> list[np.ndarray]:
    """
    stab_mode: 시간 안정화 방식.
        "legacy"        — 기존 로직. 인접 3프레임 중 2개 이상에 마스크가
                          있어야 통과. 등장/퇴장 첫 프레임이 깎이는 부작용.
        "bidirectional" — 양방향 안정화. 빠진 프레임은 이웃 합집합으로 복구,
                          고립된 단일 프레임은 노이즈로 제거. 등장/퇴장 보존.
    tight: True면 마스크 확장량 축소 (dilate 9→5, bottom 25→10).
           SAM2 정밀 마스크에서 배경 과잉 침범을 줄임.
    """
    refined_dir.mkdir(parents=True, exist_ok=True)
    mps = sorted(masks_dir.glob("*.png"))
    raws = [cv2.imread(str(p), 0) for p in mps]
    dil_size    = 5 if tight else 9
    bottom_px   = 10 if tight else 25
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(7), _odd(7)))
    k_dil   = np.ones((_odd(dil_size), _odd(dil_size)), np.uint8)
    proc = []
    for m in raws:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        m = cv2.dilate(m, k_dil, iterations=1)
        # bottom_expand: 다리 그림자 영역 확보 (아래로 확장)
        sh = np.zeros_like(m)
        sh[bottom_px:, :] = m[:-bottom_px, :]
        m = np.maximum(m, sh)
        proc.append(m)

    n = len(proc)
    stab = []
    if stab_mode == "legacy":
        # 기존: 인접 3프레임 중 2개 이상에 마스크가 있어야 통과
        for i in range(n):
            votes = sum(1 for j in range(max(0, i - 1), min(n, i + 2))
                        if proc[j].any())
            if votes >= 2:
                stab.append(proc[i])
            else:
                stab.append(np.zeros_like(proc[i]))
    else:
        # 양방향: 빠진 프레임 복구 + 고립 프레임 제거
        for i in range(n):
            cur  = proc[i].any()
            prev = proc[i - 1].any() if i > 0     else False
            nxt  = proc[i + 1].any() if i < n - 1 else False
            if cur:
                # 마스크 있음 — 양쪽 이웃 모두 없으면 노이즈로 제거
                if not prev and not nxt:
                    stab.append(np.zeros_like(proc[i]))
                else:
                    stab.append(proc[i])
            else:
                # 마스크 없음 — 양쪽 이웃에 다 있으면 깜빡임으로 보고 복구
                if prev and nxt:
                    stab.append(cv2.bitwise_or(proc[i - 1], proc[i + 1]))
                else:
                    stab.append(proc[i])

    for i, m in enumerate(stab):
        cv2.imwrite(str(refined_dir / f"{i:06d}.png"), m)
    return stab


# ── 8. build_temporal_plate ────────────────────────────────────────────────────
def build_temporal_plate(frames: list[np.ndarray],
                         masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    h, w = frames[0].shape[:2]
    plate = np.zeros((h, w, 3), np.uint8)
    residual = np.zeros((h, w), bool)
    CHUNK = 64
    masks_bool = [(m > 0) for m in masks]
    for y0 in range(0, h, CHUNK):
        y1 = min(y0 + CHUNK, h)
        chunk = np.stack([f[y0:y1] for f in frames]).astype(np.float32)  # (n, ch, w, 3)
        cmask = np.stack([m[y0:y1] for m in masks_bool])
        chunk = np.where(cmask[..., None], np.nan, chunk)
        with np.errstate(all="ignore"):
            med = np.nanmedian(chunk, axis=0)
        nan_all = np.isnan(med).any(axis=-1)
        residual[y0:y1] = nan_all
        med[np.isnan(med)] = 0
        plate[y0:y1] = np.clip(med, 0, 255).astype(np.uint8)
    return plate, residual


# ── 9. inpaint_plate ──────────────────────────────────────────────────────────
def inpaint_plate(plate: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if not residual.any(): return plate
    mask = residual.astype(np.uint8) * 255
    return cv2.inpaint(plate, mask, 5, cv2.INPAINT_TELEA)


# ── 10. expand_shadow_masks ───────────────────────────────────────────────────
def expand_shadow_masks(frames: list[np.ndarray], refined: list[np.ndarray],
                        plate: np.ndarray, diff_thresh: int = 25,
                        cap: bool = False) -> list[np.ndarray]:
    """
    cap: True면 한 프레임 추가량이 마스크 면적의 15% 넘으면 확장 스킵.
    """
    search_px = 60
    max_ratio = 0.15
    out = []
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(7), _odd(7)))
    for frame, m in zip(frames, refined):
        if not m.any():
            out.append(m); continue
        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY))
        motion = (diff > diff_thresh).astype(np.uint8) * 255
        near = cv2.dilate(m, np.ones((search_px, search_px), np.uint8), iterations=1)
        added = ((motion > 0) & (near > 0) & (m == 0)).astype(np.uint8) * 255
        # cap: 추가량이 마스크 면적의 일정 비율 넘으면 확장 스킵
        if cap:
            m_area = int((m > 0).sum())
            if m_area > 0 and int((added > 0).sum()) > m_area * max_ratio:
                out.append(m); continue
        combined = np.maximum(m, added)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        out.append(combined)
    return out


# ── 11. composite (A 또는 C) ──────────────────────────────────────────────────
def composite_with_mode(frames: list[np.ndarray], masks: list[np.ndarray],
                        clean_plate: np.ndarray, fps: float, out: Path,
                        w: int, h: int, mode: str = "A",
                        max_search: int = 5) -> None:
    """
    A: 다른 프레임 우선 (시간 무한대), plate fallback
    B: plate-only. mask 영역 전부 clean_plate로 교체 (시간 검색 안 함)
    C: 5프레임 안에서만 시도, 못 채우면 plate
    """
    n = len(frames)
    mask_bool = [m.astype(bool) for m in masks]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        if not mask_bool[i].any():
            writer.write(frames[i]); continue

        if mode == "B":
            # plate-only: mask 영역을 통째로 clean_plate로 교체
            result = frames[i].copy()
            result[mask_bool[i]] = clean_plate[mask_bool[i]]
            writer.write(result)
            continue

        search_range = range(1, max_search + 1) if mode == "C" else range(1, n)
        fill = clean_plate.copy()
        unfilled = mask_bool[i].copy()
        for d in search_range:
            if not unfilled.any(): break
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


# ── 11b. 원본 오디오를 결과 영상에 입히기 ──────────────────────────────────────
def mux_audio(src_video: Path, silent_video: Path, out_video: Path) -> bool:
    """
    OpenCV로 만든 무음 결과 영상에 원본의 오디오를 입힘.
    영상은 재인코딩 없이 복사. 원본에 오디오 없으면 무음 그대로 복사.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        shutil.copy(silent_video, out_video)
        return False

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(src_video)],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():   # 오디오 트랙 없음
        shutil.copy(silent_video, out_video)
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_video), "-i", str(src_video),
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        str(out_video),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_video.exists():
        shutil.copy(silent_video, out_video)
        return False
    return True


# ── 12. preview (감지 결과 시각화) ─────────────────────────────────────────────
def build_preview(frames_dir: Path, grouped: pd.DataFrame,
                  remove_ids: list[int]) -> np.ndarray | None:
    """
    감지 결과 미리보기 (REMOVE=빨강 / KEEP=초록).
    - 등장 그룹이 가장 많은 한 프레임을 골라, 그 프레임 시점의 실제 bbox를 그림
      → 모든 박스가 같은 시점이라 인물끼리 안 겹침.
    - 같은 위치에 겹치는 박스(occlusion으로 쪼개진 같은 인물)는 IoU로 병합.
    - 라벨은 박스 위에 공간 있으면 위, 없으면 박스 안쪽 상단.
    """
    frame_files = _sorted_frames(frames_dir)
    if not frame_files:
        return None

    # 등장 그룹 수가 최대인 프레임 선택
    per_frame = grouped.groupby("frame_idx")["group_id"].nunique()
    bg_idx = int(per_frame.idxmax()) if not per_frame.empty else 0
    preview = cv2.imread(str(frames_dir / f"{bg_idx:06d}.jpg"))
    if preview is None:
        preview = cv2.imread(str(frame_files[0]))
    if preview is None:
        return None

    # 해당 프레임에 등장하는 각 그룹의 bbox 수집
    fr = grouped[grouped["frame_idx"] == bg_idx]
    boxes = []   # (x1, y1, x2, y2, is_remove)
    for gid, g in fr.groupby("group_id"):
        row = g.iloc[0]
        boxes.append([int(row["x1"]), int(row["y1"]),
                      int(row["x2"]), int(row["y2"]),
                      int(gid) in remove_ids])

    # IoU 병합: 같은 분류(remove/keep)이고 많이 겹치면 하나로 합침
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    merged = []
    for box in boxes:
        hit = False
        for m in merged:
            if m[4] == box[4] and iou(m, box) > 0.3:
                m[0] = min(m[0], box[0]); m[1] = min(m[1], box[1])
                m[2] = max(m[2], box[2]); m[3] = max(m[3], box[3])
                hit = True
                break
        if not hit:
            merged.append(box[:])

    # 그리기
    for x1, y1, x2, y2, is_remove in merged:
        color = (0, 0, 220) if is_remove else (0, 200, 0)
        label = "REMOVE" if is_remove else "KEEP"
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        # 라벨: 박스 위 공간 있으면 위, 없으면 박스 안 상단
        ty = y1 - 10 if y1 - 10 > 20 else y1 + 28
        cv2.putText(preview, label, (x1 + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return preview


# ── 13. 디버그 영상 + 로그 생성 ────────────────────────────────────────────────
def build_debug(frames: list[np.ndarray], masks_raw_dir: Path,
                refined: list[np.ndarray], shadow_masks: list[np.ndarray],
                fps: float, w: int, h: int,
                out_video: Path, out_log: Path,
                grouped: pd.DataFrame, remove_ids: list[int]) -> None:
    """
    디버그 영상: 매 프레임에 마스크 단계별 3색 오버레이.
      노랑 — raw 마스크 (SAM2/bbox 원본, refine 전)
      초록 — refined (morphology + 시간안정화 후 = 행인 최종 마스크)
      빨강 — shadow_masks (그림자 확장 후 = 실제 지워질 최종 영역)
    좌상단에 프레임별 픽셀 통계, 우상단에 색상 범례.
    디버그 로그(txt): 영상 정보 + 그룹 + 프레임별 통계 테이블 + 요약.
    """
    n = len(frames)
    # raw 마스크 로드
    raw_files = sorted(masks_raw_dir.glob("*.png"))
    raws = []
    for i in range(n):
        if i < len(raw_files):
            m = cv2.imread(str(raw_files[i]), 0)
            raws.append(m if m is not None else np.zeros((h, w), np.uint8))
        else:
            raws.append(np.zeros((h, w), np.uint8))

    raw_bool    = [r > 0 for r in raws]
    refined_bool = [r > 0 for r in refined]
    shadow_bool  = [s > 0 for s in shadow_masks]

    keep_ids = [g for g in sorted(grouped["group_id"].unique())
                if g not in remove_ids]
    log_lines = [
        "=== SceneEraser 디버그 로그 ===",
        f"영상: {w}x{h}, {n} frames, {fps:.1f} fps",
        f"감지된 그룹: {grouped['group_id'].nunique()}개",
        f"REMOVE group_ids: {remove_ids}",
        f"KEEP group_ids: {keep_ids}",
        "",
        f"{'frame':>6} | {'raw':>8} | {'refined':>8} | {'subtracted':>10} "
        f"| {'shadow':>8} | {'shadow_add':>10}",
        "-" * 70,
    ]

    writer = cv2.VideoWriter(str(out_video),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    legend = [
        ("YELLOW: raw mask (before refine)", (0, 255, 255)),
        ("GREEN : refined (person mask)",    (0, 255, 0)),
        ("RED   : shadow (final removed)",   (0, 0, 255)),
    ]
    for i in range(n):
        dbg = frames[i].copy()
        rw, rf, sh = raw_bool[i], refined_bool[i], shadow_bool[i]
        subtracted = rw & ~rf            # raw에 있었는데 refine에서 빠진 픽셀
        shadow_add = sh & ~rf            # shadow 확장으로 새로 추가된 픽셀

        # 1) 노랑: raw 마스크
        if rw.any():
            yellow = np.zeros_like(dbg); yellow[:] = (0, 255, 255)
            dbg[rw] = cv2.addWeighted(dbg, 0.5, yellow, 0.5, 0)[rw]
        # 2) 초록: refined
        if rf.any():
            green = np.zeros_like(dbg); green[:] = (0, 255, 0)
            dbg[rf] = cv2.addWeighted(dbg, 0.5, green, 0.5, 0)[rf]
        # 3) 빨강: shadow
        if sh.any():
            red = np.zeros_like(dbg); red[:] = (0, 0, 255)
            dbg[sh] = cv2.addWeighted(dbg, 0.55, red, 0.45, 0)[sh]

        # 좌상단 통계 박스
        info_text = [
            f"frame: {i}/{n-1}",
            f"raw       : {int(rw.sum()):>8}",
            f"refined   : {int(rf.sum()):>8}",
            f"subtracted: {int(subtracted.sum()):>8}",
            f"shadow    : {int(sh.sum()):>8}",
            f"shadow add: {int(shadow_add.sum()):>8}",
        ]
        overlay = dbg.copy()
        cv2.rectangle(overlay, (0, 0), (300, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, dbg, 0.4, 0, dbg)
        for k, line in enumerate(info_text):
            cv2.putText(dbg, line, (10, 25 + k * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)

        # 우상단 색상 범례
        lx, ly = w - 330, 5
        overlay2 = dbg.copy()
        cv2.rectangle(overlay2, (lx, ly), (w - 5, ly + 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, dbg, 0.4, 0, dbg)
        for k, (label, col) in enumerate(legend):
            cv2.rectangle(dbg, (lx + 8, ly + 10 + k * 26),
                          (lx + 26, ly + 24 + k * 26), col, -1)
            cv2.putText(dbg, label, (lx + 34, ly + 23 + k * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                        1, cv2.LINE_AA)

        writer.write(dbg)
        log_lines.append(
            f"{i:>6} | {int(rw.sum()):>8} | {int(rf.sum()):>8} "
            f"| {int(subtracted.sum()):>10} | {int(sh.sum()):>8} "
            f"| {int(shadow_add.sum()):>10}"
        )
    writer.release()

    # 요약 통계
    log_lines += [
        "",
        "=== 요약 통계 ===",
        f"평균 raw px     : {np.mean([m.sum() for m in raw_bool]):.0f}",
        f"평균 refined px : {np.mean([m.sum() for m in refined_bool]):.0f}",
        f"평균 shadow px  : {np.mean([m.sum() for m in shadow_bool]):.0f}",
    ]
    out_log.write_text("\n".join(log_lines), encoding="utf-8")


# ── main entry point ──────────────────────────────────────────────────────────
def run_pipeline(src: str, top_k: int = 2,
                 mask_mode: str = "auto",
                 composite_mode: str = "A",
                 sam2_ckpt: str | None = None,
                 tight_mask: bool = False,
                 shadow_off: bool = False,
                 shadow_cap: bool = False,
                 debug: bool = False,
                 progress=None) -> dict:
    """
    composite_mode:
        "A" : 원본 v4. 다른 프레임 우선 (시간 무한대), plate fallback.
        "B" : plate-only. mask 영역 전부 clean_plate로 교체.
        "C" : 가까운 5프레임 안에서만 시도, 못 채우면 plate.
    debug:
        True  — 디버그 영상/로그/plate 이미지 추가 생성.
        False — 시연용. 디버그 산출물 없음 (빠름).
    progress: gradio Progress 객체 또는 None.
    """
    def _step(p: float, msg_ko: str, msg_en: str):
        if progress is not None:
            progress(p, desc=msg_ko)

    work = Path(tempfile.mkdtemp(prefix="scene_eraser_"))
    src_p       = Path(src)
    pre_mp4     = work / "pre.mp4"
    frames_dir  = work / "frames"
    masks_raw   = work / "masks_raw"
    masks_ref   = work / "masks_refined"
    silent_mp4  = work / "result_silent.mp4"
    result_mp4  = work / "result.mp4"
    debug_mp4   = work / "debug.mp4"
    debug_log   = work / "debug_log.txt"
    plate_png   = work / "clean_plate.png"

    _step(0.05, "영상 전처리", "Preprocessing video")
    info = preprocess_video(src_p, pre_mp4)
    if info["n"] == 0:
        return {"error": "유효한 프레임이 없습니다."}

    _step(0.10, "프레임 추출", "Extracting frames")
    extract_frames(pre_mp4, frames_dir)

    _step(0.20, "사람 검출 및 추적", "Detecting and tracking people")
    tracks = detect_track(pre_mp4)
    if tracks.empty:
        return {"error": "영상에서 사람을 감지하지 못했습니다."}

    _step(0.35, "그룹 분석", "Analyzing groups")
    grouped = merge_fragments(tracks)
    remove_ids = score_select(grouped, top_k=top_k, fw=info["w"], fh=info["h"])
    if not remove_ids:
        return {"error": "제거 대상을 결정하지 못했습니다."}
    remove_df = grouped[grouped["group_id"].isin(remove_ids)].copy()

    _step(0.45, "마스크 생성 (SAM2)", "Generating masks (SAM2)")
    generate_masks(frames_dir, remove_df, masks_raw,
                   mode=mask_mode, sam2_ckpt=sam2_ckpt)

    _step(0.65, "마스크 다듬기", "Refining masks")
    refined = refine_masks(frames_dir, masks_raw, masks_ref, tight=tight_mask)

    _step(0.75, "배경 plate 생성", "Building background plate")
    fps_list = _sorted_frames(frames_dir)
    frames = [cv2.imread(str(fp)) for fp in fps_list]
    plate, res = build_temporal_plate(frames, refined)
    clean_plate = inpaint_plate(plate, res)

    if shadow_off:
        _step(0.85, "그림자 확장 생략", "Skipping shadow expansion")
        shadow_masks = refined            # 사람 마스크만 사용
    else:
        _step(0.85, "그림자 영역 확장", "Expanding shadow regions")
        shadow_masks = expand_shadow_masks(frames, refined, clean_plate,
                                           cap=shadow_cap)

    _step(0.92, f"영상 합성 (모드 {composite_mode})",
          f"Compositing (mode {composite_mode})")
    composite_with_mode(frames, shadow_masks, clean_plate, info["fps"],
                        silent_mp4, info["w"], info["h"], mode=composite_mode)

    _step(0.96, "원본 오디오 합성", "Muxing original audio")
    mux_audio(src_p, silent_mp4, result_mp4)

    _step(0.98, "결과 생성", "Generating result")
    preview = build_preview(frames_dir, grouped, remove_ids)

    result = {
        "output_video": str(result_mp4),
        "preview": preview,
        "composite_mode": composite_mode,
        "remove_ids": remove_ids,
        "n_groups": int(grouped["group_id"].nunique()),
    }

    if debug:
        _step(0.99, "디버그 산출물 생성", "Generating debug outputs")
        build_debug(frames, masks_raw, refined, shadow_masks,
                    info["fps"], info["w"], info["h"],
                    debug_mp4, debug_log, grouped, remove_ids)
        cv2.imwrite(str(plate_png), clean_plate)
        result["debug_video"] = str(debug_mp4)
        result["debug_log"]   = str(debug_log)
        result["plate_image"] = str(plate_png)

    _step(1.0, "완료", "Done")
    return result
