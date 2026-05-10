"""
Gradio UI — Dynamic Object Remover v4 + 디버그 + 토글
- 합성 모드 라디오 (fast / AC / ABCD)
- Velocity 비교 체크박스 (ON+OFF 둘 다 처리)

실행:
    python app.py                        # localhost:7860
    python app.py --port 8080            # 포트 변경
    python app.py --share                # 외부 공유 링크 생성
    python app.py --host 0.0.0.0         # 네트워크 내 다른 기기에서 접근 허용
"""

import argparse
import tempfile
import traceback

import cv2
import gradio as gr

from pipeline import run_pipeline


# 출력 슬롯 개수 (UI 컴포넌트와 매칭)
N_OUTPUTS = 19  # preview + quad_off + quad_on + (A,B,C,D)*2 + dbg + plate*2 + log + status


def _save_preview(preview):
    if preview is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, preview)
    return tmp.name


def _empty_outputs():
    return [None] * (N_OUTPUTS - 1)


def process(video_path, top_k, mask_mode, sam2_ckpt,
            composite_mode_label, compare_velocity):
    if video_path is None:
        return _empty_outputs() + ["영상을 업로드해주세요."]

    ckpt = sam2_ckpt.strip() or None

    # 라디오 라벨 → 내부 키
    mode_map = {
        "fast (A만)": "fast",
        "AC 비교":    "ac",
        "ABCD 비교":  "abcd",
    }
    composite_mode = mode_map.get(composite_mode_label, "fast")

    try:
        # velocity OFF 처리 (항상)
        result_off = run_pipeline(video_path, top_k=int(top_k),
                                   mask_mode=mask_mode, sam2_ckpt=ckpt,
                                   use_velocity=False,
                                   composite_mode=composite_mode)
        if "error" in result_off:
            return _empty_outputs() + [result_off["error"]]

        # velocity ON 처리 (체크 시만)
        result_on = None
        if compare_velocity:
            result_on = run_pipeline(video_path, top_k=int(top_k),
                                      mask_mode=mask_mode, sam2_ckpt=ckpt,
                                      use_velocity=True,
                                      composite_mode=composite_mode)
            if "error" in result_on:
                return _empty_outputs() + [result_on["error"]]
    except Exception as e:
        tb = traceback.format_exc()
        return _empty_outputs() + [f"오류: {e}\n\n{tb}"]

    # preview는 OFF 결과 사용
    preview_path = _save_preview(result_off.get("preview"))

    # 로그: OFF + ON 둘 다 합쳐서
    log_text = ""
    try:
        with open(result_off["debug_log"], "r", encoding="utf-8") as f:
            log_text = "===== velocity OFF =====\n" + f.read()
        if result_on is not None:
            with open(result_on["debug_log"], "r", encoding="utf-8") as f:
                log_text += "\n\n\n===== velocity ON =====\n" + f.read()
    except Exception as e:
        log_text = f"로그 읽기 실패: {e}"

    # status
    flags = [f"mode={composite_mode}"]
    if compare_velocity: flags.append("velocity 비교 ON")
    flag_str = " | ".join(flags)
    status = (f"감지된 그룹: {result_off['n_groups']}개 | "
              f"REMOVE: {result_off['remove_ids']} | {flag_str}")

    # 출력 슬롯 채우기
    # OFF 영상들
    quad_off = result_off.get("result_quad")
    A_off = result_off.get("result_A")
    B_off = result_off.get("result_B")
    C_off = result_off.get("result_C")
    D_off = result_off.get("result_D")

    # ON 영상들 (compare_velocity 시만)
    quad_on = A_on = B_on = C_on = D_on = None
    if result_on is not None:
        quad_on = result_on.get("result_quad")
        A_on    = result_on.get("result_A")
        B_on    = result_on.get("result_B")
        C_on    = result_on.get("result_C")
        D_on    = result_on.get("result_D")

    return [preview_path,
            quad_off, quad_on,
            A_off, A_on,
            B_off, B_on,
            C_off, C_on,
            D_off, D_on,
            result_off.get("debug_video"),
            result_off.get("debug_plate_raw"),
            result_off.get("debug_plate_clean"),
            log_text,
            None, None, None,  # 여유 슬롯 (UI 확장 대비)
            status]


with gr.Blocks(title="Dynamic Object Remover - Debug + Toggle") as demo:
    gr.Markdown(
        "## Dynamic Object Remover - 디버그 + 토글\n"
        "합성 모드 라디오 + velocity 비교로 다양한 조합을 테스트."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="입력 영상 업로드")
            top_k       = gr.Slider(1, 5, value=2, step=1, label="제거할 객체 수 (top-k)")
            mask_mode   = gr.Radio(
                ["auto", "bbox", "sam2"],
                value="auto",
                label="마스크 모드",
                info="auto: GPU 있으면 SAM2 / bbox: 빠름 / sam2: 정밀",
            )
            sam2_ckpt   = gr.Textbox(
                label="SAM2 checkpoint 경로",
                value="checkpoints/sam2_small.pt",
            )
            gr.Markdown("### 디버그 토글")
            composite_mode_radio = gr.Radio(
                ["fast (A만)", "AC 비교", "ABCD 비교"],
                value="AC 비교",
                label="합성 모드",
                info="fast: A만 (~15초) / AC: A+C (~18초) / ABCD: 4모드 (~25초)",
            )
            compare_velocity = gr.Checkbox(
                label="Velocity 비교 (OFF + ON 둘 다 처리, 시간 2배)",
                value=False,
                info="velocity 룰(완화): 두 track 모두 이동 중일 때만 검증. 한쪽 정적이면 통과.",
            )
            run_btn = gr.Button("실행", variant="primary")
            gr.Markdown(
                "### 사용 가이드\n"
                "**합성 모드**:\n"
                "- fast: A만 — 가장 빠름 (직전 안정 베이스)\n"
                "- AC: A + C — 1x2 비교 영상\n"
                "- ABCD: A + B + C + D — 2x2 비교 영상\n"
                "\n"
                "**Velocity 비교 (완화 룰)**:\n"
                "- 체크 안 함: velocity OFF로만 처리\n"
                "- 체크: OFF/ON 둘 다 처리. 각 합성 모드 영상도 두 개씩.\n"
                "- 룰: 두 track 모두 이동 중일 때만 검증 (방향 모순 + 위치 예상)\n"
                "- 한쪽이 정적이면 검증 skip → 행인이 멈추는 케이스 정상 매칭 유지\n"
                "\n"
                "**조합 예시**:\n"
                "- AC + Velocity OFF: 영상 2개 (A, C). 가장 균형\n"
                "- AC + Velocity ON: 영상 4개 (A_off, A_on, C_off, C_on)\n"
                "- ABCD + Velocity ON: 영상 8개. 가장 디버깅용\n"
                "\n"
                "**모드 설명**:\n"
                "- A: 원본 v4 — 다른 프레임 우선, plate fallback\n"
                "- B: plate-only — mask = clean_plate\n"
                "- C: local+plate — 5프레임 안에서만, fallback plate\n"
                "- D: blend — A + plate 50:50"
            )

        with gr.Column(scale=2):
            preview_img  = gr.Image(label="감지 결과 (빨강=제거 / 초록=유지)")
            with gr.Row():
                quad_off = gr.Video(label="비교 영상 (velocity OFF) — AC 또는 ABCD 시")
                quad_on  = gr.Video(label="비교 영상 (velocity ON) — Velocity 비교 시")
            status_box   = gr.Textbox(label="상태", interactive=False)

    gr.Markdown("### 개별 모드 영상 (velocity 비교 ON 시 좌:OFF / 우:ON)")
    with gr.Row():
        video_A_off = gr.Video(label="A: original (OFF)")
        video_A_on  = gr.Video(label="A: original (ON)")
    with gr.Row():
        video_B_off = gr.Video(label="B: plate-only (OFF)")
        video_B_on  = gr.Video(label="B: plate-only (ON)")
    with gr.Row():
        video_C_off = gr.Video(label="C: local+plate (OFF)")
        video_C_on  = gr.Video(label="C: local+plate (ON)")
    with gr.Row():
        video_D_off = gr.Video(label="D: blend (OFF)")
        video_D_on  = gr.Video(label="D: blend (ON)")

    gr.Markdown("### 디버그 정보 (velocity OFF 기준)")
    with gr.Row():
        debug_video = gr.Video(label="디버그 영상 (mask 단계별 시각화)")
    with gr.Row():
        plate_raw   = gr.Image(label="plate (inpaint 전)")
        plate_clean = gr.Image(label="clean_plate (inpaint 후)")
    debug_log = gr.Textbox(label="디버그 로그 (velocity OFF + ON 둘 다)",
                           lines=30, max_lines=600, interactive=False)

    # 여유 슬롯 (현재는 안 씀)
    spare1 = gr.Textbox(visible=False)
    spare2 = gr.Textbox(visible=False)
    spare3 = gr.Textbox(visible=False)

    run_btn.click(
        fn=process,
        inputs=[video_input, top_k, mask_mode, sam2_ckpt,
                composite_mode_radio, compare_velocity],
        outputs=[preview_img,
                 quad_off, quad_on,
                 video_A_off, video_A_on,
                 video_B_off, video_B_on,
                 video_C_off, video_C_on,
                 video_D_off, video_D_on,
                 debug_video, plate_raw, plate_clean, debug_log,
                 spare1, spare2, spare3,
                 status_box],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
