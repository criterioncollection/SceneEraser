"""
Gradio UI — SceneEraser (Local GPU)

실행:
    python app.py                        # localhost:7860
    python app.py --port 8080            # 포트 변경
    python app.py --share                # 외부 공유 링크 생성
    python app.py --host 0.0.0.0         # 네트워크 내 다른 기기에서 접근 허용
"""

import argparse
import tempfile

import cv2
import gradio as gr

from pipeline import run_pipeline


def process(video_path, top_k, mask_mode, composite_mode_label,
            tight_mask, shadow_off, shadow_cap,
            debug_on, theme, sam2_ckpt):
    # 반환: preview, output_video, debug_video, debug_log, plate_image, status
    if video_path is None:
        return None, None, None, "", None, "영상을 업로드해주세요."

    if int(top_k) == 0:
        return None, None, None, "", None, "제거할 객체 수를 1개 이상으로 설정해주세요."

    ckpt = sam2_ckpt.strip() or None

    # 디버그는 라이트모드 + 체크 ON일 때만. 다크모드(시연)면 무조건 OFF.
    debug = bool(debug_on) and (theme == "light")

    # UI 라벨(A/B/C) → 내부 모드(C/A/B) 매핑
    label_to_mode = {
        "A · 안정성 우선":      "C",   # 내부 C 모드 = local+plate
        "B · 자연스러움 우선":  "A",   # 내부 A 모드 = 원본 v4
        "C · plate only":      "B",   # 내부 B 모드 = plate-only
    }
    composite_mode = label_to_mode.get(composite_mode_label, "C")

    # 내부 모드 → 표시용 라벨 (상태 메시지에 사용)
    mode_to_display = {"C": "A", "A": "B", "B": "C"}

    try:
        result = run_pipeline(video_path, top_k=int(top_k),
                              mask_mode=mask_mode,
                              composite_mode=composite_mode,
                              sam2_ckpt=ckpt,
                              tight_mask=bool(tight_mask),
                              shadow_off=bool(shadow_off),
                              shadow_cap=bool(shadow_cap),
                              debug=debug)
    except Exception as e:
        return None, None, None, "", None, f"오류: {e}"

    if "error" in result:
        return None, None, None, "", None, result["error"]

    preview_path = None
    if result["preview"] is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, result["preview"])
        preview_path = tmp.name

    # 디버그 산출물 (debug=False면 키 없음)
    debug_video = result.get("debug_video")
    plate_image = result.get("plate_image")
    debug_log_text = ""
    log_path = result.get("debug_log")
    if log_path:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                debug_log_text = f.read()
        except Exception as e:
            debug_log_text = f"로그 읽기 실패: {e}"

    display_mode = mode_to_display.get(result["composite_mode"], result["composite_mode"])
    status = (f"감지된 그룹: {result['n_groups']}개 | "
              f"제거 대상: {result['remove_ids']} | "
              f"합성 모드: {display_mode}")
    return (preview_path, result["output_video"], debug_video,
            debug_log_text, plate_image, status)


CUSTOM_CSS = """
/* 강조색 #CD5C5C — 슬라이더, 라디오, 버튼 통일 */
.gradio-container button.primary,
.gradio-container button.lg.primary,
.gradio-container .gr-button-primary,
.gradio-container button[class*="primary"] {
    background: #CD5C5C !important;
    background-color: #CD5C5C !important;
    background-image: none !important;
    border: none !important;
    color: #FFFFFF !important;
    transition: filter 0.2s ease !important;
}
.gradio-container button.primary:hover,
.gradio-container button.lg.primary:hover,
.gradio-container .gr-button-primary:hover,
.gradio-container button[class*="primary"]:hover {
    filter: brightness(1.1) !important;
    background: #CD5C5C !important;
}
.gradio-container button.primary:active,
.gradio-container button[class*="primary"]:active {
    filter: brightness(0.9) !important;
}

/* 슬라이더 */
.gradio-container input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 6px !important;
    background: #CD5C5C !important;
    background-image: none !important;
    border-radius: 3px !important;
    outline: none !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    background: #CD5C5C !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    cursor: pointer !important;
}
.gradio-container input[type="range"]::-moz-range-thumb {
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    background: #CD5C5C !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    cursor: pointer !important;
}

/* 라디오 */
input[type="radio"]:checked { accent-color: #CD5C5C !important; }
.gradio-container input[type="radio"]:checked {
    accent-color: #CD5C5C !important;
}

/* 다크모드에서만 SceneEraser 제목을 강조색으로 */
html.dark #app-header h2 {
    color: #CD5C5C !important;
}

/* SceneEraser 제목 폰트 살짝 키움 */
#app-header h2 {
    font-size: calc(1.5em + 12pt) !important;
}

/* 다크모드(시연)에서는 디버그 UI 숨김 — 라이트모드에서만 표시 */
html.dark #debug-toggle,
html.dark #debug-output {
    display: none !important;
}
"""


DARK_TOGGLE_JS = """
() => {
    const html = document.documentElement;
    html.classList.toggle('dark');
    const isDark = html.classList.contains('dark');
    return [isDark ? '라이트모드' : '다크모드', isDark ? 'dark' : 'light'];
}
"""


with gr.Blocks(title="SceneEraser", css=CUSTOM_CSS,
               theme=gr.themes.Default(primary_hue="red")) as demo:
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
                "## SceneEraser\n"
                "영상에서 동적 객체(사람)를 자동 감지하고 제거합니다.  \n"
                "입력 영상은 **최대 1920×1080px / 30초**로 자동 제한됩니다.",
                elem_id="app-header"
            )
        with gr.Column(scale=1, min_width=140):
            dark_btn = gr.Button("라이트모드", size="sm", variant="secondary")

    # 다크모드 여부 추적 (hidden). 기본 다크모드이므로 'dark'
    theme_state = gr.Textbox(value="dark", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="입력 영상 업로드", height=540)
            top_k       = gr.Slider(0, 5, value=1, step=1,
                                    label="제거할 객체 수 (top-k)")
            mask_mode   = gr.Radio(
                ["sam2", "bbox", "auto"],
                value="sam2",
                label="마스크 모드",
                info="sam2: 정밀 (기본) / bbox: 빠름 / auto: GPU 있으면 SAM2, 없으면 bbox",
            )
            composite_radio = gr.Radio(
                ["A · 안정성 우선", "B · 자연스러움 우선", "C · plate only"],
                value="A · 안정성 우선",
                label="합성 모드",
                info="A: 가까운 5프레임 + plate fallback (기본) / "
                     "B: 다른 프레임 픽셀 우선 / "
                     "C: mask 영역 전부 배경 plate로 교체",
            )
            tight_mask_check = gr.Checkbox(
                value=False,
                label="마스크 과확장 축소",
                info="마스크 확장량을 줄여 배경 침범 감소 (SAM2 권장)",
            )
            shadow_off_check = gr.Checkbox(
                value=False,
                label="그림자 안 지움",
                info="사람만 제거하고 그림자는 그대로 둠",
            )
            shadow_cap_check = gr.Checkbox(
                value=False,
                label="그림자 상한",
                info="과도하게 번지는 프레임은 그림자 확장 생략",
            )
            debug_check = gr.Checkbox(
                value=False,
                label="디버그 모드",
                info="마스크 오버레이 영상 + 통계 로그 + 배경 plate 생성 (처리 시간 증가)",
                elem_id="debug-toggle",
            )
            sam2_ckpt   = gr.Textbox(
                label="SAM2 checkpoint 경로 (sam2 모드 시 필수)",
                value="checkpoints/sam2_small.pt",
            )
            run_btn = gr.Button("실행", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(label="출력 영상 (객체 제거)", height=540)
            preview_img  = gr.Image(label="감지 결과 (빨강=제거 / 초록=유지)", height=540)
            status_box   = gr.Textbox(label="상태", interactive=False)

    with gr.Accordion("디버그 출력 (디버그 모드 ON 시)", open=False,
                      elem_id="debug-output"):
        gr.Markdown(
            "디버그 영상 색상: "
            "🟨 **노랑** = raw 마스크 (refine 전) / "
            "🟩 **초록** = refined (행인 최종 마스크) / "
            "🟥 **빨강** = shadow (실제 지워질 최종 영역)"
        )
        with gr.Row():
            debug_video = gr.Video(label="디버그 영상 (마스크 오버레이)", height=480)
            plate_image = gr.Image(label="배경 plate (clean plate)", height=480)
        debug_log = gr.Textbox(label="디버그 로그 (프레임별 통계)",
                               lines=20, max_lines=300, interactive=False)

    run_btn.click(
        fn=process,
        inputs=[video_input, top_k, mask_mode, composite_radio,
                tight_mask_check, shadow_off_check,
                shadow_cap_check, debug_check, theme_state, sam2_ckpt],
        outputs=[preview_img, video_output, debug_video,
                 debug_log, plate_image, status_box],
    )

    dark_btn.click(fn=None, inputs=None, outputs=[dark_btn, theme_state],
                   js=DARK_TOGGLE_JS)

    # 페이지 로드 시 다크모드 자동 적용
    demo.load(fn=None, inputs=None, outputs=None,
              js="() => { document.documentElement.classList.add('dark'); }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1",
                        help="서버 주소 (기본: 127.0.0.1 / 외부 허용: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=7860, help="포트 (기본: 7860)")
    parser.add_argument("--share", action="store_true",   help="Gradio 외부 공유 링크 생성")
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
