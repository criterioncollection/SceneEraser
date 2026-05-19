# SceneEraser

영상 속 동적 객체(지나가는 사람)를 자동으로 감지·제거하고 배경을 복원하는 도구입니다.

고정된 카메라로 촬영된 영상에서 행인처럼 화면을 가로지르는 인물을 자동으로 골라 지우고, 발표자처럼 정지해 있는 인물은 그대로 보존합니다. 관광지·미술관·강의실처럼 사람이 오가는 공간에서 촬영한 영상을 후처리할 때 활용할 수 있습니다.

## 주요 기능

- **자동 감지·추적**: YOLOv8n + ByteTrack으로 사람을 검출하고 프레임 간 추적
- **제거 대상 자동 선정**: 이동 거리·화면 가장자리 통과·중앙 체류 비율 등 6개 지표를 가중합산해 행인을 자동 선별 (top-k 조절 가능)
- **정밀 마스크**: SAM 2.1로 픽셀 단위 세그멘테이션 (CPU 환경에서는 bbox 모드)
- **배경 복원**: 시간축 중앙값 기반 배경 plate 생성 후 합성
- **합성 모드 3종**: 안정성 우선 / 자연스러움 우선 / plate only
- **오디오 보존**: 원본 영상의 소리를 결과물에 그대로 유지
- **웹 UI**: Gradio 기반 브라우저 인터페이스 (다크모드 지원)

## 동작 가정

다음 조건에서 가장 좋은 결과를 보입니다.

1. 카메라가 고정되어 있을 것
2. 제거 대상이 일정한 방향으로 이동할 것
3. 보존 대상(KEEP)이 거의 정지해 있을 것

## 요구 사항

- Python 3.10+
- NVIDIA GPU (SAM2 모드 사용 시 권장, 없으면 bbox 모드)
- ffmpeg (오디오 합성에 사용)
- SAM 2.1 체크포인트 (`checkpoints/sam2_small.pt`)

## 설치

```bash
# 의존 패키지
pip install gradio opencv-python numpy pandas ultralytics torch

# SAM2 (facebookresearch 공식)
pip install git+https://github.com/facebookresearch/sam2.git

# ffmpeg (Ubuntu 기준)
sudo apt install -y ffmpeg
```

SAM 2.1 체크포인트를 `checkpoints/sam2_small.pt` 경로에 배치합니다.

## 실행

```bash
cd ~/projects/SceneEraser/demo_v4
cp "$(ls -t /mnt/c/Users/<USERNAME>/Downloads/app*.py | head -1)" ./app.py

python app.py --host 127.0.0.1 --port 7860
```

브라우저에서 `http://127.0.0.1:7860` 으로 접속합니다.

### 실행 옵션

| 옵션 | 설명 |
|---|---|
| `--host` | 서버 주소 (기본 `127.0.0.1`, 외부 허용 시 `0.0.0.0`) |
| `--port` | 포트 (기본 `7860`) |
| `--share` | Gradio 외부 공유 링크 생성 |

## 사용 방법

1. 입력 영상을 업로드 (최대 1920×1080px / 30초로 자동 제한)
2. **제거할 객체 수(top-k)** 설정 — 화면을 지나가는 사람 수에 맞춤
3. **마스크 모드** 선택 — `sam2`(정밀) / `bbox`(빠름) / `auto`
4. **합성 모드** 선택 — A(안정성) / B(자연스러움) / C(plate only)
5. 실행 후 출력 영상과 감지 결과 미리보기 확인

### 합성 모드

- **A · 안정성 우선**: 가까운 5프레임 안에서 픽셀을 찾고, 못 채우면 배경 plate로 대체
- **B · 자연스러움 우선**: 시간 거리 제한 없이 다른 프레임 픽셀을 우선 사용
- **C · plate only**: 마스크 영역 전체를 배경 plate로 교체

## 파이프라인 구조

```
preprocess → extract_frames → detect_track → merge_fragments
  → score_select → generate_masks → refine_masks
  → build_temporal_plate → inpaint_plate → expand_shadow_masks
  → composite_with_mode → mux_audio
```

| 단계 | 역할 |
|---|---|
| `preprocess` | 해상도·길이 제한 |
| `detect_track` | YOLOv8n + ByteTrack 검출·추적 |
| `merge_fragments` | occlusion으로 끊긴 추적 재연결 (거리 우선 정렬로 ID swap 방지) |
| `score_select` | 6개 지표 가중합산으로 제거 대상 선정 |
| `generate_masks` | SAM 2.1 또는 bbox 마스크 생성 |
| `refine_masks` | 모폴로지 연산 + 시간 안정화 |
| `build_temporal_plate` | 시간축 중앙값으로 배경 추정 |
| `inpaint_plate` | plate의 빈 영역 보정 |
| `expand_shadow_masks` | 그림자 영역 마스크 확장 |
| `composite_with_mode` | 최종 합성 (A/B/C) |
| `mux_audio` | 원본 오디오 입히기 |

## 파일 구성

- `pipeline.py` — 객체 제거 파이프라인
- `app.py` — Gradio 웹 인터페이스

## 라이선스

학술 프로젝트용으로 작성되었습니다.
