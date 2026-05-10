<<<<<<< HEAD
# SceneEraser
=======
# SceneEraser

고정 카메라 영상에서 **행인 자동 제거**, **정적 인물(KEEP) 보존** 파이프라인.

## 무엇을 하는가

갤러리·전시장·로비 같은 공간에서 카메라 앞을 지나는 행인은 지우고, 카메라 앞에서 그림을 보는 사람 같은 정적 객체는 그대로 보존합니다.

- **YOLOv8n + ByteTrack**: 사람 검출 + 추적
- **merge_fragments**: 끊긴 track들을 같은 사람으로 묶음 (occlusion 후 ID swap 보정)
- **score_select**: 행인 점수화로 REMOVE 자동 결정
- **SAM2 small**: 픽셀 단위 mask 생성
- **temporal plate**: 시간 중앙값으로 깨끗한 배경 추출
- **composite**: mask 영역을 다른 시점 픽셀 또는 plate로 채움

## 환경

- Ubuntu (WSL 가능), Python 3.10+
- NVIDIA GPU 권장 (SAM2 사용 시 필수)
- torch 2.x + CUDA, ultralytics, sam2, gradio, opencv-python, pandas, numpy

## 설치

```bash
git clone <this-repo-url>
cd SceneEraser

python -m venv .venv
source .venv/bin/activate

pip install torch torchvision  # CUDA 버전에 맞춰
pip install ultralytics opencv-python pandas numpy gradio
pip install git+https://github.com/facebookresearch/sam2.git

# SAM2 체크포인트 (small)
mkdir -p checkpoints
wget -O checkpoints/sam2_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
```

YOLOv8n 가중치는 첫 실행 시 자동 다운로드됩니다.

## 실행

```bash
python app.py
# 브라우저: http://127.0.0.1:7860
```

옵션:
```bash
python app.py --port 8080
python app.py --share              # 외부 공유 링크
python app.py --host 0.0.0.0       # 네트워크 내 다른 기기 접근
```

## 사용 흐름

1. 영상 업로드 (1920×1080 이하, 30초 이하)
2. **top_k**: 제거할 객체 수 (보통 2)
3. **마스크 모드**: auto 권장 (GPU 있으면 SAM2)
4. **합성 모드 라디오**:
   - `fast (A만)` — 가장 빠름. 단일 결과
   - `AC 비교` — A + C 모드 1×2 비교 영상
   - `ABCD 비교` — 4가지 모드 2×2 비교 영상
5. **Velocity 비교** (옵션): merge 검증 OFF/ON 두 결과 비교
6. 실행

## 합성 모드

| 모드 | 동작 | 특징 |
|---|---|---|
| **A** | 다른 프레임 우선 (시간 무한대), plate fallback | 원본 v4. 안정 베이스 |
| **B** | mask 영역 = clean_plate | plate-only. 가장 단순 |
| **C** | 5프레임 안에서만 시도, plate fallback | 가까운 시점 픽셀 우선 |
| **D** | A 결과와 plate 50:50 평균 | A 자연스러움 + plate 안정성 |

## 디버그 인프라

UI에서 함께 출력:
- **preview**: 검출된 사람 빨강(REMOVE)/초록(KEEP) bbox
- **debug_video**: mask 단계별 시각화 (노랑=raw, 초록=refined, 빨강=shadow)
- **plate 이미지 2개**: inpaint 전/후 plate (KEEP이 박혀있는지 시각 확인)
- **debug_log**: 그룹 분석, Track 단위(velocity 포함), ID swap 의심 검출, 프레임별 픽셀 통계

## 핵심 가정

이 파이프라인은 다음 가정에서 잘 작동합니다:

- 카메라 고정
- 행인은 한 방향으로 길게 이동 (좌→우 또는 우→좌)
- KEEP은 거의 안 움직임 (손 정도 흔드는 수준 OK)

가정이 깨지면 (행인이 멈춤, KEEP이 움직임 등) 결과 품질이 떨어질 수 있습니다.

## 파일 구조

```
SceneEraser/
├── app.py              # Gradio UI
├── pipeline.py         # 영상 처리 파이프라인
├── checkpoints/        # SAM2 체크포인트 (.gitignore)
├── README.md
└── .gitignore
```

## 라이선스

(라이선스 명시하시면 추가)
>>>>>>> 3332d3d (Initial commit: SceneEraser pipeline + Gradio UI)
