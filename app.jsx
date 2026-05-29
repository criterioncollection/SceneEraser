/* global React, ReactDOM, SceneParticles */
const { useState, useEffect, useRef, useCallback } = React;

/* ───────────────────────────── Tweaks defaults ───────────────────────────── */
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#D14256",
  "particleMode": "default",
  "showOriginPanel": false
} /*EDITMODE-END*/;

/* ───────────────────────────────── Icons ─────────────────────────────────── */
const Icon = {
  Arrow: (p) =>
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" {...p}>
      <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>,

  Play: (p) =>
  <svg width="11" height="13" viewBox="0 0 11 13" fill="none" {...p}>
      <path d="M1 1l9 5.5L1 12V1z" fill="currentColor" />
    </svg>,

  Upload: (p) =>
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" {...p}>
      <path d="M11 15V3M6 8l5-5 5 5M3 17v2h16v-2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>,

  Close: (p) =>
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" {...p}>
      <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>,

  VideoCam: (p) =>
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" {...p}>
      <rect x="1" y="4" width="10" height="8" rx="1.2" stroke="currentColor" strokeWidth="1.2" />
      <path d="M11 7l4-2v6l-4-2V7z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
    </svg>,

  Image: (p) =>
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" {...p}>
      <rect x="1.5" y="2.5" width="13" height="11" rx="1.2" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="5.5" cy="6" r="1.2" stroke="currentColor" strokeWidth="1.2" />
      <path d="M2 12l3.5-3.5 3 3L11 8l3 3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
    </svg>,

  Download: (p) =>
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" {...p}>
      <path d="M7 1v8M4 6l3 3 3-3M2 12h10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>,

  Expand: (p) =>
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" {...p}>
      <path d="M2 5V2h3M12 5V2H9M2 9v3h3M12 9v3H9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>,

  Share: (p) =>
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" {...p}>
      <circle cx="3" cy="7" r="1.6" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="11" cy="3" r="1.6" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="11" cy="11" r="1.6" stroke="currentColor" strokeWidth="1.2" />
      <path d="M4.5 6L9.5 3.5M4.5 8L9.5 10.5" stroke="currentColor" strokeWidth="1.2" />
    </svg>

};

/* ────────────────────────────── Stage 1: Hero ────────────────────────────── */
function Hero({ onStart, onHow }) {
  return (
    <div className="stage hero">
      <div className="hero-grain" />
      <div className="hero-vignette" />

      <header className="topbar">
        <div className="wordmark">
          <span></span>
          <span className="sep"></span>
          <span className="muted"></span>
        </div>
        <div className="meta">
          <span className="chapter"></span>
          <span></span>
          <span></span>
        </div>
      </header>

      <div className="hero-inner">
        <div /> {/* spacer */}
        <div className="hero-body" style={{ padding: "0px 0px 20px" }}>
          <div className="hero-kicker">
            <span className="kicker-dot" />
            <span>AUTOMATIC OBJECT REMOVAL FOR VIDEO</span>
          </div>

          <h1 className="hero-title">
            <span className="line">Scene Eraser</span>
          </h1>

          <p className="hero-sub">
            영상 속 동적 객체를 자동 감지·제거하는 소프트웨어
            
            <span className="dim">YOLOv8n · ByteTrack · SAM 2.1 · temporal median plate</span>
          </p>

          <div className="hero-cta">
            <button className="pill primary" onClick={onStart}>
              <span>시연 시작</span>
              <Icon.Arrow />
            </button>
            <button className="pill ghost" onClick={onHow}>
              <span>작동 원리</span>
            </button>
          </div>
        </div>

        <div className="hero-footer">
          <div className="hero-footer-col right">
            <div className="label"></div>
            <div className="value muted">202021108 이재준 · 202322127 정채영</div>
          </div>
        </div>
      </div>
    </div>);

}

/* ─────────────────────────── Stage 1.5: How modal ────────────────────────── */
const PIPELINE_STEPS = [
['01', 'preprocess', '해상도·길이 정규화', '입력 영상을 최대 1920×1080 / 30s로 제한.'],
['02', 'detect · track', 'YOLOv8n + ByteTrack', '프레임 간 사람 검출 및 ID 추적, 오클루전으로 끊긴 트랙은 거리 우선으로 재연결.'],
['03', 'score · select', '6개 지표 가중합산', '이동 거리, 가장자리 통과율, 중앙 체류 비율 등으로 행인을 자동 선별 (top-k).'],
['04', 'mask', 'SAM 2.1  또는  bbox', 'GPU 환경에서는 SAM 2.1로 픽셀 단위, 없으면 bbox 모드.'],
['05', 'temporal plate', '시간축 중앙값 배경 추정', '인물을 비운 깨끗한 배경 plate를 시간축 통계로 생성.'],
['06', 'composite', 'A · B · C  세 가지 모드', '가까운 프레임 fallback / 다른 프레임 우선 / plate-only 중 선택.'],
['07', 'mux audio', '원본 오디오 보존', 'ffmpeg으로 원본 사운드를 결과물에 그대로 결합.']];


function HowModal({ open, onClose }) {
  return (
    <div className={`how-overlay ${open ? 'open' : ''}`} onClick={onClose}>
      <div className="how-modal" onClick={(e) => e.stopPropagation()}>
        <div className="how-head">
          <div className="how-kicker">
            <span className="kicker-dot" />
            <span>HOW IT WORKS</span>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="닫기"><Icon.Close /></button>
        </div>
        <h2 className="how-title">자동 감지에서<br /><em>배경 복원</em>까지, 7단계.</h2>
        <div className="how-steps">
          {PIPELINE_STEPS.map(([num, key, sub, desc]) =>
          <div className="how-step" key={num}>
              <div className="how-num">{num}</div>
              <div className="how-step-body">
                <div className="how-step-key">{key}</div>
                <div className="how-step-sub">{sub}</div>
                <div className="how-step-desc">{desc}</div>
              </div>
            </div>
          )}
        </div>
        <div className="how-foot">
          <div className="muted">고정 카메라 · 정지한 주인공 · 일정 방향의 행인이라는 가정에서 가장 좋은 결과.</div>
        </div>
      </div>
    </div>);

}

/* ───────────────────────── Stage 2: Tool / Workspace ─────────────────────── */
const PROC_STEPS = [
['사람 검출 및 추적', 'detect · track', 0, 1400],
['배경 plate 생성', 'temporal median', 1400, 2200],
['영상 합성', 'composite · mux', 3600, 1800]];

const PROC_TOTAL = PROC_STEPS.reduce((s, [,,, d]) => s + d, 0) + 400;

function Workspace({ onBack }) {
  const [file, setFile] = useState(null); // { name, sizeMB }
  const [topK, setTopK] = useState(2);
  const [maskMode, setMaskMode] = useState('sam2');
  const [compMode, setCompMode] = useState('A'); // A = 안정성 우선 (브리프)
  const [tightMask, setTightMask] = useState(false);
  const [keepShadow, setKeepShadow] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const [phase, setPhase] = useState('ready'); // ready | processing | done
  const [stepIdx, setStepIdx] = useState(0);
  const [progress, setProgress] = useState(0); // 0..1
  const fileObjRef = useRef(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // simulate processing
  useEffect(() => {
    if (phase !== 'processing') return;
    SceneParticles.setMode('process');
    const start = performance.now();
    let raf;
    const tick = (t) => {
      const elapsed = t - start;
      const p = Math.min(0.95, elapsed / PROC_TOTAL);
      setProgress(p);
      // active step
      let acc = 0;
      let idx = PROC_STEPS.length - 1;
      for (let i = 0; i < PROC_STEPS.length; i++) {
        const [,, offset, dur] = PROC_STEPS[i];
        if (elapsed >= offset && elapsed < offset + dur) {idx = i;break;}
        if (elapsed < offset) {idx = i - 1;break;}
      }
      setStepIdx(Math.max(0, idx));
      SceneParticles.setProgress(p);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { fileObjRef.current = f; setFile({ name: f.name, sizeMB: (f.size / (1024 * 1024)).toFixed(1), url: URL.createObjectURL(f) }); }
  }, []);

  const onPick = useCallback((e) => {
    const f = e.target.files && e.target.files[0];
    if (f) { fileObjRef.current = f; setFile({ name: f.name, sizeMB: (f.size / (1024 * 1024)).toFixed(1), url: URL.createObjectURL(f) }); }
  }, []);

  // demo: prefill with a sample
  const useSample = () => { fileObjRef.current = null; setFile({ name: 'museum_corridor_sample.mp4', sizeMB: '12.4', sample: true }); };

  const canRun = !!file && !file.sample && phase !== 'processing';

  const run = async () => {
    if (!fileObjRef.current) {
      setError('실제 영상 파일을 업로드해주세요 (샘플은 처리 불가).');
      return;
    }
    setError(null); setResult(null);
    setPhase('processing'); setProgress(0); setStepIdx(0);
    const fd = new FormData();
    fd.append('video', fileObjRef.current);
    fd.append('top_k', String(topK));
    fd.append('composite_mode', compMode);
    fd.append('mask_mode', maskMode);
    fd.append('tight_mask', String(tightMask));
    fd.append('shadow_off', String(keepShadow));
    try {
      const res = await fetch('/api/process', { method: 'POST', body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data); setProgress(1); setPhase('done');
      SceneParticles.setMode('idle'); SceneParticles.setProgress(1);
    } catch (e) {
      setError(e.message || '처리 중 오류가 발생했습니다.');
      setPhase('ready'); SceneParticles.setMode('idle');
    }
  };

  const reset = () => {
    setPhase('ready');
    setProgress(0);
    setStepIdx(0);
    setResult(null);
    setError(null);
    SceneParticles.setMode('idle');
  };

  // composite mode info (brief uses A = 안정성, B = 자연스러움)
  const compModes = [
  { id: 'A', label: '안정성 우선', sub: 'local 5-frame + plate fallback', dot: '기본' },
  { id: 'B', label: '자연스러움 우선', sub: '다른 프레임 픽셀 우선' },
  { id: 'C', label: 'plate only', sub: 'mask 영역 전부 배경 plate' }];


  return (
    <div className="stage workspace">
      <div className="ws-grain" />

      <header className="topbar">
        <button className="wordmark plain" onClick={onBack}>
          <span className="dot" />
          <span>SCENE ERASER</span>
          <span className="sep"></span>
          <span className="muted"></span>
        </button>
        <div className="ws-breadcrumb">
          <span className="muted">STAGE</span>
          <span className="step done">01 LANDING</span>
          <span className="step active">02 TOOL</span>
        </div>
        <button className="ws-back" onClick={onBack}>
          <span>← 처음으로</span>
        </button>
      </header>

      <div className="ws-grid">
        {/* LEFT — INPUT */}
        <div className="ws-col ws-left">
          <div className="col-head">
            <div className="kicker"><span className="kicker-dot" /> INPUT</div>
            <div className="hint">
            </div>
          </div>

          <div className="panel input-panel">
            <div className="panel-head">
              <div className="panel-title">
                <Icon.VideoCam />
                <span>입력 영상</span>
              </div>
              <div className="panel-actions">
                <button className="panel-action" aria-label="전체화면"><Icon.Expand /></button>
                <button className="panel-action"
                aria-label="샘플 불러오기"
                onClick={(e) => {e.preventDefault();useSample();}}>
                  <Icon.Download />
                </button>
              </div>
            </div>
            <label
              className={`drop ${dragOver ? 'over' : ''} ${file ? 'filled' : ''}`}
              onDragOver={(e) => {e.preventDefault();setDragOver(true);}}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}>
            
            <input type="file" accept="video/*" onChange={onPick} hidden />
            {!file ?
              <div className="drop-inner">
                <Icon.Upload />
                <div className="drop-title">비디오를 드래그하거나 클릭해 업로드</div>
                <div className="drop-sub">.mp4 · .mov · .webm</div>
                <button
                  type="button"
                  className="sample-link"
                  onClick={(e) => {e.preventDefault();useSample();}}>
                샘플 영상으로 보기 →</button>
              </div> :

              <div className="drop-filled">
                <video
                  className="file-video"
                  src={file.url}
                  controls
                  preload="metadata"
                  onClick={(e) => e.stopPropagation()} />
                <button
                  type="button"
                  className="file-clear-overlay"
                  onClick={(e) => {e.preventDefault();e.stopPropagation();if (file.url) URL.revokeObjectURL(file.url);setFile(null);reset();}}
                  aria-label="바꾸기"
                  title="바꾸기">
                  <Icon.Close />
                </button>
              </div>
              }
          </label>
          </div>

          {/* top-k */}
          <div className="control">
            <div className="control-head">
              <div className="control-label">제거할 객체 수 <span className="muted-sm"></span></div>
              <div className="control-value">{topK}</div>
            </div>
            <div className="slider">
              <div className="slider-track">
                <div className="slider-fill" style={{ width: `${(topK - 1) / 4 * 100}%` }} />
                {[1, 2, 3, 4, 5].map((n) =>
                <button
                  key={n}
                  className={`slider-tick ${n <= topK ? 'on' : ''} ${n === topK ? 'active' : ''}`}
                  style={{ left: `${(n - 1) / 4 * 100}%` }}
                  onClick={() => setTopK(n)} />

                )}
              </div>
              <div className="slider-axis">
                <span>1</span><span>2</span><span>3</span><span>4</span><span>5</span>
              </div>
            </div>
            <div className="hint"></div>
          </div>

          {/* composite mode */}
          <div className="control">
            <div className="control-head">
              <div className="control-label">합성 모드</div>
            </div>
            <div className="radio-stack">
              {compModes.map((m) =>
              <button
                key={m.id}
                className={`radio-row ${compMode === m.id ? 'on' : ''}`}
                onClick={() => setCompMode(m.id)}>
                
                  <span className="radio-dot" />
                  <span className="radio-id">{m.id}</span>
                  <span className="radio-body">
                    <span className="radio-label">{m.label}{m.dot && <span className="default-chip">{m.dot}</span>}</span>
                    <span className="radio-sub">{m.sub}</span>
                  </span>
                </button>
              )}
            </div>
          </div>

          {/* advanced */}
          <div className={`advanced ${showAdvanced ? 'open' : ''}`}>
            <button className="advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
              <span>고급 옵션</span>
              <span className="caret">{showAdvanced ? '−' : '+'}</span>
            </button>
            {showAdvanced &&
            <div className="advanced-body">
                <div className="adv-row">
                  <div className="adv-label">마스크 모드</div>
                  <div className="seg">
                    {['sam2', 'bbox', 'auto'].map((m) =>
                  <button
                    key={m}
                    className={`seg-btn ${maskMode === m ? 'on' : ''}`}
                    onClick={() => setMaskMode(m)}>
                    {m}</button>
                  )}
                  </div>
                </div>
                <div className="adv-toggle">
                  <button className={`check ${tightMask ? 'on' : ''}`} onClick={() => setTightMask((v) => !v)}>
                    <span className="check-box" />
                    <span>마스크 과확장 축소</span>
                  </button>
                  <span className="adv-hint">배경 침범 감소 (SAM2 권장)</span>
                </div>
                <div className="adv-toggle">
                  <button className={`check ${keepShadow ? 'on' : ''}`} onClick={() => setKeepShadow((v) => !v)}>
                    <span className="check-box" />
                    <span>그림자 안 지움</span>
                  </button>
                  <span className="adv-hint">사람만 제거, 그림자 보존</span>
                </div>
              </div>
            }
          </div>

          <button
            className={`pill primary run-btn ${!canRun ? 'disabled' : ''}`}
            onClick={canRun ? run : undefined}
            disabled={!canRun}>
            
            <Icon.Play />
            <span>{phase === 'processing' ? '처리 중…' : phase === 'done' ? '다시 실행' : '실행'}</span>
          </button>
        </div>

        {/* RIGHT — OUTPUT */}
        <div className="ws-col ws-right">
          <div className="col-head">
            <div className="kicker"><span className="kicker-dot" /> OUTPUT</div>
            <div className="hint">
            </div>
          </div>

          {/* output video */}
          <div className="panel video-panel">
            <div className="panel-head">
              <div className="panel-title">
                <Icon.VideoCam />
                <span>출력 영상 (객체 제거)</span>
              </div>
              <div className="panel-actions">
                <button className="panel-action" aria-label="다운로드"><Icon.Download /></button>
              </div>
            </div>
            <div className="video-stage">
              <VideoStage phase={phase} progress={progress} stepIdx={stepIdx} compMode={compMode} topK={topK} result={result} />
            </div>
          </div>

          {/* detection preview */}
          <div className="panel preview-panel">
            <div className="panel-head">
              <div className="panel-title">
                <Icon.Image />
                <span>감지 결과 (빨강=제거 / 초록=유지)</span>
              </div>
              <div className="panel-actions">
                <button className="panel-action" aria-label="전체화면"><Icon.Expand /></button>
                <button className="panel-action" aria-label="다운로드"><Icon.Download /></button>
                <button className="panel-action" aria-label="공유"><Icon.Share /></button>
              </div>
            </div>
            <div className="preview-canvas">
              <DetectPreview phase={phase} topK={topK} progress={progress} result={result} />
            </div>
          </div>

          {/* status */}
          <div className="status">
            <span className="status-text">
              {phase === 'ready' && '\n'}
              {phase === 'processing' && <>
                  <strong>{PROC_STEPS[stepIdx][0]}</strong>
                  <span className="muted"> &nbsp;·&nbsp; {PROC_STEPS[stepIdx][1]} &nbsp;·&nbsp; {Math.round(progress * 100)}%</span>
                </>
              }
              {phase === 'done' && !error && result &&
              <>감지된 그룹: <strong>{result.n_groups}</strong>개 &nbsp;·&nbsp; 제거 대상: <strong>[{(result.remove_ids||[]).join(', ')}]</strong> &nbsp;·&nbsp; 모드: <strong>{result.composite_mode}</strong></>
              }
              {error && <span style={{color:'#FF6B6B'}}>오류: {error}</span>}
            </span>
          </div>
        </div>
      </div>

      {phase === 'processing' &&
      <ProcessingOverlay stepIdx={stepIdx} progress={progress} />
      }
    </div>);

}

/* ───────────────────────── detection preview canvas ──────────────────────── */
function DetectPreview({ phase, topK, progress, result }) {
  if (phase === 'ready') {
    return (
      <div className="dp dp-empty">
        <div className="dp-icon">
          <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
            <rect x="3" y="6" width="38" height="32" rx="2" stroke="currentColor" strokeWidth="1.4" />
            <circle cx="14" cy="17" r="3" stroke="currentColor" strokeWidth="1.4" />
            <path d="M5 33l10-10 8 8 6-6 10 10" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
          </svg>
        </div>
      </div>);

  }

  if (phase === 'done' && result && result.preview) {
    return (
      <div className="dp dp-real">
        <img src={result.preview} alt="감지 결과"
          style={{width:'100%',height:'100%',objectFit:'contain',borderRadius:'inherit'}} />
      </div>);
  }

  // Stylised scene with N bbox figures, R=remove (top-k), G=keep
  // Position figures across the frame.
  const figures = [
  { x: 0.10, y: 0.45, w: 0.10, h: 0.45, id: 1, kind: 'remove' },
  { x: 0.28, y: 0.48, w: 0.10, h: 0.42, id: 2, kind: 'remove' },
  { x: 0.46, y: 0.42, w: 0.13, h: 0.52, id: 3, kind: 'keep' },
  { x: 0.66, y: 0.50, w: 0.10, h: 0.40, id: 4, kind: 'remove' },
  { x: 0.82, y: 0.52, w: 0.09, h: 0.38, id: 5, kind: 'remove' }];

  // Recompute remove based on top-k: keep the centre (id=3), top-k removes from the rest by distance from centre.
  const ranked = figures.
  filter((f) => f.id !== 3).
  sort((a, b) => Math.abs(a.x + a.w / 2 - 0.5) - Math.abs(b.x + b.w / 2 - 0.5)).
  slice(0, topK).
  map((f) => f.id);
  const removeIds = new Set(ranked);

  return (
    <div className="dp">
      <div className="dp-bg" />
    </div>);

}

/* ──────────────────────────── output video stage ─────────────────────────── */
function VideoStage({ phase, progress, stepIdx, compMode, topK, result }) {
  if (phase === 'ready') {
    return (
      <div className="vs vs-empty">
        <div className="vs-icon">
          <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
            <rect x="3" y="9" width="38" height="26" rx="2" stroke="currentColor" strokeWidth="1.4" />
            <path d="M18 16l10 6-10 6v-12z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
          </svg>
        </div>
      </div>);

  }

  if (phase === 'processing') {
    return (
      <div className="vs vs-proc">
        <div className="vs-scan" />
        <div className="vs-proc-pulse" />
        <div className="vs-proc-text">
          <div className="vs-proc-step">{`STEP ${(stepIdx + 1).toString().padStart(2, '0')} / 03`}</div>
          <div className="vs-proc-name">{PROC_STEPS[stepIdx][0]}</div>
          <div className="vs-proc-key">{PROC_STEPS[stepIdx][1]}</div>
        </div>
        <div className="vs-proc-bar">
          <div className="vs-proc-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>);

  }

  if (result && result.output_video) {
    return (
      <div className="vs vs-done vs-real">
        <video src={result.output_video} controls autoPlay loop
          style={{width:'100%',height:'100%',objectFit:'contain',background:'#000',borderRadius:'inherit',display:'block'}} />
      </div>);
  }
  return (
    <div className="vs vs-done">
      <div className="vs-bg" />
    </div>);

}

/* ─────────────────────── processing overlay (subtle) ─────────────────────── */
function ProcessingOverlay({ stepIdx, progress }) {
  return (
    <div className="proc-overlay">
      <div className="proc-card">
        <div className="proc-kicker">
          <span className="kicker-dot pulse" />
          <span>PROCESSING</span>
        </div>
        <div className="proc-step-row">
          {PROC_STEPS.map((s, i) =>
          <div key={i} className={`proc-step ${i < stepIdx ? 'done' : ''} ${i === stepIdx ? 'active' : ''}`}>
              <div className="proc-step-num">{`0${i + 1}`}</div>
              <div className="proc-step-name">{s[0]}</div>
              <div className="proc-step-sub">{s[1]}</div>
            </div>
          )}
        </div>
        <div className="proc-bar">
          <div className="proc-bar-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    </div>);

}

/* ──────────────────────────────── Tweaks UI ──────────────────────────────── */
function TweakControls({ tweaks, setTweak }) {
  return (
    <>
      <TweakSection title="Accent" subtitle="단일 액센트 컬러">
        <TweakColor
          label="액센트"
          value={tweaks.accent}
          onChange={(v) => setTweak('accent', v)}
          options={['#D67272', '#D14256', '#C25862', '#B53C4D']} />
        
      </TweakSection>
      <TweakSection title="Particles" subtitle="히어로 파동 강도">
        <TweakRadio
          label="모드"
          value={tweaks.particleMode}
          onChange={(v) => setTweak('particleMode', v)}
          options={[
          { value: 'default', label: '기본' },
          { value: 'calm', label: '잔잔' },
          { value: 'intense', label: '격렬' }]
          } />
        
      </TweakSection>
    </>);

}

/* ───────────────────────────────── App root ──────────────────────────────── */
function App() {
  const [stage, setStage] = useState('hero'); // hero | tool
  const [howOpen, setHowOpen] = useState(false);
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);

  // wire tweaks → particles
  useEffect(() => {SceneParticles.setAccent(tweaks.accent);}, [tweaks.accent]);
  useEffect(() => {
    // we just toggle dissolve speed via mode for now
    if (stage === 'hero') SceneParticles.setMode('hero');
  }, [tweaks.particleMode, stage]);

  // Lock the hero to a 1536×960 design canvas and scale-to-fit so every
  // viewport sees identical proportions (margins, type, dot positions, etc.)
  useEffect(() => {
    const DESIGN_W = 1536,DESIGN_H = 960;
    const update = () => {
      const s = Math.min(window.innerWidth / DESIGN_W, window.innerHeight / DESIGN_H);
      document.documentElement.style.setProperty('--hero-scale', s);
      document.documentElement.style.setProperty('--design-w', DESIGN_W);
      document.documentElement.style.setProperty('--design-h', DESIGN_H);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // root accent for CSS variables
  useEffect(() => {
    document.documentElement.style.setProperty('--maroon', tweaks.accent);
  }, [tweaks.accent]);

  const goTool = () => {
    setStage('tool');
    SceneParticles.setMode('idle');
  };
  const goHero = () => {
    setStage('hero');
    SceneParticles.setMode('hero');
  };

  return (
    <>
      {stage === 'hero' ?
      <Hero
        onStart={goTool}
        onHow={() => setHowOpen(true)} /> :


      <Workspace onBack={goHero} />
      }
      <HowModal open={howOpen} onClose={() => setHowOpen(false)} />

      <TweaksPanel title="Tweaks">
        <TweakControls tweaks={tweaks} setTweak={setTweak} />
      </TweaksPanel>
    </>);

}

// init immediately — babel-transformed scripts may run after 'load' has fired
(function bootstrap() {
  const canvas = document.getElementById('particle-canvas');
  if (canvas && window.SceneParticles) {
    SceneParticles.init({ canvas, accent: TWEAK_DEFAULTS.accent });
    window.addEventListener('resize', () => SceneParticles.resize());
  }
  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
})();