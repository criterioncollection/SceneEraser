/**
 * SceneEraser — Waveform Eraser (AM cover-style).
 * A horizontal audio-waveform sits in a tight rectangular band. An "erase
 * front" sweeps across it; bars beyond the front shrink and dissolve into
 * small drifting embers (the erased pedestrian's leftovers).
 *
 * Canvas 2D only — easier to fine-tune than WebGL for this aesthetic.
 *
 * Public API on window.SceneParticles:
 *   init({ canvas, accent })
 *   setAccent(hex)
 *   setMode('hero' | 'process' | 'idle')
 *   setProgress(0..1)
 *   resize()
 *   dispose()
 */
(function (global) {
  function hexToRgb(hex) {
    const m = hex.replace('#', '');
    return [
      parseInt(m.slice(0, 2), 16),
      parseInt(m.slice(2, 4), 16),
      parseInt(m.slice(4, 6), 16),
    ];
  }

  // Deterministic per-bar "amplitude character" so the waveform feels like a
  // real recording rather than a sine pattern. Sum of three sin layers + jitter.
  function barShape(xNorm, jitterSeed) {
    // low-freq envelope (the AM-style overall sweep)
    const env =
      Math.pow(Math.sin(xNorm * Math.PI * 2.1 + 0.4), 2) * 0.55 +
      Math.pow(Math.sin(xNorm * Math.PI * 1.1 + 1.7), 2) * 0.45;
    // mid-freq detail
    const mid =
      Math.sin(xNorm * 38.0 + 0.6) * 0.5 +
      Math.sin(xNorm * 17.0 + 1.9) * 0.5;
    // high-freq jitter from seed
    const j =
      Math.sin(xNorm * 250 + jitterSeed * 6.28) * 0.5;
    // combine — final in 0.05..1.0
    return Math.max(0.05, Math.min(1.0, 0.32 * env + 0.45 * Math.abs(mid) + 0.18 * Math.abs(j) + 0.08));
  }

  class WaveformEraser {
    constructor() {
      this.mode = 'hero';
      this.progress = 0;
      this.accent = [205, 92, 92];
      this.time = 0;
      this.eraseProgress = 0;
      this.barCount = 220;
      this.embers = [];      // pool of drifting embers near the erase front
      this.lastSpawnT = 0;
    }

    init(opts = {}) {
      this.canvas = opts.canvas;
      this.ctx = this.canvas.getContext('2d');
      this.accent = hexToRgb(opts.accent || '#CD5C5C');
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.resize();
      this._loop = this._loop.bind(this);
      this._rafId = requestAnimationFrame(this._loop);
    }

    setAccent(hex) { this.accent = hexToRgb(hex); }
    setMode(mode) { this.mode = mode; }
    setProgress(p) { this.progress = Math.max(0, Math.min(1, p)); }

    resize() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      this.w = w; this.h = h;
      this.canvas.width  = Math.floor(w * this.dpr);
      this.canvas.height = Math.floor(h * this.dpr);
      this.canvas.style.width  = w + 'px';
      this.canvas.style.height = h + 'px';
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }

    _spawnEmbers(eraseFront, baseY, bandHalf, intensity) {
      // a few embers per second from the erase front, drifting up
      const want = Math.max(1, Math.floor(intensity * 3));
      for (let i = 0; i < want; i++) {
        const jitter = (Math.random() - 0.5) * 0.012; // bandWidth-normalised
        this.embers.push({
          xNorm: eraseFront + jitter,
          y: baseY + (Math.random() - 0.5) * bandHalf * 1.4,
          vx: (Math.random() - 0.5) * 0.12,             // horizontal drift
          vy: -10 - Math.random() * 22,                  // upward
          life: 1.0,
          decay: 0.4 + Math.random() * 0.6,
          size: 1 + Math.random() * 1.6,
        });
      }
    }

    _draw() {
      const ctx = this.ctx;
      const w = this.w, h = this.h;

      // Locked 1536×960 design canvas centred in viewport (matches the
      // .hero-inner CSS scale-to-fit). Everything draws in design coords
      // then is scaled & offset to land where the DOM expects it.
      const DESIGN_W = 1536, DESIGN_H = 960;
      const scale = Math.min(w / DESIGN_W, h / DESIGN_H);
      const dW = DESIGN_W * scale;
      const dH = DESIGN_H * scale;
      const dx = (w - dW) / 2;
      const dy = (h - dH) / 2;

      const bandPadX = dW * 0.08;
      const bandLeft  = dx + bandPadX;
      const bandRight = dx + dW - bandPadX;
      const bandW     = bandRight - bandLeft;
      const bandY     = dy + dH * 0.32;
      const amplitude = dH * 0.16;

      ctx.clearRect(0, 0, w, h);

      // axis whisper
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bandLeft, bandY);
      ctx.lineTo(bandRight, bandY);
      ctx.stroke();

      // drawing progress: 0 = nothing drawn, 1 = full wave drawn
      let progress;
      if (this.mode === 'hero') {
        const period = 22.0;   // ~3.3x slower
        const t = (this.time % period) / period;
        if (t < 0.75) {
          const raw = t / 0.75;
          progress = 1 - Math.pow(1 - raw, 2.2);
        } else if (t < 0.90) {
          progress = 1.0;
        } else {
          progress = 1.0;
        }
        this._fade = t < 0.90 ? 1.0 : 1.0 - (t - 0.90) / 0.10;
      } else if (this.mode === 'process') {
        const period = 9.0;
        progress = ((this.time % period) / period);
        this._fade = 1.0;
      } else { // idle
        progress = 1.0;
        this._fade = 0.7;
      }
      progress = Math.max(0, Math.min(1, progress));

      // ── trace the wave from bandLeft up to (bandLeft + bandW * progress) ──
      // Composed sine wave: two octaves whose sum keeps a near-uniform peak
      // amplitude (so crests and troughs sit at ±amplitude).
      const segments = 480;                  // smoothness of the polyline
      const headFrac = progress;
      const headX = bandLeft + bandW * headFrac;
      const drawnSegments = Math.max(2, Math.floor(segments * headFrac));

      const a = this.accent;
      const accentRGB = `${a[0]}, ${a[1]}, ${a[2]}`;
      const fade = this._fade;

      // Pure single-frequency sine — each cycle is a perfect S (point-symmetric
       // around its zero crossings, crests and troughs at equal amplitude).
      const k1 = 5.5;
      const v1 = 0.05;
      const t = this.time;

      const waveY = (xNorm) => {
        const phase = xNorm * Math.PI * 2;
        return amplitude * Math.sin(phase * k1 + t * v1);
      };

      // helper: stroke the currently-drawn portion of the wave with a given style
      const tracePath = () => {
        ctx.beginPath();
        for (let i = 0; i <= drawnSegments; i++) {
          const xNorm = (i / segments);
          const x = bandLeft + xNorm * bandW;
          const y = bandY + waveY(xNorm);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      };

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // outer halo glow — visible red bloom
      ctx.strokeStyle = `rgba(${accentRGB}, ${fade * 0.22})`;
      ctx.lineWidth = 48 * scale;
      tracePath();

      // soft maroon halo — graduated red fade
      ctx.strokeStyle = `rgba(${accentRGB}, ${fade * 0.40})`;
      ctx.lineWidth = 32 * scale;
      tracePath();

      // maroon outline — saturated red, very thin against white
      ctx.strokeStyle = `rgba(${accentRGB}, ${fade * 0.95})`;
      ctx.lineWidth = 22 * scale;
      tracePath();

      // white inner core — thinner overall
      ctx.strokeStyle = `rgba(244, 238, 229, ${fade * 0.85})`;
      ctx.lineWidth = 17 * scale;
      tracePath();

      // bright white spine
      ctx.strokeStyle = `rgba(255, 255, 255, ${fade * 0.98})`;
      ctx.lineWidth = 10 * scale;
      tracePath();

      // leading tip — small bright dot only when actively drawing
      if (progress > 0.005 && progress < 0.998 && this.mode !== 'idle') {
        const headXNorm = headFrac;
        const hy = bandY + waveY(headXNorm);
        const halo = ctx.createRadialGradient(headX, hy, 0, headX, hy, 50);
        halo.addColorStop(0,   `rgba(255, 255, 255, ${fade * 0.9})`);
        halo.addColorStop(0.35, `rgba(${accentRGB}, ${fade * 0.6})`);
        halo.addColorStop(1,   `rgba(${accentRGB}, 0)`);
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(headX, hy, 50, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = `rgba(255, 255, 255, ${fade})`;
        ctx.beginPath();
        ctx.arc(headX, hy, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    _loop(ts) {
      this._rafId = requestAnimationFrame(this._loop);
      const dt = this._lastT ? Math.min(0.05, (ts - this._lastT) / 1000) : 0.016;
      this._lastT = ts;
      this._dt = dt;
      this.time += dt;
      this._draw();
    }

    dispose() {
      cancelAnimationFrame(this._rafId);
    }
  }

  global.SceneParticles = new WaveformEraser();
})(window);
