/* VoxEMW 2D Live 数字人引擎模块。
 *
 * 核心从 yukino2d 模型的独立页（yukino Vtuber live 2d/vtuber_model/index.html）
 * v2 移植：三部件伪 Live2D 差分渲染 v2（去 UI/麦克风/调试/背景）：
 *   base.png   基础立绘（透明 PNG），整张绘制（v2 去掉头/身拆分）
 *   blink.png  闭眼差分（全图，按 eyeRect 裁剪 + 羽化掩膜，alpha 控制闭眼）
 *   mouth.png  张嘴差分（全图，按 mouthRect 裁剪，随口型缩放叠加）
 * 立绘经同源 HTTP 加载（/yukino2d/parts/*.png，orchestrator 静态路由托管）；
 * 口型由外部 setMouth 驱动（assistant.js 用 TTS 播放链路 AnalyserNode 实时
 * RMS → OpennessTracker 映射，参数沿用 viseme 校准值）。眨眼/呼吸/身体飘动
 * 完全引擎自带，待机时自然持续；说话能量（talkEnergy）驱动飘动增强与摇摆。
 *
 * API（window 暴露，不污染其他全局）：
 *   initYukino2D(canvas, {parts}) → { setMouth(v), isReady(), destroy() }
 *     parts 可覆盖三部件 URL，默认 /yukino2d/parts/ 下
 *   createOpennessTracker(opts)  → { step(rms) -> 0..1 }
 *
 * 与独立版引擎 v2 的差异：
 *   ①嘴目标恒取外部 setMouth（原 mic/sim/space 三分支已删），talkEnergy 由
 *     该目标驱动（原由 RMS target 驱动，语义一致）；
 *   ②整体跟随恒关（数字人场景无鼠标交互），保留轻微自动漂移待机微动；
 *   ③口型/talkEnergy 平滑改为帧率无关指数逼近（原逐帧系数），其余逐字对齐；
 *   ④canvas 自适应容器尺寸（clientWidth/Height × DPR），模型等比 fit。
 */
(function () {
  "use strict";

  /* ---------------- 可调配准参数（归一化坐标，相对图片宽高） ----------------
   * 与 yukino Vtuber live 2d/vtuber_model/index.html v2 CFG 逐字一致。 */
  const CFG = {
    parts: {
      base: "/yukino2d/parts/base.png",
      blink: "/yukino2d/parts/blink.png",
      mouth: "/yukino2d/parts/mouth.png",
    },
    leanPivot: { x: 0.500, y: 0.985 },                       // 整体摆动支点（脚底）
    eyeRect: { x: 0.402, y: 0.270, w: 0.210, h: 0.080, srcDy: 18 }, // 双眼区域（闭眼差分，srcDy=源偏移px）
    mouthRect: { x: 0.462, y: 0.368, w: 0.072, h: 0.040, srcDy: 0 },  // 嘴部区域（张嘴差分，紧贴嘴部不含下巴）
    swayStartY: 0.55,                          // 此高度以下做飘动（发梢/裙摆/尾巴）
    follow: { x: 10, y: 6, rot: 0.028 },       // 整体跟随幅度（px / rad）
    mouth: { maxOpen: 0.72,                    // 口型开合上限（防止张太大）
             baseSX: 0.80, growX: 0.20,        // 贴片横向缩放：base + grow*open
             baseSY: 0.28, growY: 0.72 },      // 贴片纵向缩放（开口度，最大≈0.80）
  };

  /* RMS→开合度参数：voxemw/avatar/viseme_core.py OpennessTracker 校准值 */
  const RMS_PARAMS = {
    floor: 0.003,     // ~-50 dBFS
    ceil: 0.25,       // ~-12 dBFS
    curve: 1.3,       // 弱音不张嘴、强音不开满
    openMin: 0.05,    // 开合度下限（静音近闭不僵死）
    openMax: 0.95,    // 开合度上限
    attack: 0.45,     // 开口速度（每帧）
    release: 0.18,    // 闭嘴惯性（每帧）
  };

  /* ---------------- 工具 ---------------- */
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

  function loadImage(src) {
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => rej(new Error("加载失败: " + src));
      im.src = src;
    });
  }

  /* ---------------- OpennessTracker：viseme_core.py step 的 JS 直译 ----------------
   * 输入 rms（浮点域，静音≈0），输出开合度 0..1（EMA 平滑）。 */
  function OpennessTracker(opts) {
    const p = Object.assign({}, RMS_PARAMS, opts || {});
    this.floor = p.floor; this.ceil = p.ceil; this.curve = p.curve;
    this.openMin = p.openMin; this.openMax = p.openMax;
    this.attack = p.attack; this.release = p.release;
    this.value = p.openMin;              // 初始即"静音闭口"态
  }
  OpennessTracker.prototype.step = function (rms) {
    let level = 0;
    if (rms > this.floor) {
      level = Math.log10(rms / this.floor) / Math.log10(this.ceil / this.floor);
      level = Math.max(0, Math.min(1, level));
      level = Math.pow(level, this.curve);
    }
    const target = this.openMin + (this.openMax - this.openMin) * level;
    const alpha = target > this.value ? this.attack : this.release;
    this.value = alpha * target + (1 - alpha) * this.value;
    return this.value;
  };

  /* ---------------- 图片预处理（与独立版 v2 逐字一致） ---------------- */
  /* 立绘已离线去背（透明 PNG），直接绘制即可 */
  function processImage(im) {
    const c = document.createElement("canvas");
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    c.getContext("2d").drawImage(im, 0, 0);
    return c;
  }

  /* 差分小贴片：实心核心 + 四边 smoothstep 羽化，用于闭眼/张嘴叠加 */
  function edgeMask(w, h, fx, fy) {
    const m = document.createElement("canvas"); m.width = w; m.height = h;
    const g = m.getContext("2d");
    const img = g.createImageData(w, h), d = img.data;
    const rx = fx * w, ry = fy * h;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const ax = Math.max(0, Math.min(1, Math.min(x, w - 1 - x) / rx));
      const ay = Math.max(0, Math.min(1, Math.min(y, h - 1 - y) / ry));
      let a = Math.min(ax, ay); a = a * a * (3 - 2 * a);
      const o = (y * w + x) * 4;
      d[o] = d[o + 1] = d[o + 2] = 255; d[o + 3] = Math.round(a * 255);
    }
    g.putImageData(img, 0, 0);
    return m;
  }

  function makeSprite(src, rect) {
    const x = rect.x * src.width, y = rect.y * src.height;
    const w = Math.round(rect.w * src.width), h = Math.round(rect.h * src.height);
    const dy = rect.srcDy || 0;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d");
    g.drawImage(src, x, y + dy, w, h, 0, 0, w, h);
    g.globalCompositeOperation = "destination-in";
    g.drawImage(edgeMask(w, h, 0.07, 0.09), 0, 0);
    return { c, x, y };
  }

  /* ---------------- 引擎实例 ---------------- */
  function createYukino2D(canvas, opts) {
    const parts = Object.assign({}, CFG.parts, (opts && opts.parts) || {});
    const state = { leanX: 0, leanY: 0, leanRot: 0, blink: 0, mouth: 0, talkEnergy: 0 };
    let mouthTarget = 0;                 // setMouth 写入的外部目标
    let rafId = 0, ready = false, destroyed = false;
    let fullC, blinkSpr, mouthSpr, imgW = 0, imgH = 0;

    /* 眨眼状态机（与独立版 v2 逐字一致，dt 为 ms） */
    let blinkTimer = 0, nextBlink = 2500, blinkPhase = 0;
    function blinkUpdate(dt) {
      blinkTimer += dt;
      if (blinkPhase === 0 && blinkTimer > nextBlink) { blinkPhase = 1; blinkTimer = 0; }
      else if (blinkPhase === 1) {
        state.blink = Math.min(1, blinkTimer / 70);
        if (state.blink >= 1) { blinkPhase = 2; blinkTimer = 0; }
      } else if (blinkPhase === 2) {
        if (blinkTimer > 50) { blinkPhase = 3; blinkTimer = 0; }
      } else if (blinkPhase === 3) {
        state.blink = Math.max(0, 1 - blinkTimer / 130);
        if (state.blink <= 0) {
          blinkPhase = 0; blinkTimer = 0;
          nextBlink = Math.random() < 0.15 ? 260 : 1800 + Math.random() * 3800;
        }
      }
    }

    /* 主循环参数更新：嘴目标恒取外部 setMouth（原 mic/sim/space 三分支已删），
     * 眨眼/呼吸/身体飘动完全自带；整体做轻微自动漂移（无鼠标跟随，等效独立版
     * 的"空闲 4 秒后进入自动漂移"）；talkEnergy 驱动说话时的飘动增强与摇摆 */
    function updateParams(t, dt) {
      blinkUpdate(dt);
      const nx = Math.sin(t * 0.0006) * 0.35 + Math.sin(t * 0.0013) * 0.15;
      const ny = Math.cos(t * 0.0008) * 0.22;
      const k = 1 - Math.pow(0.0001, dt / 1000);   // 帧率无关平滑
      state.leanX   += (nx * CFG.follow.x  - state.leanX)   * k;
      state.leanY   += (ny * CFG.follow.y  - state.leanY)   * k;
      state.leanRot += (nx * CFG.follow.rot - state.leanRot) * k;
      /* 口型：原逐帧系数 0.55/0.22 换算为帧率无关指数逼近（≈60fps 等价），
       * maxOpen 钳制上限；talkEnergy 同样换算（原 0.08/0.03） */
      const rate = mouthTarget > state.mouth ? 0.55 : 0.22;
      state.mouth += (mouthTarget - state.mouth) * (1 - Math.exp(-dt / 1000 * rate * 60));
      state.mouth = Math.min(state.mouth, CFG.mouth.maxOpen);
      const eRate = mouthTarget > state.talkEnergy ? 0.05 : 0.02;
      state.talkEnergy += (Math.min(mouthTarget, 0.6) - state.talkEnergy)
                          * (1 - Math.exp(-dt / 1000 * eRate * 60));
    }

    function render(ctx, t) {
      /* canvas 自适应容器尺寸（CSS 像素绘制，DPR 缩放） */
      const cw = canvas.clientWidth || 640, ch = canvas.clientHeight || 480;
      const DPR = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(cw * DPR) || canvas.height !== Math.round(ch * DPR)) {
        canvas.width = Math.round(cw * DPR); canvas.height = Math.round(ch * DPR);
      }
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      if (!ready) return;

      const W = cw, H = ch;
      const scale = Math.min(H * 0.96 / imgH, W * 0.92 / imgW);
      const ox = (W - imgW * scale) / 2, oy = (H - imgH * scale) / 2;
      const breath = Math.sin(t * 0.0016);
      const swayT = t * 0.001;
      const energy = state.talkEnergy;               // 说话能量 0~0.6
      const swayGain = 1 + energy * 1.6;             // 说话时摆动增强
      const rock = energy * 0.010 * Math.sin(t * 0.004);  // 说话时的身体摇摆（低频≈0.6Hz）

      ctx.save();
      ctx.translate(ox, oy); ctx.scale(scale, scale);

      /* 整体跟随变换：绕脚底支点的轻微旋转 + 平移 */
      const px = CFG.leanPivot.x * imgW, py = CFG.leanPivot.y * imgH;
      ctx.translate(px + state.leanX, py + state.leanY + breath * 1.2);
      ctx.rotate(state.leanRot + rock);
      ctx.translate(-px, -py);

      /* 主体：上半整块绘制，下半逐条横向飘动 */
      const swayY = CFG.swayStartY * imgH;
      ctx.drawImage(fullC, 0, 0, imgW, swayY, 0, 0, imgW, swayY);
      const strip = 8;
      for (let y = swayY; y < imgH; y += strip) {
        const p = (y - swayY) / (imgH - swayY);
        const dx = (Math.sin(swayT * 1.7 + y * 0.012) * 3.5 +
                    Math.sin(swayT * 0.9 + y * 0.030) * 2.0) * p * p * swayGain;
        const sh = Math.min(strip, imgH - y);
        ctx.drawImage(fullC, 0, y, imgW, sh, dx, y, imgW, sh);
      }

      /* 闭眼差分叠加（眨眼） */
      if (state.blink > 0.01) {
        ctx.globalAlpha = state.blink;
        ctx.drawImage(blinkSpr.c, blinkSpr.x, blinkSpr.y);
        ctx.globalAlpha = 1;
      }
      /* 张嘴差分叠加（口型同步）：快速升至不透明以完全盖住底层原嘴，仅靠缩放表现开合。
       * moEff 减去 OpennessTracker 的 openMin 下限：VoxEMW 链路静音时 mouth≈0.05，
       * 若不偏移，贴片会在静音时以 ~25% 透明度缩小的嘴叠在原嘴上（视觉双嘴），
       * 与独立版（静音 target=0，不画贴片）行为对齐；maxOpen 钳制后最大
       * mouth=0.72 → moEff≈0.71，贴片缩放上限与作者参数设计一致。 */
      const mo = state.mouth;
      const moEff = clamp((mo - 0.05) / 0.95, 0, 1);
      if (moEff > 0.01) {
        const cx = mouthSpr.x + mouthSpr.c.width / 2;
        const cy = mouthSpr.y + mouthSpr.c.height / 2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.scale(CFG.mouth.baseSX + CFG.mouth.growX * moEff,
                  CFG.mouth.baseSY + CFG.mouth.growY * moEff);
        ctx.globalAlpha = Math.min(1, moEff * 5);
        ctx.drawImage(mouthSpr.c, -mouthSpr.c.width / 2, -mouthSpr.c.height / 2);
        ctx.restore();
        ctx.globalAlpha = 1;
      }
      ctx.restore();
    }

    /* 启动：HTTP 加载三部件（同源，Canvas 2D 不受跨域污染）→ 裁贴片 → rAF 循环 */
    (async function boot() {
      try {
        const ctx = canvas.getContext("2d", { alpha: true });
        const [imB, imE, imM] = await Promise.all([
          loadImage(parts.base), loadImage(parts.blink), loadImage(parts.mouth)]);
        fullC = processImage(imB);
        const blink = processImage(imE), mouth = processImage(imM);
        imgW = fullC.width; imgH = fullC.height;
        blinkSpr = makeSprite(blink, CFG.eyeRect);
        mouthSpr = makeSprite(mouth, CFG.mouthRect);
        ready = true;
        let prevT = 0;
        const loop = (now) => {
          if (destroyed) return;
          rafId = requestAnimationFrame(loop);
          const dt = Math.min(50, now - prevT || 16);   // 切后台恢复时钳制防跳变
          prevT = now;
          updateParams(now, dt);
          render(ctx, now);
        };
        rafId = requestAnimationFrame(loop);
      } catch (e) {
        console.error("[yukino2d] 引擎启动失败:", e);
      }
    })();

    return {
      /* 目标开合度 0..1（外部 OpennessTracker 输出直接喂） */
      setMouth(v) { mouthTarget = clamp(v, 0, 1); },
      isReady() { return ready; },
      destroy() { destroyed = true; if (rafId) cancelAnimationFrame(rafId); },
    };
  }

  window.initYukino2D = createYukino2D;
  window.createOpennessTracker = (opts) => new OpennessTracker(opts);
})();
