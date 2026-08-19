/* VoxEMW 2D Live 数字人引擎模块 — Live2D Cubism 4 版（雪之下雪乃）。
 *
 * 模型来源：春物角色 live2d-win/雪乃 live2d win 版（本地个人预览用）。
 * 渲染链路：pixi.js v6 + pixi-live2d-display（PIXI.live2d）+ Cubism 4 Core，
 * 运行时脚本在 web/index.html 中按序加载：
 *   /static/live2d/runtime/pixi.min.js
 *   /static/live2d/runtime/live2dcubismcore.min.js
 *   /static/live2d/runtime/cubism4.min.js
 *
 * API（与旧 yukino2d 三部件引擎一致）：
 *   initYukino2D(canvas, {model}) → {
 *     setMouth(v), isReady(), destroy(),
 *     playMotion(name), setExpression(name), setPose(name),
 *     setFollow(on),      // 鼠标目光跟随（focusController，默认开；仅画布内跟随）
 *     setAutoPose(on),    // 待机姿势自动轮换（手动选姿态后关）
 *     talkDemo(sec),      // 模拟说话口型（无需音频）
 *     playChoreography(steps),  // 动作编排：按作者预置时长顺序播 表情/动作/姿势/停顿
 *     listMotions(), listExpressions(),
 *   }
 *     model 可覆盖 model3.json URL；默认制服 yukino_seihuku，?outfit=shihuku 用私服
 *   createOpennessTracker(opts)  → { step(rms) -> 0..1 }
 *
 * 口型：外部 setMouth 喂 0..1 开合度（TTS 播放链 RMS → OpennessTracker），
 * 在模型 beforeModelUpdate 钩子里写入 PARAM_MOUTH_OPEN_Y；眨眼/呼吸/待机
 * Idle motion 由 Live2D 模型自身驱动，无需自研差分。
 */
(function () {
  "use strict";

  /* 默认模型：?outfit=shihuku 可切换私服（私服=shihuku 为官方拼写）。
   * ?v= 缓存版本号：模型文件改名/更新后浏览器强制拉新 model3.json，
   * 否则旧缓存仍引用旧 motion 名 → 404（2026-08-19 事故）。 */
  const DEFAULT_MODEL_URL = (() => {
    const outfit = new URLSearchParams(location.search).get("outfit");
    return (outfit === "shihuku"
      ? "/static/live2d/models/yukino/yukino_shihuku.model3.json"
      : "/static/live2d/models/yukino/yukino_seihuku.model3.json") + "?v=20260819e";
  })();

  /* 默认姿势：?pose=pose1|pose2|pose3 或 poseA|poseB|poseC 指定初始 Idle 姿势 */
  const DEFAULT_POSE = (() => {
    const p = (new URLSearchParams(location.search).get("pose") || "pose1").toLowerCase();
    if (p === "pose2" || p === "2" || p === "poseb" || p === "b") return "Pose2";
    if (p === "pose3" || p === "3" || p === "posec" || p === "c") return "Pose3";
    return "Pose1";
  })();

  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

  /* 一维平滑值噪声：sin-hash + 三次缓动插值，输出约 -1..1。
   * 比固定 sin 叠加更不像机械循环，适合做待机微动。 */
  function hash01(n) {
    const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  function smoothNoise1(t, seed) {
    const i = Math.floor(t);
    const f = t - i;
    const u = f * f * (3 - 2 * f);
    const a = hash01(i + seed * 101.13);
    const b = hash01(i + 1 + seed * 101.13);
    return (a + (b - a) * u) * 2 - 1;
  }
  function fbm1(t, seed) {
    return smoothNoise1(t, seed) * 0.65
         + smoothNoise1(t * 2.13 + 7.7, seed + 17) * 0.25
         + smoothNoise1(t * 4.31 + 3.1, seed + 37) * 0.10;
  }

  /* ---------------- OpennessTracker：viseme_core.py step 的 JS 直译 ----------------
   * 输入 rms（浮点域，静音≈0），输出开合度 0..1（EMA 平滑）。参数与旧引擎一致。 */
  const RMS_PARAMS = {
    floor: 0.003,     // ~-50 dBFS
    ceil: 0.25,       // ~-12 dBFS
    curve: 1.3,       // 弱音不张嘴、强音不开满
    openMin: 0.05,    // 开合度下限（静音近闭不僵死）
    openMax: 0.95,    // 开合度上限
    attack: 0.45,     // 开口速度（每帧）
    release: 0.18,    // 闭嘴惯性（每帧）
  };

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

  /* ---------------- Live2D Cubism 4 引擎实例 ---------------- */
  function createYukino2D(canvas, opts) {
    const modelUrl = (opts && opts.model) || DEFAULT_MODEL_URL;

    if (!window.PIXI || !window.PIXI.live2d || !window.PIXI.live2d.Live2DModel) {
      throw new Error("Live2D 运行时未加载（pixi/cubism4/live2dcubismcore）");
    }

    const state = {
      ready: false, destroyed: false,
      mouth: 0, talk: 0, lastMouthAt: 0,
      eye: {
        x: 0, y: 0,          // 当前实际位置
        tx: 0, ty: 0,        // 扫视目标
        sx: 0, sy: 0,        // 扫视起点
        saccadeStart: 0,     // 本次扫视开始时间
        saccadeDur: 0.1,     // 本次扫视时长
        nextAt: 0, lastT: 0,
      },
      poseIndex: DEFAULT_POSE === "Pose2" ? 1 : DEFAULT_POSE === "Pose3" ? 2 : 0,
      nextPoseAt: 0,       // 下次自动换 Idle 姿势的时间戳（首次由 boot 设置）
      actionUntil: 0,      // 一次性动作播放期间不自动换姿势
      poseTransition: null, // {from,to,group,start,duration,holdUntil,started}
      armNoiseEnv: 1,      // 手臂自然动作权重：姿势过渡期间淡出、结束后淡入，避免到位瞬间抖动
      _lastArmT: 0,
      // 鼠标目光跟随（作者 demo 同款 focusController）：光标在画布内才跟随；
      // 移出画布/说话/编排期间缓慢复位看向中央（不瞬跳）
      followOn: true,
      cursor: { x: 0, y: 0 },          // 缓动后的实际注视点（喂给 focusController）
      cursorTarget: { x: 0, y: 0 },    // 光标原始位置（仅画布内更新）
      cursorIn: false,                 // 光标是否在画布内
      _lastFollowT: 0,
      choreoRunning: false,            // 动作编排进行中（跟随让位给演出）
      demoStart: 0, demoUntil: 0,   // 模拟说话（talkDemo）时间窗，期间口型/手势由引擎接管
      autoPose: true,                // 待机姿势自动轮换；手动选姿态后可由 setAutoPose 关闭
      currentMotionDuration: 0,      // 最近一次创建的 motion 的 Meta.Duration（作者预置时长，秒）
    };
    canvas.dataset.live2d = "loading";
    // 右侧对话面板悬浮宽度（12px 间距 + 380px 面板）：模型在余下舞台居中。
    // 紧凑面板模式（body.compact，?compact=1）：对话面板在下方不在右侧，预留 0。
    const PANEL_RESERVE = (opts && opts.panelWidth)
      || (document.body.classList.contains("compact") ? 0 : 392);
    // 放大倍数：原 fit 尺寸 * 1.5（用户要求），模型更大更有存在感。
    const SCALE_BOOST = (opts && opts.scaleBoost) || 1.5;
    // 手臂/手部图层：模型默认 render order 里手臂一部分被身体/头发/脸压住。
    // 用户要求手臂在最上层，这里把手臂相关 drawable 的渲染顺序整体提到最后。
    const ARM_LAYER_IDS = new Set([
      "D_PSD_15", "D_PSD_16", "D_PSD_35", "D_PSD_37", "D_PSD_38", "D_PSD_39", "D_PSD_40",
      "D_SEIHUKU_HAND_R_01_00", "D_SEIHUKU_HAND_R_02_00",
      "D_PSD_58", "D_PSD_60", "D_PSD_64", "D_PSD_71", "D_PSD_72", "D_PSD_78",
      "D_SEIHUKU_HAND_L_01_00",
    ]);

    // 手臂过渡用参数：切换姿势时先从当前值 ease-out 旋转到目标姿势的手臂值，
    // 再开始目标姿势的预设 motion。
    const ARM_TRANSITION_PARAMS = [
      "PARAM_SHOULDER_R", "PARAM_SHOULDER_L",
      "PARAM_UPPERARM_R", "PARAM_UPPERARM_L",
      "PARAM_FOREARM_R", "PARAM_FOREARM_L",
      "PARAM_FOREARM_AP_R", "PARAM_FOREARM_AP_L",
      "PARAM_WRIST_R", "PARAM_WRIST_L",
      "PARAM_ARM1_AP", "PARAM_ARM2_AP",
      "PARAM_HAND1_AP", "PARAM_HAND2_AP",
    ];
    const POSE_ARM_TARGETS = {
      Pose1: {
        "PARAM_SHOULDER_R": 0, "PARAM_SHOULDER_L": 0,
        "PARAM_UPPERARM_R": 0, "PARAM_UPPERARM_L": -10,
        "PARAM_FOREARM_R": 0, "PARAM_FOREARM_L": 0,
        "PARAM_FOREARM_AP_R": 0, "PARAM_FOREARM_AP_L": -1,
        "PARAM_WRIST_R": 0, "PARAM_WRIST_L": 0,
        "PARAM_ARM1_AP": 0, "PARAM_ARM2_AP": 1,
        "PARAM_HAND1_AP": 0, "PARAM_HAND2_AP": 0,
      },
      Pose2: {
        "PARAM_SHOULDER_R": 2, "PARAM_SHOULDER_L": -3,
        "PARAM_UPPERARM_R": 10, "PARAM_UPPERARM_L": -8,
        "PARAM_FOREARM_R": 8, "PARAM_FOREARM_L": -1,
        "PARAM_FOREARM_AP_R": 1, "PARAM_FOREARM_AP_L": 1,
        "PARAM_WRIST_R": 0, "PARAM_WRIST_L": 0,
        "PARAM_ARM1_AP": 1, "PARAM_ARM2_AP": 0,
        "PARAM_HAND1_AP": 1, "PARAM_HAND2_AP": 0,
      },
      Pose3: {
        "PARAM_SHOULDER_R": -6, "PARAM_SHOULDER_L": 0,
        "PARAM_UPPERARM_R": 6.5, "PARAM_UPPERARM_L": -10,
        "PARAM_FOREARM_R": 13.5, "PARAM_FOREARM_L": -6,
        "PARAM_FOREARM_AP_R": 1, "PARAM_FOREARM_AP_L": 1,
        "PARAM_WRIST_R": 0, "PARAM_WRIST_L": 0.53,
        "PARAM_ARM1_AP": 1, "PARAM_ARM2_AP": 0,
        "PARAM_HAND1_AP": 0, "PARAM_HAND2_AP": 1,
      },
    };
    let app = null;
    let model = null;
    let resizeTimer = 0;

    /* 模型居中 + 放大：逻辑尺寸至少 320×480；宽度按“扣除右侧面板后的可视
     * 区域”计算，避免模型中心点被右浮面板带走。 */
    function fitModel() {
      if (!app || !model) return;
      const cw = canvas.clientWidth || window.innerWidth || 640;
      const ch = canvas.clientHeight || window.innerHeight || 480;
      const W = Math.max(cw, 320);
      const H = Math.max(ch, 480);
      app.renderer.resize(W, H);
      // 左侧历史抽屉打开时，模型在「左栏 + 右面板」之间的可视区域居中；
      // 关闭时只在右面板之外的区域居中。
      const sidebar = document.getElementById("history-sidebar");
      const historyOpen = document.body.classList.contains("history-open");
      const leftInset = (sidebar && historyOpen) ? sidebar.offsetWidth : 0;
      // 可视舞台宽度 = 全屏宽 - 左栏 - 右侧面板预留；极小屏至少保留 50% 宽度。
      const stageW = Math.max(W - leftInset - PANEL_RESERVE, W * 0.5);
      model.scale.set(1);
      // 用可见内容包围盒计算 fit，并让“可见内容中心”落在舞台中心
      //（Live2D 画布边缘常有大片透明，按画布尺寸居中会偏上/偏右）。
      const lb = model.getLocalBounds();
      const mw = Math.max(lb.width, model.width, 1);
      const mh = Math.max(lb.height, model.height, 1);
      const s = Math.min(stageW * 0.9 / mw, H * 1.02 / mh) * SCALE_BOOST;
      model.scale.set(s);
      model.anchor.set(0.5, 0.5);
      const offX = (lb.x + lb.width / 2) - model.internalModel.width / 2;
      const offY = (lb.y + lb.height / 2) - model.internalModel.height / 2;
      // 该模型可见内容在画布内偏上，垂直位置略下移（0.62H）让实际人物
      // 在可视舞台里看起来更居中、不贴顶。
      model.position.set(leftInset + stageW * 0.5 - offX * s, H * 0.62 - offY * s);
    }

    function bringArmLayersToTop() {
      if (!model || !model.internalModel) return;
      const core = model.internalModel.coreModel;
      const d = core && core.drawables;
      if (!d || !d.renderOrders || d.count === 0) return;
      const n = d.count;
      // 先按原 render order 排好所有 drawable，再把手臂/手部抽到最后（最高层）
      const order = Array.from({ length: n }, (_, i) => i);
      order.sort((a, b) => d.renderOrders[a] - d.renderOrders[b]);
      const rest = order.filter((i) => !ARM_LAYER_IDS.has(d.ids[i]));
      const arms = order.filter((i) => ARM_LAYER_IDS.has(d.ids[i]));
      if (arms.length === 0) return;
      const newOrder = rest.concat(arms);
      for (let pos = 0; pos < n; pos++) {
        d.renderOrders[newOrder[pos]] = pos;
      }
      console.info("[yukino2d] 手臂图层已置顶:", arms.map((i) => d.ids[i]).join(", "));
    }

    function slowMotionFades() {
      // 模型自带的动作/Idle 淡入淡出默认偏快（motion JSON 里是 1s），
      // 这里在 motion 创建后统一调慢：Idle 归位慢一点，动作切入也柔一点。
      const mm = model && model.internalModel && model.internalModel.motionManager;
      if (!mm || typeof mm.createMotion !== "function" || mm._yukinoSlowFadePatched) return;
      const origCreate = mm.createMotion.bind(mm);
      mm.createMotion = function (motionData, group, definition) {
        const meta = motionData && motionData.Meta;
        // 运行时 bug：核心 motion 从不应用 Meta.Loop，所有动作都按非循环处理。
        // 姿势 idle 动作(Loop:true)播完会触发 idle 返回把同一动作再播一遍
        // (即"被复位多执行一次")。这里把 Loop 写回、关掉循环衔接的重复淡入，
        // 并把每条曲线的首点值对齐末点值，消除循环边界的短段复位跳变。
        if (meta && meta.Loop && Array.isArray(motionData.Curves)) {
          for (const c of motionData.Curves) {
            const seg = c && c.Segments;
            if (seg && seg.length >= 2) seg[1] = seg[seg.length - 1];
          }
        }
        const motion = origCreate(motionData, group, definition);
        if (!motion) return motion;
        // 记录作者预置时长（Meta.Duration，秒），供编排引擎按动作自身时长推进
        state.currentMotionDuration = (motionData && motionData.Meta
          && typeof motionData.Meta.Duration === "number") ? motionData.Meta.Duration : 0;
        try {
          if (meta && typeof motion.setIsLoop === "function") {
            motion.setIsLoop(!!meta.Loop);
            motion.setIsLoopFadeIn(false);
          }
          if (group === mm.groups.idle) {
            motion.setFadeInTime(1.5);   // 姿势淡入与手臂过渡时长对齐，一次平滑到位
            motion.setFadeOutTime(1.5);
          } else {
            motion.setFadeInTime(1.4);   // 动作切入也放慢
            motion.setFadeOutTime(1.4);
          }
        } catch (e) {
          console.warn("[yukino2d] 调整 motion 淡入淡出失败:", e);
        }
        return motion;
      };
      mm._yukinoSlowFadePatched = true;
      console.info("[yukino2d] 动作/Idle 淡入淡出已调慢（Idle 2.8s，动作 1.4s）");
    }

    function easeOutCubic(u) {
      // 先快后慢：1-(1-u)^3
      return 1 - Math.pow(1 - u, 3);
    }

    function readArmParams() {
      const core = model && model.internalModel && model.internalModel.coreModel;
      if (!core) return null;
      const out = {};
      for (const id of ARM_TRANSITION_PARAMS) out[id] = core.getParameterValueById(id);
      return out;
    }

    function writeArmParams(values) {
      const core = model && model.internalModel && model.internalModel.coreModel;
      if (!core) return;
      for (const id of ARM_TRANSITION_PARAMS) {
        if (values[id] !== undefined) core.setParameterValueById(id, values[id]);
      }
    }

    function startPoseTransition(group) {
      if (!state.ready || !model || !model.internalModel) return false;
      if (state.poseTransition) return false;
      const target = POSE_ARM_TARGETS[group];
      if (!target) return false;
      const from = readArmParams();
      if (!from) return false;
      const now = performance.now() / 1000;
      state.poseTransition = {
        from,
        to: target,
        group,
        start: now,
        duration: 1.2 + Math.random() * 0.4,  // 1.2~1.6s，先快后慢，节奏更从容
        holdUntil: 0,
        started: false,
      };
      console.info("[yukino2d] 手臂过渡 ->", group,
                  "(", state.poseTransition.duration.toFixed(2), "s ease-out)");
      // 立即播新姿势动作：动作淡入与手臂过渡同时进行。之前等 ease 到位才播，
      // 动作异步加载未完成时手臂会被旧姿势拉回，造成"到位后初始位置不一样"的重叠。
      setPose(group, true);
      return true;
    }

    // 在 beforeModelUpdate 中调用：逐帧演算手臂从当前值旋转到目标值
    function updatePoseTransition(t) {
      const tr = state.poseTransition;
      if (!tr) return false;
      const core = model && model.internalModel && model.internalModel.coreModel;
      if (!core) return true;

      const u = clamp((t - tr.start) / tr.duration, 0, 1);
      const e = easeOutCubic(u);
      const vals = {};
      for (const id of ARM_TRANSITION_PARAMS) {
        vals[id] = tr.from[id] + (tr.to[id] - tr.from[id]) * e;
      }
      writeArmParams(vals);

      // 保持目标手臂值一小段时间，等新 motion 的淡入接管，避免跳变。
      // （motion 已在 startPoseTransition 时启动，这里只负责结束过渡）
      if (u >= 1 && !tr.started) {
        tr.started = true;
        tr.holdUntil = t + 0.35;
      }
      if (tr.started && t >= tr.holdUntil) {
        state.poseTransition = null;
      }
      return true;
    }

    function onResize() {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(fitModel, 60);
    }

    /* ---------------- Agent 动作 API ----------------
     * 供 agent / 控制台 / 自定义 ws 事件调用。动作名接受：
     *   - motion: "yuk_ikari" / "ikari" / "mot1a" / 数字 index
     *   - expression: "yuk_em0" / "em0" / 数字 index
     *   - pose: "pose1" / "pose2" / "pose3"（同时把待机切换为该 pose）
     * 返回 Promise<boolean>；模型未就绪时恒为 false。 */
    function getMotionGroups() {
      const settings = model && model.internalModel && model.internalModel.settings;
      return (settings && settings.motions) || {};
    }
    function getActionMotions() {
      const groups = getMotionGroups();
      return groups.Action || groups.action || [];
    }
    function getExpressionDefs() {
      const settings = model && model.internalModel && model.internalModel.settings;
      return (settings && settings.expressions) || [];
    }

    function motionIndexFromName(name) {
      if (typeof name === "number") return name;
      const action = getActionMotions();
      const raw = String(name || "").toLowerCase().replace(/\.motion3\.json$/, "");
      if (/^\d+$/.test(raw)) return parseInt(raw, 10);
      const alias = { angry: "ikari", confused: "hatena", sweat: "sweat", sparkle: "uruuru", dark: "yami", eyebrow: "mayu" };
      const key = alias[raw] || raw.replace(/^yuk_/, "");
      for (let i = 0; i < action.length; i++) {
        const file = String(action[i].File || action[i].file || "").toLowerCase();
        if (file.includes(key)) return i;
      }
      return -1;
    }

    function expressionIndexFromName(name) {
      if (typeof name === "number") return name;
      const exps = getExpressionDefs();
      const raw = String(name || "").toLowerCase().replace(/\.exp3\.json$/, "");
      if (/^\d+$/.test(raw)) return parseInt(raw, 10);
      const key = raw.replace(/^yuk_/, "");
      for (let i = 0; i < exps.length; i++) {
        const file = String(exps[i].File || exps[i].file || "").toLowerCase();
        if (file.includes(key)) return i;
      }
      return -1;
    }

    async function playMotion(name, priority) {
      if (!state.ready || !model) return false;
      const idx = motionIndexFromName(name);
      if (idx < 0) {
        console.warn("[yukino2d] 未知动作:", name, "可用:", listMotions().map((m) => m.file));
        return false;
      }
      try {
        const prio = typeof priority === "number" ? priority : 3;
        state.poseTransition = null;  // 一次性动作优先：取消未完成的手臂姿势过渡
        state.actionUntil = performance.now() + 3500;  // 动作播放期间不自动换 Idle 姿势
        return !!(await model.motion("Action", idx, prio));
      } catch (e) {
        console.warn("[yukino2d] 动作播放失败:", e);
        return false;
      }
    }

    async function setExpression(name) {
      if (!state.ready || !model) return false;
      const idx = expressionIndexFromName(name);
      if (idx < 0) {
        console.warn("[yukino2d] 未知表情:", name, "可用:", listExpressions().map((e) => e.file));
        return false;
      }
      try {
        return !!(await model.expression(idx));
      } catch (e) {
        console.warn("[yukino2d] 表情切换失败:", e);
        return false;
      }
    }

    async function setPose(name, immediate) {
      if (!state.ready || !model) return false;
      // 接受 pose1/pose2/pose3 与用户重命名后的 poseA/poseB/poseC（a/b/c → 1/2/3）
      const key = String(name || "pose1").toLowerCase().replace(/^pose/, "");
      const num = key === "a" ? "1" : key === "b" ? "2" : key === "c" ? "3" : key;
      const group = "Pose" + num;
      const groups = getMotionGroups();
      if (!groups[group]) {
        console.warn("[yukino2d] 未知姿态:", name);
        return false;
      }
      // 非 immediate：先从当前手臂位置 ease-out 旋转到目标姿势，再启动预设 motion
      if (!immediate) {
        return startPoseTransition(group);
      }
      try {
        model.internalModel.motionManager.groups.idle = group;
        return !!(await model.motion(group, 0, 3));
      } catch (e) {
        console.warn("[yukino2d] 姿态切换失败:", e);
        return false;
      }
    }

    function cyclePose() {
      const order = ["Pose1", "Pose2", "Pose3"];
      state.poseIndex = (state.poseIndex + 1) % order.length;
      const next = order[state.poseIndex];
      console.info("[yukino2d] 自动切换 Idle 姿势 ->", next);
      setPose(next);
    }

    /* 待机姿势自动轮换开关：手动点选姿态后关闭，雪乃保持用户选的姿势；
     * 重开则从 8~14s 后重新开始轮换。 */
    function setAutoPose(on) {
      state.autoPose = !!on;
      if (state.autoPose) {
        state.nextPoseAt = performance.now() + 8000 + Math.random() * 6000;
      } else {
        state.nextPoseAt = 0;
      }
      console.info("[yukino2d] 待机姿势自动轮换:", state.autoPose ? "开" : "关");
    }

    function listMotions() {
      return getActionMotions().map((m, i) => ({ index: i, file: m.File || m.file || "" }));
    }
    function listExpressions() {
      return getExpressionDefs().map((e, i) => ({ index: i, file: e.File || e.file || "" }));
    }

    /* 口型参数：静音 openMin=0.05 以下视为闭口；最大开合映射到 0.9，
     * 避免 Cubism 参数拉满导致口型过张失真。 */
    function mouthParam(v) {
      return Math.max(0, (clamp(v, 0, 1) - RMS_PARAMS.openMin) /
                          (RMS_PARAMS.openMax - RMS_PARAMS.openMin)) * 0.9;
    }

    /* ---------------- 说话时眼睛转动 ----------------
     * 模型眼睛是单独切片：PARTS_01_EYE_001（眼眶）+ PARTS_01_EYE_BALL_001（眼珠），
     * 眼珠由 PARAM_EYE_BALL_X/Y 驱动。这里在 beforeModelUpdate 里叠加一个
     * 随机扫视（saccade）控制器：说话时扫视更频繁、幅度更大；待机时偶尔微动。
     * 使用 addParameterValueById 叠加在 Idle motion 之上，不覆盖动作系统。 */
    function updateEyeGaze(t, talk) {
      const e = state.eye;
      const dt = e.lastT ? clamp(t - e.lastT, 0.008, 0.05) : 0.016;
      e.lastT = t;

      // 到点换新目标；间隔用「基础 + 随机 + 偶尔长注视」，比固定周期更接近真人
      if (e.nextAt === 0 || t >= e.nextAt) {
        const ampX = 0.16 + talk * 0.72;   // 说话时视线更活跃
        const ampY = 0.09 + talk * 0.30;
        e.sx = e.x;
        e.sy = e.y;
        e.tx = (Math.random() * 2 - 1) * ampX;
        e.ty = (Math.random() * 2 - 1) * ampY;
        // 更高概率回到中央/看向对方，减少“机械乱看”
        if (Math.random() < 0.38 + talk * 0.16) {
          e.tx *= 0.1;
          e.ty *= 0.1;
        }
        // 扫视快而短：真实扫视 60~140ms 完成，之后缓慢漂移
        e.saccadeStart = t;
        e.saccadeDur = 0.06 + Math.random() * 0.08;
        const base = talk > 0.3
          ? 1.1 + Math.random() * 1.8    // 说话：1.1~2.9s
          : 2.6 + Math.random() * 3.4;   // 待机：2.6~6.0s
        // 偶尔发一次长注视（人看东西会停顿更久）
        e.nextAt = t + base * (Math.random() < 0.18 ? 1.7 : 1);
      }

      // 扫视阶段用 smoothstep 快速到位；之后缓慢向目标漂移（微追随）
      const elapsed = t - e.saccadeStart;
      let k;
      if (elapsed < e.saccadeDur) {
        const u = Math.max(0, elapsed / e.saccadeDur);
        k = u * u * (3 - 2 * u);
      } else {
        k = 1 - Math.exp(-dt * 1.6);
      }
      e.x += (e.tx - e.x) * k;
      e.y += (e.ty - e.y) * k;

      // 注视期间叠加很轻的微漂移，避免完全静止的“假”
      const microX = (smoothNoise1(t * 0.5, 17) - 0) * 0.05;
      const microY = (smoothNoise1(t * 0.45, 29) - 0) * 0.04;
      return { x: e.x + microX, y: e.y + microY };
    }

    /* ---------------- 手/手臂自然动作 ----------------
     * 手可以动：在 Idle motion 之上叠加低频手臂摆动。待机时像站姿微调，
     * 说话时动作幅度和频率随 talk 增加，像配合语气做手势，不会一直僵在
     * 撩头发/举手姿势上。全部使用 addParameterValueById，不覆盖动作系统。 */
    function updateArmMotion(t, talk) {
      // 呼吸感包络：整体动作能量 0~1 缓慢起伏，有时接近静止，更像真人站姿
      const idle = 0.35 + 0.65 * (0.5 + 0.5 * smoothNoise1(t * 0.09, 71));
      // 说话手势：噪声驱动，不再是匀速正弦；talk 高时动作多，也有自然停顿
      const gesture = talk * (0.45 + 0.55 * (0.5 + 0.5 * smoothNoise1(t * 0.75, 83)));
      const energy = clamp(idle + gesture * 1.15, 0, 1.6);

      return {
        shoulderR:  fbm1(t * 0.31, 101) * 0.45 * energy,
        upperArmR:  fbm1(t * 0.27, 102) * 1.0 * energy
                  + talk * fbm1(t * 1.2, 112) * 0.9,
        forearmR:   fbm1(t * 0.23, 103) * 1.25 * energy
                  + talk * fbm1(t * 1.35, 113) * 1.2,
        forearmApR: fbm1(t * 0.19, 104) * 0.65 * energy
                  + talk * fbm1(t * 0.9, 114) * 0.55,
        wristR:     fbm1(t * 0.45, 105) * 0.3 * energy
                  + talk * fbm1(t * 1.5, 115) * 0.25,
        shoulderL:  fbm1(t * 0.29, 106) * 0.4 * energy,
        upperArmL:  fbm1(t * 0.25, 107) * 0.9 * energy,
        forearmL:   fbm1(t * 0.21, 108) * 1.05 * energy,
        forearmApL: fbm1(t * 0.18, 109) * 0.6 * energy,
        wristL:     fbm1(t * 0.42, 110) * 0.28 * energy,
      };
    }

    /* ---------------- 鼠标目光跟随（作者 demo 同款） ----------------
     * 光标相对画布归一化到 [-1,1]（x 左负右正，y 上正下负）。仅光标在画布
     * （模型框）内时记录目标；移出画布/失焦只清 cursorIn 标志，由帧循环
     * 每帧缓动回中央（缓慢复位，不瞬跳）。focusController 自带二阶阻尼，
     * 叠加此处缓动 = 平滑追随 + 平缓归位。 */
    function onPointerMove(ev) {
      if (!state.followOn) return;
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const px = ev.clientX, py = ev.clientY;
      const inside = px >= rect.left && px <= rect.right && py >= rect.top && py <= rect.bottom;
      if (inside) {
        state.cursorIn = true;
        state.cursorTarget.x = clamp((px - rect.left) / rect.width * 2 - 1, -1, 1);
        state.cursorTarget.y = clamp(1 - (py - rect.top) / rect.height * 2, -1, 1);
      } else {
        state.cursorIn = false;   // 移出模型框 → 帧循环缓慢复位
      }
    }
    function onBlurReset() {
      state.cursorIn = false;
    }
    function setFollow(on) {
      const v = !!on;
      if (state.followOn === v) return;
      state.followOn = v;
      const fc = model && model.internalModel && model.internalModel.focusController;
      if (fc && !v) fc.focus(0, 0);  // 关闭时复位视线到中央
      console.info("[yukino2d] 目光跟随:", v ? "开" : "关");
    }

    /* 模拟说话：duration 秒内的假口型包络（音节半波 × 语调起伏），
     * 返回 0..1 或 null（不在演示窗口内）。窗口期内口型与手势由引擎接管，
     * 外部 setMouth(RMS≈0) 不会把它覆盖掉。 */
    function demoMouthValue(t) {
      if (state.demoUntil <= t) return null;
      const el = t - state.demoStart;
      const pulse = Math.max(0, Math.sin(el * Math.PI * 5));   // ~5 音节/秒
      const wobble = 0.6 + 0.4 * Math.sin(el * 2.3);           // 语调起伏
      return clamp(0.12 + pulse * 0.55 * wobble, 0, 1);
    }
    function talkDemo(duration) {
      const dur = (typeof duration === "number" && duration > 0) ? duration : 4;
      const now = performance.now() / 1000;
      state.demoStart = now;
      state.demoUntil = now + dur;
      console.info(`[yukino2d] 模拟说话 ${dur}s`);
    }

    /* ---------------- 动作编排（演出流程） ----------------
     * playChoreography(steps) 按顺序播放一串步骤：
     *   {type:"expression", value} / {type:"motion", value}
     *   {type:"pose", value} / {type:"pause", ms}
     * motion 步骤按作者预置时长（Meta.Duration，秒）推进，并在结尾提前 fadeOut
     * 重叠启动下一步（衔接过渡保留，不硬切）。编排期间暂停姿势自动轮换；
     * 新一次调用会打断上一轮。 */
    let choreoId = 0;
    function waitMs(ms) {
      return new Promise((res) => setTimeout(res, Math.max(0, ms)));
    }
    async function playMotionChoreo(name) {
      if (!state.ready || !model) return 0;
      const idx = motionIndexFromName(name);
      if (idx < 0) {
        console.warn("[yukino2d] 编排未知动作:", name);
        return 0;
      }
      state.poseTransition = null;
      state.actionUntil = performance.now() + 3500;
      state.currentMotionDuration = 0;
      await model.motion("Action", idx, 3);
      return state.currentMotionDuration || 3.5;   // 兜底 3.5s
    }
    async function playChoreography(steps) {
      if (!state.ready || !Array.isArray(steps) || !steps.length) return false;
      const id = ++choreoId;
      const wasAuto = state.autoPose;
      state.autoPose = false;        // 编排期间不自动换姿势
      state.choreoRunning = true;    // 编排期间目光跟随让位（避免与动作打架）
      try {
        for (const step of steps) {
          if (id !== choreoId || state.destroyed) break;   // 新编排打断旧编排
          if (!step || typeof step !== "object") continue;
          const s = step.type;
          try {
            if (s === "expression") {
              await setExpression(step.value);
              await waitMs(500);
            } else if (s === "motion") {
              const dur = await playMotionChoreo(step.value);
              if (dur > 0) await waitMs(Math.max(0, dur * 1000 - 1400));  // 留 1.4s fadeOut 重叠
            } else if (s === "pose") {
              await setPose(step.value);
              await waitMs(400);
            } else if (s === "pause") {
              await waitMs(step.ms || 0);
            }
          } catch (e) {
            console.warn("[yukino2d] 编排步骤失败:", e);
          }
        }
      } finally {
        state.choreoRunning = false;
        state.autoPose = wasAuto;
      }
      return true;
    }



    /* 自然肢体动作 + 说话协同：
     * - 待机：低频轻微头/身体摆动（PARAM_ANGLE_X/Y/Z + PARAM_BODY_ANGLE_X）
     * - 说话：以 state.talk 为能量，摆动幅度与频率增强，并带轻微点头
     * 使用 addParameterValueById 叠加在 Idle motion 之上，不覆盖动作系统。 */
    function handleBeforeModelUpdate() {
      if (!model || !model.internalModel) return;
      const core = model.internalModel.coreModel;
      const t = performance.now() / 1000;
      // 模拟说话窗口期内：口型由假包络接管（外部 setMouth 不覆盖），手势能量拉高
      const demo = demoMouthValue(t);
      const talk = (demo !== null) ? 0.8 : (state.talk || 0);
      // 手臂姿势过渡：先快后慢旋转到目标手臂值，完成后再启动预设 motion
      const transitioning = updatePoseTransition(t);
      // 手臂自然动作权重：过渡期间淡出、结束后平滑淡入（时间常数 ~0.2s），
      // 避免到位瞬间噪声从 0 突然恢复造成手臂小抖。
      const armDt = state._lastArmT ? Math.min(0.1, t - state._lastArmT) : 0.016;
      state._lastArmT = t;
      state.armNoiseEnv += ((transitioning ? 0 : 1) - state.armNoiseEnv) * Math.min(1, armDt * 5);
      core.setParameterValueById(
        "PARAM_MOUTH_OPEN_Y",
        mouthParam(demo !== null ? demo : state.mouth)
      );
      // 目光：followOn → focusController 接管（运行时 updateFocus 写 EYE_BALL_X/Y、
      // ANGLE_X/Y/Z、BODY_ANGLE_X，带二阶阻尼）；关闭 → 自身随机扫视 + 头摆噪声。
      // 跟随条件：光标在画布内 + 未在说话（talk<0.25）+ 无编排在跑——
      // 说话/演出期间跟随让位，缓慢复位看向中央，避免与动作打架。
      if (state.followOn) {
        const fc = model.internalModel.focusController;
        if (fc) {
          const followActive = state.cursorIn && talk < 0.25 && !state.choreoRunning;
          const dt = state._lastFollowT ? Math.min(0.05, t - state._lastFollowT) : 0.016;
          state._lastFollowT = t;
          const tx = followActive ? state.cursorTarget.x : 0;
          const ty = followActive ? state.cursorTarget.y : 0;
          const k = 1 - Math.exp(-dt * 2.5);   // ~0.4s 半衰期：平缓追随、移出缓慢复位
          state.cursor.x += (tx - state.cursor.x) * k;
          state.cursor.y += (ty - state.cursor.y) * k;
          fc.focus(state.cursor.x, state.cursor.y);
        }
      } else {
        // 眼睛转动（说话时更明显）：眼珠切片 PARTS_01_EYE_BALL_001 由
        // PARAM_EYE_BALL_X/Y 驱动，叠加随机扫视
        const gaze = updateEyeGaze(t, talk);
        core.addParameterValueById("PARAM_EYE_BALL_X", gaze.x);
        core.addParameterValueById("PARAM_EYE_BALL_Y", gaze.y);
      }
      // 手/手臂自然动作（过渡期间噪声权重趋 0，让过渡曲线干净）
      const arm = updateArmMotion(t, talk);
      const aw = state.armNoiseEnv;
      if (aw > 0.001) {
        core.addParameterValueById("PARAM_SHOULDER_R", arm.shoulderR * aw);
        core.addParameterValueById("PARAM_UPPERARM_R", arm.upperArmR * aw);
        core.addParameterValueById("PARAM_FOREARM_R", arm.forearmR * aw);
        core.addParameterValueById("PARAM_FOREARM_AP_R", arm.forearmApR * aw);
        core.addParameterValueById("PARAM_WRIST_R", arm.wristR * aw);
        core.addParameterValueById("PARAM_SHOULDER_L", arm.shoulderL * aw);
        core.addParameterValueById("PARAM_UPPERARM_L", arm.upperArmL * aw);
        core.addParameterValueById("PARAM_FOREARM_L", arm.forearmL * aw);
        core.addParameterValueById("PARAM_FOREARM_AP_L", arm.forearmApL * aw);
        core.addParameterValueById("PARAM_WRIST_L", arm.wristL * aw);
      }
      // 自动轮换 Idle 姿势：说话/动作/过渡期间不换，避免打断表演
      const nowMs = performance.now();
      if (state.autoPose && !transitioning && state.nextPoseAt > 0 && nowMs >= state.nextPoseAt
          && talk < 0.15 && nowMs >= state.actionUntil) {
        state.nextPoseAt = nowMs + 18000 + Math.random() * 14000;
        setTimeout(cyclePose, 0);  // 延迟到本帧外再切，避免在 beforeModelUpdate 里动 motion
      }
      // 头/身体自然摆动（噪声驱动，避免固定周期机械感）。
      // 仅 followOff 时本模块写；followOn 时 focusController 已接管 ANGLE/BODY_ANGLE。
      if (!state.followOn) {
        core.addParameterValueById(
          "PARAM_ANGLE_X",
          fbm1(t * 0.24, 201) * (2.6 + talk * 1.5)
          + talk * fbm1(t * 1.1, 211) * 1.8   // 说话时不规则点头/强调
        );
        core.addParameterValueById(
          "PARAM_ANGLE_Y",
          fbm1(t * 0.2, 202) * (2.0 + talk * 1.0)
        );
        core.addParameterValueById(
          "PARAM_ANGLE_Z",
          fbm1(t * 0.16, 203) * 1.3
        );
        core.addParameterValueById(
          "PARAM_BODY_ANGLE_X",
          fbm1(t * 0.18, 204) * 1.5
        );
      }
    }

    (async function boot() {
      try {
        const width = canvas.clientWidth || window.innerWidth || 640;
        const height = canvas.clientHeight || window.innerHeight || 480;

        app = new PIXI.Application({
          view: canvas,
          autoStart: true,
          transparent: true,
          backgroundAlpha: 0,
          antialias: true,
          autoDensity: true,
          resolution: Math.min(window.devicePixelRatio || 1, 2),
          width: width,
          height: height,
        });

        model = await PIXI.live2d.Live2DModel.from(modelUrl, {
          autoInteract: false,   // 数字人场景无点击/拖拽交互
          autoUpdate: true,      // 挂到 PIXI.Ticker.shared 自动 update
        });

        if (state.destroyed) {
          model.destroy();
          app.destroy(false, { children: true, texture: true, baseTexture: true });
          return;
        }

        /* 官方预览页的兼容处理：pixi-live2d-display 默认参数 ID 是旧版
         * "ParamAngleX" 风格，这个模型的 moc3 里是 "PARAM_ANGLE_X" 大写风格，
         * 不覆盖会导致焦点/呼吸/身体角度参数写不进去。 */
        Object.assign(model.internalModel, {
          idParamAngleX: "PARAM_ANGLE_X",
          idParamAngleY: "PARAM_ANGLE_Y",
          idParamAngleZ: "PARAM_ANGLE_Z",
          idParamEyeBallX: "PARAM_EYE_BALL_X",
          idParamEyeBallY: "PARAM_EYE_BALL_Y",
          idParamBodyAngleX: "PARAM_BODY_ANGLE_X",
        });

        bringArmLayersToTop();
        slowMotionFades();

        model.internalModel.on("beforeModelUpdate", handleBeforeModelUpdate);

        app.stage.addChild(model);
        fitModel();

        /* 启动初始 Idle 姿势（默认 Pose1，可用 ?pose=pose2|pose3 指定），
         * 让 Live2D 自带呼吸/眨眼/待机动作；motion 加载失败不影响主体渲染。 */
        try {
          model.internalModel.motionManager.groups.idle = DEFAULT_POSE;
          await model.motion(DEFAULT_POSE, 0, 3);
        } catch (e) {
          console.warn("[yukino2d] Idle motion 启动失败（仍可渲染静态模型）:", e);
        }
        // 首次 8~14s 后换一次姿势，之后每 18~32s 轮换一次，
        // 避免频繁切换带来机械感
        state.nextPoseAt = performance.now() + 8000 + Math.random() * 6000;

        window.addEventListener("resize", onResize);
        window.addEventListener("pointermove", onPointerMove);
        window.addEventListener("mousemove", onPointerMove);   // 老内核回退
        window.addEventListener("blur", onBlurReset);          // 失焦回看中央
        canvas.addEventListener("mouseleave", onBlurReset);    // 光标离开画布回看中央
        state.ready = true;
        canvas.dataset.live2d = "ready";
        console.info(`[yukino2d] Live2D 模型已就绪: ${modelUrl}`);
      } catch (e) {
        console.error("[yukino2d] Live2D 引擎启动失败:", e);
        state.ready = false;
        canvas.dataset.live2d = "error";
        const fallback = document.getElementById("avatar-fallback");
        if (fallback) fallback.classList.remove("hidden");
      }
    })();

    const api = {
      /* 目标开合度 0..1（外部 OpennessTracker 输出直接喂） */
      setMouth(v) {
        const now = performance.now();
        const dt = state.lastMouthAt
          ? Math.min(0.1, (now - state.lastMouthAt) / 1000)
          : 0.016;
        state.lastMouthAt = now;
        const target = clamp(v, 0, 1);
        state.mouth = target;
        // 说话能量做 EMA 平滑（~0.22s），避免 RMS 抖动直接传到肢体上造成抽搐
        const alpha = 1 - Math.exp(-dt * 4.5);
        state.talk += (target - state.talk) * alpha;
      },
      isReady() { return state.ready; },
      playMotion,
      setExpression,
      setPose,
      setFollow,
      setAutoPose,
      talkDemo,
      playChoreography,
      listMotions,
      listExpressions,
      destroy() {
        state.destroyed = true;
        window.removeEventListener("resize", onResize);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("mousemove", onPointerMove);
        window.removeEventListener("blur", onBlurReset);
        canvas.removeEventListener("mouseleave", onBlurReset);
        window.clearTimeout(resizeTimer);
        if (model) {
          try {
            if (model.internalModel) {
              model.internalModel.off("beforeModelUpdate", handleBeforeModelUpdate);
            }
          } catch (e) { /* ignore */ }
          try { model.destroy(); } catch (e) { /* ignore */ }
          model = null;
        }
        if (app) {
          try {
            app.destroy(false, { children: true, texture: true, baseTexture: true });
          } catch (e) { /* ignore */ }
          app = null;
        }
      },
    };
    window.yukino2dAgent = api;  // 供 agent/控制台直接调用
    canvas.dataset.agent = "ready";
    return api;
  }

  window.initYukino2D = createYukino2D;
  window.createOpennessTracker = (opts) => new OpennessTracker(opts);
})();
