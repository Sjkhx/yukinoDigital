/* VoxEMW 数字人语音助手前端。
 *
 * 一路 ws（/ws，orchestrator）承载三种流量：
 *   上行 JSON：OpenAI Realtime 事件（input_audio_buffer.append / response.cancel）
 *              + 自定义 {"type":"vox.persona","id":...} 人设切换
 *   下行 JSON：Realtime 事件透传（转写/音频 delta/打断）+ {"type":"vox.status",...}
 *   下行二进制：0x01 + tag(1B) + JPEG 数字人视频帧
 *               （tag 0x00=idle 待机微动；0x01=speech 说话帧；合流同队列沿音频时钟连播）
 *
 * 音频：麦克风 AudioWorklet 16kHz int16 上行；TTS PCM16 delta 在 AudioContext
 * 时间轴上无缝拼接播放（播放链挂 AnalyserNode，2dlive 口型驱动采样点）；
 * speech_started（打断）时清空播放队列。
 * 视频：JPEG 帧到就画；数字人缺席时显示 persona 静态肖像（纯语音模式）。
 * 2dlive（avatar_backend=2dlive）：雪乃 Live2D Cubism 4 前端 WebGL 渲染
 * （web/yukino2d.js + /static/live2d/runtime pixi/pixi-live2d-display），
 * 无 JPEG 帧流——口型由 TTS 播放链 AnalyserNode 实时 RMS → OpennessTracker
 * （viseme 同款参数）驱动，眨眼/呼吸/待机动作由 Live2D 模型自身驱动，
 * 见 init2DLive/start2DLoop。
 */

"use strict";

const SAMPLE_RATE = 16000;
const FRAME_TYPE_JPEG = 0x01;       // 下行：数字人视频帧
const FRAME_TAG_IDLE = 0x00;        // 视频帧 tag：静音驱动的待机微动
const FRAME_TAG_SPEECH = 0x01;      // 视频帧 tag：真实音频驱动

const els = {
  status: document.getElementById("status"),
  canvas: document.getElementById("avatar-canvas"),
  still: document.getElementById("avatar-still"),
  fallback: document.getElementById("avatar-fallback"),
  avatarLabel: document.getElementById("avatar-label"),
  avatarState: document.getElementById("avatar-state"),
  personaBar: document.getElementById("persona-bar"),
  transcript: document.getElementById("transcript"),
  micBtn: document.getElementById("mic-btn"),
  textInput: document.getElementById("text-input"),
  sendBtn: document.getElementById("send-btn"),
  historyBtn: document.getElementById("history-toggle"),
  historyView: document.getElementById("history-sidebar"),
  historyClose: document.getElementById("history-close"),
  historyList: document.getElementById("history-list"),
  historyBack: document.getElementById("history-back"),
  historyRefresh: document.getElementById("history-refresh"),
  historyTitle: document.getElementById("history-title"),
  historyNew: document.getElementById("history-new"),
  perfToggle: document.getElementById("perf-toggle"),
  perfDrawer: document.getElementById("performance-drawer"),
};

let ws = null;
let mic = null;
let player = null;
let personas = [];
let currentPersona = null;
let avatarOn = false;
let assistantLine = null; // 正在流式累积的助手文本行
let assistantJaText = "";   // 本轮助手日语全文（live2d 情绪 agent 用）
// 2dlive（Live2D Cubism 4 前端 WebGL 渲染）状态：本地引擎 + RMS→开合度映射
let mode2d = false;
let yukino2d = null;        // { setMouth, isReady, destroy }，来自 /static/yukino2d.js
let mouthTracker = null;    // OpennessTracker（viseme 同款 RMS→开合度）
let aiChoreoPending = false;  // 本轮 LLM 自带【演出】编排（vox.choreo 已收），跳过默认情绪演出
let aiChoreoResetTimer = null;  // task-done 编排没有后续 response.done，定时自动清除
// ASR final 后的收音门控：final → 暂停送麦克风音频，等本轮 TTS 播放排空再恢复。
// 效果：回答播放期间不再接受用户插话音频（也不会误触发新 ASR/打断）。
let micInputPaused = false;   // true = 暂停向 s2s 发送 input_audio_buffer.append
let resumeMicOnDrain = false; // true = 已收到 response.done/error，只等播放排空即恢复
// solo 模式（?solo=1）：demo 录制用，隐藏用户画面、数字人单栏居中、不开摄像头
const SOLO_MODE = new URLSearchParams(location.search).has("solo");
if (SOLO_MODE) document.body.classList.add("solo");
// 紧凑面板模式（?compact=1）：DSH 右侧 1/4 屏嵌入式面板，上部 Live2D、下部对话+打字
const COMPACT_MODE = new URLSearchParams(location.search).has("compact");
if (COMPACT_MODE) document.body.classList.add("compact");

// Live2D Cubism 4 模型：默认制服（yukino_seihuku），?outfit=shihuku 切换私服。
// ?v= 缓存版本号：模型文件改名后浏览器强制拉新 model3.json（旧缓存会 404）
const MODEL_VERSION = "20260819e";
let LIVE2D_MODEL_URL = (new URLSearchParams(location.search).get("outfit") === "shihuku"
  ? "/static/live2d/models/yukino/yukino_shihuku.model3.json"
  : "/static/live2d/models/yukino/yukino_seihuku.model3.json") + "?v=" + MODEL_VERSION;


// ---------------------------------------------------------------------------
// PCM 编解码
// ---------------------------------------------------------------------------

function floatTo16BitPCM(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function base64FromInt16(int16) {
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function int16FromBase64(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

// ---------------------------------------------------------------------------
// 播放（无缝拼接 + 打断清空）
// ---------------------------------------------------------------------------

// 已排程的 BufferSource 引用：src.start() 后节点若被 GC 回收，
// 未轮到播放的音频会静音（长回复数百块排程时尤甚——"只听到第一句"事故根因）。
// 保持引用直到 onended 触发（播放完毕）才释放。
const activeSources = new Set();

function ensurePlayer() {
  if (!player) {
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    // AnalyserNode 挂在播放链路上（2dlive 口型驱动用，直通节点零延迟）：
    // 所有 TTS 音频经 analyser → destination，读到的正是耳朵听到的混音
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;              // 512/16000 = 32ms 时间窗，贴近 viseme 40ms
    analyser.connect(ctx.destination);
    player = { ctx, analyser, nextStartTime: 0 };
  }
  // Chrome 自动挂起恢复：长对话静音间隔后 ctx 可能被挂起（suspended），
  // 不恢复则只弹文本无语音（对话滚动条出现 ≈ 对话时间长的典型触发，2026-08-12）
  if (player.ctx.state === "suspended") {
    player.ctx.resume();
  }
  if (player.ctx.state === "suspended") player.ctx.resume();
  return player;
}

// 口型同步模型:TTS 以 ~2x 实时速度流式送音频,播放链会积压(实测 3-4s),
// 各自计时必然漂移 → 视频帧节奏从属于音频播放时钟:音频播到第几秒就放第几帧。
// AVATAR_AUDIO_DELAY 保留基础延迟,保证音频播放位置始终落后于视频供帧点,
// 视频始终有帧可放。AVTR-1 0.2s chunk + 0.2s 前瞻 + 生成 ≈ 实测 0.35s 最佳。
// 可用 ?adelay=N 覆盖(?debug=1 看帧队列深度,经常见底就调回去)
const ADELAY_OVERRIDE = (() => {
  const v = parseFloat(new URLSearchParams(location.search).get("adelay"));
  return Number.isFinite(v) && v >= 0 ? v : null;
})();
// adelay：音频播放起点相对供帧的延迟缓冲。avtr1 = 0.2s 前瞻 + ~0.08s 生成；
// tha3/viseme 零模型前瞻（音量包络直驱口型），帧随音频即刻产出，0.2s 覆盖
// 传输/解码抖动即可；未知 backend 兜底 0.8。
const BACKEND_ADELAY = { avtr1: 0.35, tha3: 0.2, viseme: 0.2, "2dlive": 0.02 };
let avatarAudioDelay = ADELAY_OVERRIDE ?? 0.35;

// 口型-语音时间偏移补偿（帧数）：AVTR-1 模型固有口型滞后（音频包络 vs 唇部开合
// 互相关实测：爆破音段 +3 帧/120ms、自然语音段 +2 帧/80ms，官方 generate_offline
// 同幅——模型属性非链路错位），视频提前 3 帧（120ms）补偿。
// 可用 ?vlag=N 覆盖（正负号：target = pos*25 - videoLagFrames）
const VLAG_OVERRIDE = (() => {
  const v = parseInt(new URLSearchParams(location.search).get("vlag") || "", 10);
  return Number.isFinite(v) ? v : null;
})();
const BACKEND_VLAG = { avtr1: -3, tha3: 0, viseme: 0, "2dlive": 0 };  // 贴片合成嘴随包络即刻响应
let videoLagFrames = VLAG_OVERRIDE ?? 0;

let audioBlockCount = 0, audioBlockSamples = 0;
function playPCM(int16) {
  // 诊断：每 100 块打印一次累计（656 块 ≈ 21s），确认 ws 收到完整音频流
  audioBlockCount++;
  audioBlockSamples += int16.length;
  if (audioBlockCount % 100 === 1) {
    console.info(`[audio] blocks=${audioBlockCount} ≈${(audioBlockSamples / SAMPLE_RATE).toFixed(1)}s`);
  }
  const p = ensurePlayer();
  const buf = p.ctx.createBuffer(1, int16.length, SAMPLE_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) data[i] = int16[i] / 0x8000;
  const src = p.ctx.createBufferSource();
  src.buffer = buf;
  src.connect(p.analyser);   // 经 analyser 出 destination（口型驱动采样点）
  const prevEnd = p.nextStartTime;  // 本 delta 排程前的音频链尾（= 上一段回复的播放结束点）
  const start = Math.max(p.ctx.currentTime + (avatarOn ? avatarAudioDelay : 0.02), prevEnd);
  src.start(start);
  p.nextStartTime = start + buf.duration;
  activeSources.add(src);    // 保持引用防 GC 回收未播放节点
  src.onended = () => activeSources.delete(src);
  if (needVideoBase) {
    if (prevEnd - p.ctx.currentTime < 0.3) {
      // 常规：上一段回复已播完。本 response 首个音频 delta:记录它在 ctx 时间轴上的
      // 起点作为视频对齐基准。同时清空队列:里面滞留的是上一回复的"闭嘴尾帧"
      // (句尾零填充生成),不清掉会被当作本回复的开头播出,嘴型整体慢 ~1s
      responseAudioBase = start;
      videoFrameIdx = 0;
      frameQueue.length = 0;
    } else {
      // 注入式连续回复（垫场→打分）：生成远快于播放，新回复 delta 到达时上一段
      // 还在播。此时绝不能重锚+清队——上一段的真帧被扔掉、数字人又不会补发，
      // 视频就会半路定格（音频还在放）。音频链是连续的，帧按到达顺序从属同一
      // 时钟即可；只砍掉旧回复的"闭嘴尾帧"（零填充生成，对应播放中不存在的静音段）
      const oldTotalFrames = Math.floor((prevEnd - responseAudioBase) * 25);
      const keep = Math.max(0, oldTotalFrames - videoFrameIdx);
      if (frameQueue.length > keep) frameQueue.length = keep;
    }
    needVideoBase = false;
  }
}

function flushPlayback() {
  // 打断：整个 AudioContext 关掉重建，已排程的音频全部作废
  if (player) {
    player.ctx.close();
    player = null;
  }
  activeSources.clear();  // ctx 已关，引用一并释放
}

// 播放是否已排空：无播放器（未开始/已关闭）或已排程的最后一个音频节点已播完
function isAudioPlaybackDrained() {
  return !player || player.nextStartTime <= player.ctx.currentTime;
}

// 本轮回复结束且声音播完后，恢复向 s2s 送麦克风音频
function maybeResumeMicInput() {
  if (!micInputPaused || !resumeMicOnDrain) return;
  if (mic && isAudioPlaybackDrained()) {
    micInputPaused = false;
    resumeMicOnDrain = false;
    console.info("[mic] TTS 播放排空，恢复麦克风收音");
  }
}

// ---------------------------------------------------------------------------
// 麦克风（AudioWorklet 采集，16kHz mono）
// ---------------------------------------------------------------------------

const WORKLET_SRC = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
`;

async function startMic() {
  // 若上一轮 TTS 还在播放，开麦后继续等待播完再收音（保持“回答期间不插话”）
  micInputPaused = !isAudioPlaybackDrained();
  resumeMicOnDrain = micInputPaused;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await ctx.audioWorklet.addModule(
    URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }))
  );
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => {
    if (micInputPaused) return;  // ASR final → LLM/TTS 播放期间暂停收音
    if (ws && ws.readyState === WebSocket.OPEN) {
      const int16 = floatTo16BitPCM(e.data);
      ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: base64FromInt16(int16) }));
    }
  };
  source.connect(node);
  const gain = ctx.createGain();
  gain.gain.value = 0;
  node.connect(gain);
  gain.connect(ctx.destination);
  mic = { ctx, stream, node };
}

function stopMic() {
  micInputPaused = false;
  resumeMicOnDrain = false;
  if (!mic) return;
  mic.node.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close();
  mic = null;
}

// ---------------------------------------------------------------------------
// 对话区（气泡样式：name + 气泡文本，无头像图标）
// ---------------------------------------------------------------------------

let assistantBody = null;   // 当前助手气泡的消息体（msg-body），译文行挂这里
let translationLine = null; // 当前气泡的译文行（msg-translation，vox.translation.delta 流式追加）

function addLine(cls, who, text) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  if (cls === "sys") {
    div.textContent = text;  // 系统提示：居中一行
    els.transcript.appendChild(div);
    els.transcript.scrollTop = els.transcript.scrollHeight;
    return div;
  }
  const body = document.createElement("div");
  body.className = "msg-body";
  if (who) {
    const name = document.createElement("div");
    name.className = "msg-name";
    name.textContent = who;
    body.appendChild(name);
  }
  const txt = document.createElement("div");
  txt.className = "msg-text";
  txt.textContent = text;
  body.appendChild(txt);
  div.appendChild(body);
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  if (cls === "assistant") {
    assistantBody = body;      // 新助手气泡：译文行挂到这个 body 下
    translationLine = null;
  }
  return txt;  // 返回文本容器，流式增量直接 append
}

function appendAssistantDelta(delta) {
  if (!assistantLine) {
    const name = (personas.find((p) => p.id === currentPersona) || {}).name || "助手";
    assistantLine = addLine("assistant", name, "");
  }
  assistantLine.appendChild(document.createTextNode(delta));
  assistantJaText += delta;  // 供 live2d 情绪 agent 判断
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function appendTranslationDelta(delta) {
  if (!assistantBody) return;  // 无助手气泡（异常/被清空）则丢弃
  if (!translationLine) {
    translationLine = document.createElement("div");
    translationLine.className = "msg-translation";
    assistantBody.appendChild(translationLine);
  }
  translationLine.appendChild(document.createTextNode(delta));
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

// ---------------------------------------------------------------------------
// 数字人画面
// ---------------------------------------------------------------------------

// 画布上下文惰性创建：alpha:false 跳过上屏混合、desynchronized 降撕裂
//（浏览器不支持会自动忽略）；平滑质量拉满——移动端糊就是默认平滑质量太低。
// 必须惰性：2dlive 模式下 canvas 归 yukino2d 引擎 getContext("2d") 用，
// 模块加载时抢先 getContext 会让引擎拿不到上下文（同一 canvas 只能
// 绑定一种上下文类型），数字人直接起不来。
let avatarCtx = null;
function getAvatarCtx() {
  if (!avatarCtx) {
    avatarCtx = els.canvas.getContext("2d", { alpha: false, desynchronized: true });
    avatarCtx.imageSmoothingEnabled = true;
    avatarCtx.imageSmoothingQuality = "high";
  }
  return avatarCtx;
}

let frameDecodeMs = 0;
let frameDecodeCount = 0;

async function drawFrame(jpegBytes) {
  if (mode2d) return;  // 2dlive 本地渲染，JPEG 帧链路停用
  try {
    const t0 = performance.now();
    const bitmap = await createImageBitmap(new Blob([jpegBytes], { type: "image/jpeg" }));
    frameDecodeMs += performance.now() - t0;
    frameDecodeCount++;
    getAvatarCtx().drawImage(bitmap, 0, 0, els.canvas.width, els.canvas.height);
    bitmap.close();
  } catch {
    /* 坏帧丢弃 */
  }
}

// ---------------------------------------------------------------------------
// 2dlive 模式：Live2D Cubism 4 前端 WebGL 渲染 + TTS 播放链 RMS 口型驱动
// ---------------------------------------------------------------------------

function init2DLive() {
  if (yukino2d) return;  // 幂等：init() 与 vox.status 都可能触发
  mode2d = true;
  document.body.classList.add("live2d");
  // RMS→开合度映射沿用 viseme 校准值（web/yukino2d.js 内嵌默认参数）
  mouthTracker = window.createOpennessTracker();
  try {
    yukino2d = window.initYukino2D(els.canvas, { model: LIVE2D_MODEL_URL });
    window.yukino2dAgent = yukino2d;  // agent 动作 API（playMotion/setExpression/setPose）
  } catch (e) {
    console.error("[2dlive] 引擎初始化失败:", e);
    mode2d = false;
    return;  // 兜底走静态肖像（showStill）
  }
  start2DLoop();
  showPerfUI();
}

function start2DLoop() {
  const rmsData = new Float32Array(512);  // 与 analyser.fftSize 一致
  const tick = () => {
    requestAnimationFrame(tick);
    maybeResumeMicInput();
    const p = player;
    let rms = 0;
    if (p && p.analyser) {  // 打断 flushPlayback 后 player=null → 喂 RMS=0 嘴回落
      p.analyser.getFloatTimeDomainData(rmsData);  // 必须浮点域：字节域静音 RMS≈0.0045
      let s = 0;                                   // > floor 0.003，嘴会卡在 ~0.09 开合
      for (let i = 0; i < rmsData.length; i++) s += rmsData[i] * rmsData[i];
      rms = Math.sqrt(s / rmsData.length);
    }
    if (yukino2d) yukino2d.setMouth(mouthTracker.step(rms));  // 服装切换重建期间短暂无引擎
    if (avatarState === "speaking" && p && p.nextStartTime <= p.ctx.currentTime) {
      setAvatarState("idle");  // 说话排空→待机角标（复用原逻辑）
    }
  };
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Live2D 情绪 agent：根据本轮助手日语内容，选择表情/动作。
// 情绪 → 表情 index 映射按模型 expressions 顺序：
// 0 平静 1 微笑 2 认真 3 冷淡 4 无奈 5 汗颜 6 审视 7 侧目
// 8 惊讶 9 不悦 10 困扰 11 害羞 12 脸红 13 轻笑
// ---------------------------------------------------------------------------
function detectYukinoEmotion(text) {
  const t = text || "";
  if (/(怒|ムカ|不愉快|いい加減|ふざけ|バカ|アホ|最低|許さない)/.test(t)) {
    return { emotion: "angry", expression: 9, motion: "yuk_ikari" };
  }
  if (/(えっ|まさか|嘘|驚|びっくり)/.test(t)) {
    return { emotion: "surprise", expression: 8, motion: null };
  }
  if (/(なぜ|どうして|分からない|は？|何の話|意味が分から|質問|どういう)/.test(t)) {
    return { emotion: "confused", expression: 10, motion: "yuk_hatena" };
  }
  if (/(悲|泣|つらい|寂|切ない|涙)/.test(t)) {
    return { emotion: "sad", expression: 0, motion: "yuk_uruuru" };
  }
  if (/(ふふ|笑|楽しい|うれしい|良かった|よかった|嬉しい)/.test(t)) {
    return { emotion: "happy", expression: 1, motion: null };
  }
  if (/(呆|やれやれ|ため息|しょうがない|仕方ない|全くもう)/.test(t)) {
    return { emotion: "tired", expression: 4, motion: "yuk_sweat" };
  }
  if (/(恥|照|べつに|勘違いしないで|勘違い)/.test(t)) {
    return { emotion: "shy", expression: 11, motion: null };
  }
  return { emotion: "calm", expression: 0, motion: null };
}

// 情绪 → 编排序列：动作时长取作者 motion 预置（Meta.Duration），衔接过渡由
// 引擎交叉淡入淡出保留。index 与 model3.json Action 顺序一致：
// 15 汗颜/16 泪眼/17 阴沉/18 疑问/19 生气/20 挑眉（头部特效）；6=1B 闭眼低头 11=1C 闭眼低头。
const EMOTION_CHOREO = {
  angry:    [{ type: "expression", value: 9 }, { type: "motion", value: 19 }],
  surprise: [{ type: "expression", value: 8 }, { type: "pause", ms: 900 }, { type: "expression", value: 0 }],
  confused: [{ type: "expression", value: 10 }, { type: "motion", value: 18 }],
  sad:      [{ type: "expression", value: 0 }, { type: "motion", value: 16 }],
  happy:    [{ type: "expression", value: 1 }, { type: "motion", value: 6 }, { type: "expression", value: 1 }],
  tired:    [{ type: "expression", value: 4 }, { type: "motion", value: 15 }],
  shy:      [{ type: "expression", value: 11 }, { type: "motion", value: 11 }],
  calm:     [{ type: "expression", value: 0 }],
};
// 抽屉「试演」用的示例编排（验证流程与衔接过渡）
const SAMPLE_CHOREO = [
  { type: "expression", value: 1 },
  { type: "motion", value: 6 },
  { type: "expression", value: 10 },
  { type: "motion", value: 18 },
  { type: "expression", value: 0 },
  { type: "pose", value: "pose2" },
];

function runLive2dEmotionAgent(text) {
  if (!yukino2d || !yukino2d.isReady()) return;
  const t = (text || "").trim();
  if (!t) return;
  const choice = detectYukinoEmotion(t);
  if (choice.expression !== undefined && choice.expression !== null) {
    updateExpressionHighlight(choice.expression);  // 演出抽屉表情高亮同步
  }
  // 动作流程演出：按情绪播一小段编排（时长取动作预置，衔接过渡保留）
  const seq = EMOTION_CHOREO[choice.emotion];
  if (seq) {
    yukino2d.playChoreography(seq);
  } else if (choice.expression !== undefined && choice.expression !== null) {
    yukino2d.setExpression(choice.expression);
  }
  if (choice.motion && !seq) {
    yukino2d.playMotion(choice.motion, 3);
  }
  console.info(
    "[live2d-agent] 情绪: %s | 编排: %s | 文本: %s",
    choice.emotion, seq ? seq.length + " 步" : "-", t.slice(0, 24)
  );
}

// ---------------------------------------------------------------------------
// 演出控制（作者 demo 同款交互）：目光跟随 / 模拟说话 / 服装 / 表情 / 动作。
// 只在 2dlive 引擎就绪后构建；图标按钮 → 抽屉。动作/表情 index 与
// model3.json 的 Action / Expressions 顺序一致。
// ---------------------------------------------------------------------------
const EXPRESSION_LABELS = [
  "平静", "微笑", "认真", "冷淡", "无奈", "汗颜", "审视", "侧目",
  "惊讶", "不悦", "困扰", "害羞", "脸红", "轻笑",
];
const ACTION_LABELS = [
  "姿态A·撩头发", "1A 闭眼低头", "2A 闭眼摇头", "3A 向右摇头", "4A 闭眼叹气",
  "姿态B·搭手", "1B 闭眼低头", "2B 闭眼摇头", "3B 向右摇头", "4B 闭眼叹气",
  "姿态C·叉腰思考", "1C 闭眼低头", "2C 闭眼摇头", "3C 向右摇头", "4C 闭眼叹气",
  "汗颜", "泪眼", "阴沉", "疑问", "生气", "挑眉",
  "A→B", "A→C", "B→A", "B→C", "C→A", "C→B",
];
const POSE_ACTION_INDEXES = new Set([0, 5, 10]);  // 姿态A/B/C：点击同时切待机姿势
let perfBuilt = false;
let expressionChips = [];       // 表情 chip 引用（高亮用）
let currentExpression = 0;      // 当前表情 index

function updateExpressionHighlight(idx) {
  currentExpression = idx;
  expressionChips.forEach((chip, i) => chip.classList.toggle("active", i === idx));
}

function togglePerfDrawer() {
  const open = els.perfDrawer.classList.toggle("open");
  els.perfToggle.classList.toggle("active", open);
}

function buildPerformanceUI() {
  if (perfBuilt || !yukino2d) return;
  perfBuilt = true;
  const drawer = els.perfDrawer;
  drawer.innerHTML = "";

  // 头：标题 + 关闭
  const head = document.createElement("div");
  head.className = "perf-head";
  const title = document.createElement("span");
  title.textContent = "演出控制";
  const close = document.createElement("button");
  close.type = "button";
  close.className = "perf-close";
  close.textContent = "✕";
  close.addEventListener("click", togglePerfDrawer);
  head.append(title, close);
  drawer.appendChild(head);

  // 行 1：目光跟随开关 + 模拟说话
  const row1 = document.createElement("div");
  row1.className = "perf-row";
  const followBtn = document.createElement("button");
  followBtn.type = "button";
  followBtn.className = "perf-chip perf-chip-follow active";
  followBtn.textContent = "目光跟随 · 开";
  followBtn.addEventListener("click", () => {
    const on = !followBtn.classList.contains("active");
    if (yukino2d) yukino2d.setFollow(on);
    followBtn.classList.toggle("active", on);
    followBtn.textContent = on ? "目光跟随 · 开" : "目光跟随 · 关";
  });
  const talkBtn = document.createElement("button");
  talkBtn.type = "button";
  talkBtn.className = "perf-chip";
  talkBtn.textContent = "模拟说话";
  talkBtn.addEventListener("click", () => { if (yukino2d) yukino2d.talkDemo(4); });
  const demoBtn = document.createElement("button");
  demoBtn.type = "button";
  demoBtn.className = "perf-chip";
  demoBtn.textContent = "试演";
  demoBtn.title = "播放示例编排（表情→动作→表情→动作→姿态，含衔接过渡）";
  demoBtn.addEventListener("click", () => { if (yukino2d) yukino2d.playChoreography(SAMPLE_CHOREO); });
  row1.append(followBtn, talkBtn, demoBtn);
  drawer.appendChild(row1);

  // 行 2：服装 制服/私服
  const row2 = document.createElement("div");
  row2.className = "perf-row";
  ["seihuku", "shihuku"].forEach((o) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "perf-chip perf-chip-outfit" + (LIVE2D_MODEL_URL.includes(o) ? " active" : "");
    b.textContent = o === "seihuku" ? "制服" : "私服";
    b.addEventListener("click", () => switchOutfit(o, b));
    row2.appendChild(b);
  });
  drawer.appendChild(row2);

  // 表情（14）
  const expLabel = document.createElement("div");
  expLabel.className = "perf-section";
  expLabel.textContent = "表情";
  drawer.appendChild(expLabel);
  const expGrid = document.createElement("div");
  expGrid.className = "perf-grid";
  expressionChips = EXPRESSION_LABELS.map((label, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "perf-chip";
    b.textContent = label;
    b.addEventListener("click", () => {
      if (!yukino2d) return;
      yukino2d.setExpression(i);
      updateExpressionHighlight(i);
    });
    expGrid.appendChild(b);
    return b;
  });
  drawer.appendChild(expGrid);

  // 动作 / 姿态（27）
  const actLabel = document.createElement("div");
  actLabel.className = "perf-section";
  actLabel.textContent = "动作 / 姿态";
  drawer.appendChild(actLabel);
  const actGrid = document.createElement("div");
  actGrid.className = "perf-grid";
  ACTION_LABELS.forEach((label, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "perf-chip" + (POSE_ACTION_INDEXES.has(i) ? " perf-chip-pose" : "");
    b.textContent = label;
    b.addEventListener("click", () => {
      if (!yukino2d) return;
      if (POSE_ACTION_INDEXES.has(i)) {
        // 姿态A/B/C：切待机姿势 + 暂停自动轮换（用户手动选择优先）
        const poseName = i === 0 ? "poseA" : i === 5 ? "poseB" : "poseC";
        yukino2d.setPose(poseName);
        yukino2d.setAutoPose(false);
      } else {
        yukino2d.playMotion(i, 3);
      }
    });
    actGrid.appendChild(b);
  });
  drawer.appendChild(actGrid);

  updateExpressionHighlight(0);
}

function showPerfUI() {
  if (!mode2d || !els.perfToggle) return;
  buildPerformanceUI();
  els.perfToggle.classList.remove("hidden");
  els.perfToggle.addEventListener("click", togglePerfDrawer);
}

// 服装切换：重建 Live2D 引擎（destroy → 同 canvas 重 init），保留目光跟随状态
function switchOutfit(outfit, btn) {
  if (!mode2d || !yukino2d) return;
  const url = `/static/live2d/models/yukino/yukino_${outfit}.model3.json?v=${MODEL_VERSION}`;
  if (url === LIVE2D_MODEL_URL) return;
  const followOn = els.perfDrawer.querySelector(".perf-chip-follow")?.classList.contains("active") ?? true;
  try { yukino2d.destroy(); } catch (e) { console.warn("[perf] destroy 失败:", e); }
  yukino2d = null;
  mouthTracker = window.createOpennessTracker();
  try {
    yukino2d = window.initYukino2D(els.canvas, { model: url });
    window.yukino2dAgent = yukino2d;
    LIVE2D_MODEL_URL = url;
    yukino2d.setFollow(followOn);
  } catch (e) {
    console.error("[2dlive] 服装切换引擎重建失败:", e);
    yukino2d = null;
    return;
  }
  els.perfDrawer.querySelectorAll(".perf-chip-outfit").forEach((chip) => chip.classList.remove("active"));
  btn.classList.add("active");
  updateExpressionHighlight(0);
}

// 视频帧队列 + 音画对齐：帧按到达顺序编号（每个 response 从 0 起），
// 播放计时器按"音频已播秒数 × 25fps"放帧；视频落后音频 >1s 时跳帧追赶
const FRAME_QUEUE_MAX = 1000;  // ~40s 内容。供帧天然快于播放(TTS 1.5x 流式),
                               // 队列会持续增长,必须给足深度;丢帧绝不递增序号
                               // (否则嘴型整体超前,越丢越乱)
const frameQueue = [];
let frameTimer = null;
let responseAudioBase = 0;   // 当前 response 音频在 ctx 时间轴上的起点
let videoFrameIdx = 0;       // 当前 response 已消费（播放或丢弃）的帧序号
let needVideoBase = true;    // 下一个音频 delta 是 response 起点（response.done/打断后置位）
const idleQueue = [];        // idle 帧缓冲（无音频时钟，按 ~25fps 均匀释放）
let lastIdleDraw = 0;

function enqueueFrame(jpegBytes) {
  frameRecvCount++;
  if (frameQueue.length >= FRAME_QUEUE_MAX) {
    frameQueue.shift();  // 极端情况丢最旧帧:嘴型最多滞后,绝不超前(滞后比超前自然)
    return;
  }
  frameQueue.push(jpegBytes);
}

function startFramePlayback() {
  if (frameTimer || mode2d) return;  // 2dlive 本地渲染：JPEG 帧播放链路停用
  // rAF 驱动(60Hz,跟屏幕刷新):setInterval(40ms) 在 iOS 上漂移严重,
  // 放帧跟不上音频时钟,越落越多再跳帧,表现为一卡一卡
  const tick = () => {
    frameTimer = requestAnimationFrame(tick);
    maybeResumeMicInput();
    // 音频播放排空：「说话」结束回待机（角标隐藏；句尾后的 idle 帧已排在
    // 说话帧队列尾沿时钟连播，见下行二进制 handler 的注释，无需在此切换通道）
    if (avatarState === "speaking" && player && player.nextStartTime <= player.ctx.currentTime) {
      setAvatarState("idle");
    }
    // idle 帧均匀释放：服务端 0.2s 一簇 5 帧推流，到就画会成簇卡顿
    //（说话结束从时钟播放切换到直画的那一刻尤其明显）
    const now = performance.now();
    if (idleQueue.length > 0 && now - lastIdleDraw >= 38) {
      lastIdleDraw = now;
      drawFrame(idleQueue.shift());
    }
    if (!player || frameQueue.length === 0) return;
    const pos = player.ctx.currentTime - responseAudioBase;  // 本 response 已播音频秒数
    if (pos < 0) return;
    const target = Math.floor(pos * 25) - videoLagFrames;
    // 落后 >1s：跳到最新，保同步优先于完整
    while (frameQueue.length > 0 && videoFrameIdx < target - 25) {
      frameQueue.shift();
      videoFrameIdx++;
    }
    if (frameQueue.length > 0 && videoFrameIdx <= target) {
      drawFrame(frameQueue.shift());
      videoFrameIdx++;
    }
  };
  frameTimer = requestAnimationFrame(tick);
}

// 调试角标（URL 加 ?debug=1 开启）：音频缓冲时长 / 帧队列深度 / 实际到帧率
let frameRecvCount = 0;
let frameRecvWindowStart = performance.now();
if (new URLSearchParams(location.search).has("debug")) {
  const dbg = document.createElement("div");
  dbg.style.cssText =
    "position:fixed;right:8px;bottom:8px;background:#000c;color:#0f0;" +
    "font:12px monospace;padding:6px 10px;border-radius:6px;z-index:99;white-space:pre";
  document.body.appendChild(dbg);
  setInterval(() => {
    const now = performance.now();
    const fps = frameRecvCount / ((now - frameRecvWindowStart) / 1000);
    frameRecvCount = 0;
    frameRecvWindowStart = now;
    const audioBuf = player ? Math.max(0, player.nextStartTime - player.ctx.currentTime) : 0;
    const pos = player ? Math.max(0, player.ctx.currentTime - responseAudioBase) : 0;
    const decodeAvg = frameDecodeCount ? (frameDecodeMs / frameDecodeCount) : 0;
    frameDecodeMs = 0;
    frameDecodeCount = 0;
    dbg.textContent =
      `audio缓冲: ${audioBuf.toFixed(2)}s\n帧队列: ${frameQueue.length}\n到帧率: ${fps.toFixed(1)}fps\n` +
      `解码: ${decodeAvg.toFixed(0)}ms/帧\n音频位置: ${pos.toFixed(2)}s\n帧序号: ${videoFrameIdx} (目标 ${Math.floor(pos * 25)})`;
  }, 1000);
}

function showStill(personaId) {
  if (mode2d) return;  // 2dlive 本地渲染立绘，不叠加静态肖像
  const persona = personas.find((p) => p.id === personaId);
  if (persona && persona.has_image) {
    els.still.src = `/api/personas/${personaId}/image`;
    els.still.classList.remove("hidden");
  } else {
    els.still.removeAttribute("src");
  }
}

// ---------------------------------------------------------------------------
// 对话状态角标：listening（用户说话中）/ thinking（说完到开口前）显示角标，
// speaking / idle 隐藏。画面动感由 avatar 服务驱动：倾听/思考时循环 persona
// 嘟囔音频（TTS 预合成）产生真实沉吟/附和微动，待机时纯静音基线微动
// ---------------------------------------------------------------------------

let avatarState = "idle"; // idle | listening | thinking | speaking

function setAvatarState(state) {
  avatarState = state;
  const el = els.avatarState;
  if (state === "listening") {
    el.textContent = "👂 倾听中…";
    el.className = "state-listening";
  } else if (state === "thinking") {
    el.textContent = "🤔 思考中…";
    el.className = "state-thinking";
  } else {
    el.className = "hidden";
  }
}

// ---------------------------------------------------------------------------
// Realtime 事件处理
// ---------------------------------------------------------------------------

const realtimeHandlers = {
  "input_audio_buffer.speech_started"() {
    // 用户开口（打断）：本地播放队列清空，助手文本行封口
    flushPlayback();
    assistantJaText = "";
    frameQueue.length = 0;  // 视频帧队列一并清空，嘴型跟着归位
    idleQueue.length = 0;
    needVideoBase = true;
    assistantLine = null;
    setAvatarState("listening");
  },
  "input_audio_buffer.speech_stopped"() {
    // 用户说完：到助手首个音频 delta 之前是「思考」窗口
    if (avatarState === "listening") setAvatarState("thinking");
  },
  "conversation.item.input_audio_transcription.completed"(event) {
    const text = (event.transcript || "").trim();
    if (text) addLine("user", "你", text);
    // ASR final：开始 LLM/TTS 前暂停收音，直到本轮声音播放完毕
    micInputPaused = true;
    resumeMicOnDrain = false;
    console.info("[mic] ASR final，暂停收音，等待本轮回复播放完成");
  },
  "response.output_audio_transcript.delta"(event) {
    if (event.delta) {
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_text.delta"(event) {
    if (event.delta) {
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_audio_transcript.done"(event) {
    // 音频模式下上游不发 delta,只在 done 里带整段文本——必须在这里显示
    if (event.transcript) {
      appendAssistantDelta(event.transcript);
    }
    assistantLine = null;
  },
  "response.output_text.done"() {
    assistantLine = null;
  },
  "response.output_audio.delta"(event) {
    if (event.delta) {
      if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
      playPCM(int16FromBase64(event.delta));
    }
  },
  "response.audio.delta"(event) {
    if (event.delta) {
      if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
      playPCM(int16FromBase64(event.delta));
    }
  },
  "response.done"() {
    assistantLine = null;
    needVideoBase = true;  // 下一个音频 delta 开启新 response,重设视频对齐基准
    if (avatarState === "thinking") setAvatarState("idle");  // 无音频回复的兜底
    resumeMicOnDrain = true;   // 回复已结束：等已排程音频播完即恢复收音
    maybeResumeMicInput();
    // LLM 自带【演出】编排时跳过默认情绪演出（vox.choreo 已先行播放）
    if (aiChoreoPending) {
      console.info("[live2d-agent] 本轮 AI 自带演出编排，跳过默认情绪演出");
    } else {
      runLive2dEmotionAgent(assistantJaText);
    }
    clearTimeout(aiChoreoResetTimer);
    aiChoreoPending = false;
    assistantJaText = "";
  },
  error(event) {
    addLine("sys", "", `⚠ ${(event.error && event.error.message) || "未知错误"}`);
    resumeMicOnDrain = true;   // 出错也恢复收音（若还有残存音频，等播完）
    maybeResumeMicInput();
  },
};

function handleTextMessage(data) {
  let event;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (event.type === "vox.translation.delta") {
    // 双语模式：LLM 日语的逐句中文翻译（管线 on_assistant_text 拆出），
    // 追加到当前助手气泡的译文行
    appendTranslationDelta(event.delta || "");
    return;
  }
  if (event.type === "vox.task-done") {
    // DSH 插件推送：雪乃 AI 总结后的任务完成消息，日语正文 + 中文译文
    const text = (event.text || "").trim();
    const translation = (event.translation || "").trim();
    if (text) {
      const name = (personas.find((p) => p.id === currentPersona) || {}).name || "雪乃";
      addLine("assistant", name, text);
      if (translation) appendTranslationDelta(translation);
      assistantJaText = text;  // 供 live2d 情绪 agent 使用
      assistantLine = null;
    }
    return;
  }
  if (event.type === "vox.task-done-audio" && event.pcm) {
    // 后端用雪乃 GPT-SoVITS 合成好的音频：直接走播放链，口型同步
    try {
      playPCM(int16FromBase64(event.pcm));
      if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
    } catch (e) {
      console.warn("[task-done-audio] 播放失败:", e);
    }
    return;
  }
  if (event.type === "vox.live2d") {
    // agent 动作协议：{type:"vox.live2d", action:"motion"|"expression"|"pose", name:"...", priority?:3}
    if (yukino2d) {
      if (event.action === "expression") yukino2d.setExpression(event.name);
      else if (event.action === "pose") yukino2d.setPose(event.name);
      else if (event.action === "motion") yukino2d.playMotion(event.name, event.priority);
    }
    return;
  }
  if (event.type === "vox.choreo" && Array.isArray(event.steps)) {
    // LLM 自带的【演出】编排（管线或 task-done 解析后下发）：按顺序播 表情/动作/姿态/停顿。
    // 管线对话里它排在 response.done 之前 → 本轮跳过默认情绪演出；
    // task-done 编排没有后续 response.done，3s 后自动清除，避免误吞下一轮
    if (yukino2d) {
      yukino2d.playChoreography(event.steps);
      aiChoreoPending = true;
      clearTimeout(aiChoreoResetTimer);
      aiChoreoResetTimer = setTimeout(() => { aiChoreoPending = false; }, 3000);
      console.info("[vox.choreo] 编排 %d 步:", event.steps.length, event.steps);
    }
    return;
  }
  if (event.type === "vox.status") {
    if (event.avatar_backend === "2dlive") {
      // yukino2d 前端本地渲染：无 avatar ws、无 JPEG 帧，状态按 on 处理
      avatarOn = true;
      currentPersona = event.persona;
      init2DLive();
      els.fallback.classList.add("hidden");
      updatePersonaBar();
      return;
    }
    avatarOn = event.avatar === "on";
    currentPersona = event.persona;
    if (ADELAY_OVERRIDE == null && event.avatar_backend) {
      avatarAudioDelay = BACKEND_ADELAY[event.avatar_backend] ?? 0.8;
    }
    if (VLAG_OVERRIDE == null && event.avatar_backend) {
      videoLagFrames = BACKEND_VLAG[event.avatar_backend] ?? 0;
    }
    els.fallback.classList.toggle("hidden", avatarOn);
    updatePersonaBar();
    showStill(currentPersona);
    return;
  }
  const handler = realtimeHandlers[event.type];
  if (handler) handler(event);
}

// ---------------------------------------------------------------------------
// 连接与人设
// ---------------------------------------------------------------------------

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls || ""}`;
}

function updatePersonaBar() {
  // 只有一个人设时隐藏切换条（chip 标签没意义）；多个人设自动恢复
  els.personaBar.style.display = personas.length > 1 ? "" : "none";
  els.personaBar.innerHTML = "";
  for (const p of personas) {
    const chip = document.createElement("button");
    chip.className = "persona-chip" + (p.id === currentPersona ? " active" : "");
    chip.textContent = p.name;
    chip.onclick = () => switchPersona(p.id);
    els.personaBar.appendChild(chip);
  }
  const cur = personas.find((p) => p.id === currentPersona);
  if (cur) els.avatarLabel.textContent = cur.label || cur.name;
}

function switchPersona(id) {
  if (!ws || ws.readyState !== WebSocket.OPEN || id === currentPersona) return;
  ws.send(JSON.stringify({ type: "vox.persona", id }));
  currentPersona = id;
  assistantLine = null;
  updatePersonaBar();
  showStill(id);
}

// ---------------------------------------------------------------------------
// 对话历史（同页抽屉，数据来自 /api/history，SQLite 本地 log/chat_history.db）
// ---------------------------------------------------------------------------
let historyMode = "list";  // list | detail
let historySessions = [];

function openHistoryView() {
  document.body.classList.add("history-open");
  loadHistorySessions();
  // 让 Live2D 重新 fit 到「左侧栏打开后的可视区域」
  window.dispatchEvent(new Event("resize"));
}

function closeHistoryView() {
  document.body.classList.remove("history-open");
  historyMode = "list";
  els.historyTitle.textContent = "对话历史";
  // 让 Live2D 重新 fit 到全可视区域
  window.dispatchEvent(new Event("resize"));
}

function newConversation() {
  // 通知 orchestrator 开始新会话（生成新 session_id，后续消息写入新历史）
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "vox.new_session" }));
  }
  // 清空当前聊天界面与轮次状态
  els.transcript.innerHTML = "";
  assistantLine = null;
  assistantBody = null;
  translationLine = null;
  assistantJaText = "";
  closeHistoryView();
  setStatus("新会话", "");
  updatePersonaBar();
}

async function loadHistorySessions() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    historySessions = data.enabled ? (data.conversations || []) : [];
  } catch (e) {
    historySessions = [];
  }
  historyMode = "list";
  els.historyTitle.textContent = "对话历史";
  renderHistorySessions();
}

function renderHistorySessions() {
  const box = els.historyList;
  box.innerHTML = "";
  if (!historySessions.length) {
    box.innerHTML = '<div style="color:var(--text-light);font-size:13px;padding:12px">还没有对话记录。去说几句话再回来看。</div>';
    return;
  }
  for (const c of historySessions) {
    const item = document.createElement("div");
    item.className = "history-item";
    const title = document.createElement("div");
    title.className = "history-item-title";
    title.textContent = c.title || c.session_id;
    const meta = document.createElement("div");
    meta.className = "history-item-meta";
    meta.textContent = `${c.updated_at} · ${c.message_count} 条 · ${c.persona || ""}`;
    item.appendChild(title);
    item.appendChild(meta);
    item.onclick = () => openHistorySession(c.session_id);
    const actions = document.createElement("div");
    actions.className = "history-item-actions";
    const del = document.createElement("button");
    del.className = "history-item-delete";
    del.textContent = "🗑 删除";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("删除当前会话记录？此操作不可恢复。")) return;
      await fetch(`/api/history/${c.session_id}`, { method: "DELETE" });
      await loadHistorySessions();
    };
    actions.appendChild(del);
    item.appendChild(actions);
    box.appendChild(item);
  }
}

async function openHistorySession(sid) {
  let messages = [];
  try {
    const res = await fetch(`/api/history/${sid}`);
    const data = await res.json();
    messages = data.messages || [];
  } catch (e) {
    messages = [];
  }
  historyMode = "detail";
  const conv = historySessions.find((c) => c.session_id === sid);
  els.historyTitle.textContent = (conv && (conv.title || conv.session_id)) || sid;
  const box = els.historyList;
  box.innerHTML = "";
  const del = document.createElement("button");
  del.className = "history-detail-delete";
  del.textContent = "🗑 删除此会话";
  del.onclick = async () => {
    if (!confirm("删除当前会话记录？此操作不可恢复。")) return;
    await fetch(`/api/history/${sid}`, { method: "DELETE" });
    await loadHistorySessions();
  };
  box.appendChild(del);
  for (const m of messages) {
    const wrap = document.createElement("div");
    wrap.className = "history-msg " + (m.role === "user" ? "user" : "assistant");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = m.content;
    wrap.appendChild(bubble);
    if (m.translation) {
      const trans = document.createElement("div");
      trans.className = "trans";
      trans.textContent = "译文：" + m.translation;
      wrap.appendChild(trans);
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${m.role === "user" ? "用户" : "雪乃"} · ${m.created_at}`;
    wrap.appendChild(meta);
    box.appendChild(wrap);
  }
  box.scrollTop = box.scrollHeight;
}

function onHistoryBack() {
  if (historyMode === "detail") {
    loadHistorySessions();
  } else {
    closeHistoryView();
  }
}

els.historyBtn.onclick = () => {
  if (document.body.classList.contains("history-open")) closeHistoryView();
  else openHistoryView();
};
els.historyClose.onclick = closeHistoryView;
els.historyBack.onclick = onHistoryBack;
els.historyRefresh.onclick = loadHistorySessions;
els.historyNew.onclick = newConversation;

// 调试/入口：?history=1 打开页面时直接展开左侧历史栏
if (new URLSearchParams(location.search).has("history")) openHistoryView();

let reconnectTimer = null;
let recoverHookBound = false;

function isConnected() {
  return !!ws && ws.readyState <= 1;  // CONNECTING/OPEN
}

// 恢复连接统一入口：后台标签页不持有连接（双页互顶 → 无限断开/连上循环，
// 2026-08-19 实测），页面重新可见/聚焦时再恢复。
function maybeConnect() {
  if (document.hidden) { ensureRecoverHook(); return; }
  if (isConnected()) return;
  connect();
}

function scheduleReconnect(delayMs) {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(maybeConnect, delayMs);
}

// 只绑定一次 visibilitychange/focus：后台页隐藏时主动释放 ws（把唯一槽位
// 让给可见页，避免互顶死循环），重新可见/聚焦时恢复连接。
function ensureRecoverHook() {
  if (recoverHookBound) return;
  recoverHookBound = true;
  const onVis = () => {
    if (document.hidden) {
      if (isConnected()) { try { ws.close(1000, "hidden"); } catch (e) {} }
      return;
    }
    maybeConnect();
  };
  document.addEventListener("visibilitychange", onVis);
  window.addEventListener("focus", maybeConnect);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setStatus("已连接", "live");
    els.micBtn.disabled = false;
    if (!mode2d) startFramePlayback();
  };
  ws.onclose = (ev) => {
    els.micBtn.disabled = true;
    if (ev && ev.code === 4001) {
      // 被其他页面接管（orchestrator 顶掉会话）：后台页退避，等重新可见再连；
      // 可见页稍后重试接管（赢家往往是后台页，隐藏后会主动释放槽位）
      setStatus("已被其他页面接管", "warn");
      ensureRecoverHook();
      if (document.hidden) return;
      scheduleReconnect(2000);
      return;
    }
    setStatus("已断开", "warn");
    // 简单重连：3s 重试。无条件重连——服务器重启/会话被顶掉时，
    // 未开麦状态（mic=null）也要恢复，否则页面永久「已断开」需手动刷新
    scheduleReconnect(3000);
  };
  ws.onerror = () => setStatus("连接错误", "warn");
  ws.onmessage = (msg) => {
    if (typeof msg.data === "string") {
      handleTextMessage(msg.data);
    } else if (mode2d) {
      return;  // 2dlive 无二进制帧，防御
    } else {
      const bytes = new Uint8Array(msg.data);
      if (bytes[0] === FRAME_TYPE_JPEG) {
        // tag 0x00=idle：无音频时钟（待机微动/倾听反应），进 idleQueue 按
        // ~25fps 均匀释放（成簇直画会卡）；tag 0x01=speech：照常排队
        if (bytes[1] === FRAME_TAG_IDLE) {
          // 句尾平滑（2026-08-04 终版）：说话帧队列未空（或音频仍在播）时，idle 帧
          // 排进同一队列、沿音频时钟 25fps 连播——引擎内容本就连贯（尾帧回落→idle
          // 微动），按到达顺序播即无跳变也无断供。这正是官方 demo 的播放模型
          // （帧按内容顺序持续上屏）。完全空闲（无音频时钟）才走 idleQueue 直画。
          // 旧实现两处败笔：①积压期丢 idle 帧→接管瞬间姿态跳变；②drain 后才
          // flush→队列耗尽到尾帧到达之间定格。
          if (frameQueue.length > 0 || (player && player.nextStartTime > player.ctx.currentTime)) {
            enqueueFrame(bytes.subarray(2));
          } else {
            if (idleQueue.length >= 10) idleQueue.shift();  // 满则丢最旧
            idleQueue.push(bytes.subarray(2));
          }
        } else {
          enqueueFrame(bytes.subarray(2));
        }
      }
    }
  };
}

async function init() {
  const res = await fetch("/api/personas");
  const data = await res.json();
  personas = data.list;
  currentPersona = data.default;
  if (data.avatar_backend === "2dlive") {
    // 提前启动本地渲染：1.9MB 立绘立刻开始加载，比等 ws 首帧状态早 ~100ms
    init2DLive();
    els.fallback.classList.add("hidden");
  } else {
    avatarOn = data.avatar === "on";
    els.fallback.classList.toggle("hidden", avatarOn);
    showStill(currentPersona);
  }
  updatePersonaBar();
  setAvatarState("idle");
  restoreLatestConversation();
  connect();
}

// 进页时自动恢复最近一条对话到主区（不进历史抽屉），避免重开 yukino 是空会话。
// 雪乃是单用户单会话产品，ws 每次连接都新建 session_id，故靠 updated_at 取最近历史恢复。
async function restoreLatestConversation() {
  try {
    const res = await fetch("/api/history/latest");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.enabled || !data.conversation) return;
    const messages = data.messages || [];
    if (!messages.length) return;
    els.transcript.innerHTML = "";
    assistantLine = null;
    assistantBody = null;
    translationLine = null;
    assistantJaText = "";
    const personaName = (personas.find((p) => p.id === currentPersona) || {}).name || "雪乃";
    for (const m of messages) {
      if (m.role === "user") {
        addLine("user", "你", m.content || "");
      } else {
        const txt = addLine("assistant", personaName, m.content || "");
        if (m.translation) appendTranslationDelta(m.translation);
        assistantLine = null;
        assistantBody = null;
        translationLine = null;
      }
    }
    assistantLine = null;
    if (els.transcript.lastElementChild) {
      els.transcript.scrollTop = els.transcript.scrollHeight;
    }
  } catch (e) {
    // 恢复失败不阻塞主流程，ws 照常连
    console.warn("[restoreLatestConversation] 失败:", e);
  }
}

els.micBtn.onclick = async () => {
  if (mic) {
    stopMic();
    setAvatarState("idle");
    els.micBtn.textContent = "🎙 开始对话";
    els.micBtn.classList.remove("live");
    setStatus("已连接（麦克风关）", "");
    return;
  }
  try {
    await startMic();
    els.micBtn.textContent = "■ 结束对话";
    els.micBtn.classList.add("live");
    setStatus("聆听中", "live");
  } catch (e) {
    addLine("sys", "", `⚠ 麦克风不可用: ${e.message}`);
    return;
  }
};

// 打字输入：文本经 conversation.item.create（user input_text）进 s2s LLM，
// 再 response.create 触发回复；orchestrator 原样透传，回复走现有 response.* 事件。
function sendText() {
  const text = (els.textInput.value || "").trim();
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addLine("sys", "", "⚠ 未连接，稍后再试");
    return;
  }
  addLine("user", "你", text);
  ws.send(JSON.stringify({
    type: "conversation.item.create",
    item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
  }));
  ws.send(JSON.stringify({ type: "response.create" }));
  els.textInput.value = "";
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

if (els.sendBtn && els.textInput) {
  els.sendBtn.onclick = sendText;
  els.textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) sendText();
  });
}

// DSH 插件（基础面板）经 postMessage 控制紧凑页：新会话 / 载入指定历史会话
window.addEventListener("message", (e) => {
  const data = e.data || {};
  if (data.type === "yukino.new-session") {
    newConversation();
  } else if (data.type === "yukino.load-history" && data.sessionId) {
    loadSessionIntoTranscript(data.sessionId);
  }
});

// 把指定历史会话的文本载入主区（与 restoreLatestConversation 同款渲染；
// 当前 ws 会话不变，新消息仍写入新会话）
async function loadSessionIntoTranscript(sid) {
  let messages = [];
  try {
    const res = await fetch(`/api/history/${sid}`);
    const data = await res.json();
    messages = data.messages || [];
  } catch (e) {
    messages = [];
  }
  if (!messages.length) return;
  els.transcript.innerHTML = "";
  assistantLine = null;
  assistantBody = null;
  translationLine = null;
  assistantJaText = "";
  const personaName = (personas.find((p) => p.id === currentPersona) || {}).name || "雪乃";
  for (const m of messages) {
    if (m.role === "user") {
      addLine("user", "你", m.content || "");
    } else {
      const txt = addLine("assistant", personaName, m.content || "");
      if (m.translation) appendTranslationDelta(m.translation);
      assistantLine = null;
      assistantBody = null;
      translationLine = null;
    }
  }
  assistantLine = null;
  if (els.transcript.lastElementChild) {
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }
}

init();
