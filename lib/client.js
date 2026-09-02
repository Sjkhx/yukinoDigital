window.__ModuleLoader__.load({
  id: "yukino-dsh",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;

    var React = require("react");
    var g = require("react/jsx-runtime");

    var useEffect = React.useEffect;
    var useState = React.useState;

    function sessionTitle(s) {
      return s.displayTitle || s.title || s.id || "未命名会话";
    }

    function workspaceOf(s) {
      if (!s) return "默认工作区";
      if (typeof s.workspaceId === "string" && s.workspaceId) return s.workspaceId;
      if (s.workspace && typeof s.workspace.id === "string") return s.workspace.id;
      if (typeof s.workspace === "string" && s.workspace) return s.workspace;
      if (typeof s.cwd === "string" && s.cwd) return s.cwd;
      if (s.project && typeof s.project.name === "string") return s.project.name;
      return "默认工作区";
    }

    function projectName(s) {
      var ws = workspaceOf(s) || "";
      var parts = String(ws).split(/[\\/]/).filter(function (x) { return x; });
      return parts.length ? parts[parts.length - 1] : "默认项目";
    }

    /* 会话日志记录 → 总结文本（alpha.4 wire 形状）：
     * 记录为 {type:'event',event:{type,data}}；只认 assistant/user 消息事件，
     * 文本在 data.message.content（或 data.content）的 {type:'text',text} 块里。 */
    function recordText(rec) {
      if (!rec || rec.type !== "event" || !rec.event) return "";
      var ev = rec.event;
      if (ev.type !== "assistant/message" && ev.type !== "user/message") return "";
      var data = ev.data || {};
      var msg = data.message || data;
      var blocks = msg.content;
      if (typeof blocks === "string") return blocks;
      var out = "";
      if (Array.isArray(blocks)) {
        blocks.forEach(function (b) {
          if (b && b.type === "text" && typeof b.text === "string") out += b.text;
        });
      }
      return out;
    }

    /* 总结走 remote.session.follow 流：首帧 snapshot 自带尾部 records，
     * 不读客户端投影（投影仅 stage 会话有历史，后台会话 nodes 恒空）。 */
    async function fetchFinalSummary(ns, sessionId) {
      var stream = await ns.follow({
        address: { kind: "session", sessionId: sessionId },
        maxMessages: 40,
      });
      try {
        var res = await stream.next();
        if (res.done || !res.value) return "[yukino诊断] follow 无首帧";
        var frame = res.value;
        if (frame.type !== "snapshot") {
          return "[yukino诊断] 首帧非快照: " + frame.type;
        }
        var records = frame.records || [];
        for (var i = records.length - 1; i >= 0; i--) {
          var t = recordText(records[i]);
          if (t) return t;
        }
        var types = records.slice(-8).map(function (r) {
          return (r && r.event && r.event.type) || (r && r.type) || "?";
        });
        return "[yukino诊断] 快照无消息事件, records=" + records.length + " types=" + types.join(",");
      } finally {
        try {
          if (stream && typeof stream.dispose === "function") await stream.dispose();
          else if (stream && typeof stream.return === "function") await stream.return();
        } catch (e) { /* 已尽力关闭 */ }
      }
    }

    function postTaskDone(payload) {
      try {
        fetch("http://127.0.0.1:8000/api/yukino/task-done", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).catch(function (e) {
          console.warn("[yukino-dsh] post task-done failed:", e);
        });
      } catch (e) {
        console.warn("[yukino-dsh] post task-done failed:", e);
      }
    }

    function speakSummary(text) {
      try {
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance(text);
        u.lang = "zh-CN";
        u.rate = 1.0;
        u.pitch = 1.1;
        u.volume = 1;
        window.speechSynthesis.speak(u);
      } catch (e) {
        console.warn("[yukino-dsh] TTS failed:", e);
      }
    }

    /* 任务完成监听：低频轮询 sessions.list 快照，检测 running true→false 的会话。
     * 完全在 apply 侧执行，不进 React 渲染路径，不影响 DSH 消息流。
     *
     * 性能约束：DSH 在流式输出时每个 host/mux frame 都会触发 list 的 subscribe
     * 回调，实测在该高频推送下即使回调只置标志 + rAF 合并，仍会拖慢/卡住主线程
     * （cordis 服务代理解析 + 回调调度本身在高频下有不可忽略的累积成本）。故这里
     * 彻底不订阅，改用 setInterval 每 2 秒主动读一次快照（list 引用缓存一次，
     * 穿透 Proxy），把与 DSH 推送节奏完全解耦——代价是任务完成检测最多 2 秒延迟，
     * 对 4 秒防抖的任务完成通知无感。 */
    function createTaskDoneWatcher(ctx) {
      /* list 快照源一次性穿透 Proxy 取引用；remote 命名空间服务在 flush 时懒解析——
       * alpha.4 的 wire RPC 挂在 cordis 服务 "remote.session"（RemoteNamespaceService）上，
       * connection 服务不再持有 api。任务完成时各服务必然已启动，按名直取即可。 */
      var list = ctx.sessions.list;
      var runningById = new Map();
      var pending = new Map();
      var timer = null;

      /* 解析 remote.session 服务：先插件 ctx，失败再走会话绑定的 Agent ctx 兜底 */
      function fetchWireNamespace(sessionId, probe) {
        var contexts = [];
        if (ctx && typeof ctx.get === "function") contexts.push(ctx);
        try {
          var binding = ctx.sessions.binding(sessionId);
          if (binding && binding.ctx && typeof binding.ctx.get === "function") contexts.push(binding.ctx);
          else probe.binding = binding ? "no-ctx-get" : "undefined";
        } catch (e) {
          probe.bindErr = String((e && e.message) || e).slice(0, 120);
        }
        for (var i = 0; i < contexts.length; i++) {
          try {
            var ns = contexts[i].get("remote.session");
            if (ns && typeof ns.follow === "function") return ns;
            probe.ns = ns ? "object:" + Object.keys(ns).slice(0, 8).join("|") : "null";
          } catch (e) {
            probe.nsErr = String((e && e.message) || e).slice(0, 120);
          }
        }
        return null;
      }

      function flush() {
        timer = null;
        var items = Array.from(pending.values());
        pending.clear();
        if (!items.length) return;
        items.sort(function (a, b) { return (b.at || 0) - (a.at || 0); });
        var item = items[0];
        function post(summary) {
          postTaskDone({ project: item.project, title: item.title, summary: summary || "" });
        }
        var probe = {};
        var ns = fetchWireNamespace(item.sessionId, probe);
        if (!ns) {
          console.warn("[yukino-dsh] remote.session 不可用", probe);
          post("[yukino诊断] remote.session 不可用 " + JSON.stringify(probe).slice(0, 280));
          return;
        }
        fetchFinalSummary(ns, item.sessionId)
          .then(post)
          .catch(function (e) {
            console.warn("[yukino-dsh] 拉取任务历史失败:", e);
            post("[yukino诊断] follow异常: " + String((e && e.message) || e).slice(0, 200));
          });
      }

      function schedule() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(flush, 4000);
      }

      /* 扫描当前快照，检测 running true→false 边缘；用 for-in 避免 Object.keys 数组分配 */
      function scan() {
        var state;
        try { state = list.getSnapshot(); } catch (e) { return; }
        var byId = (state && state.byId) || {};
        for (var id in byId) {
          if (!Object.prototype.hasOwnProperty.call(byId, id)) continue;
          var s = byId[id];
          if (!s) continue;
          var isRunning = !!s.running;
          var wasRunning = runningById.get(id);
          if (wasRunning === true && isRunning === false) {
            var project = projectName(s);
            var title = sessionTitle(s);
            pending.set(project + "::" + title, {
              sessionId: id, project: project, title: title, at: Date.now(),
            });
            schedule();
          }
          runningById.set(id, isRunning);
        }
        /* 清理已从列表消失的会话（Map.forEach 中删除当前 key 安全；用 hasOwnProperty 避免原型链误判） */
        runningById.forEach(function (_, rid) {
          if (!Object.prototype.hasOwnProperty.call(byId, rid)) runningById.delete(rid);
        });
      }

      /* 初始同步：默认全部标记为当前状态，只处理之后的边缘变化，不误报旧完成 */
      try {
        var st = list.getSnapshot();
        var b = (st && st.byId) || {};
        for (var id in b) {
          if (Object.prototype.hasOwnProperty.call(b, id) && b[id]) {
            runningById.set(id, !!b[id].running);
          }
        }
      } catch (e) {}

      /* 低频轮询：每 2 秒一次，彻底脱离 DSH 推送节奏（不 subscribe） */
      var interval = setInterval(scan, 2000);
      return function () { clearInterval(interval); };
    }

    function useRoute() {
      var routeState = useState(function () {
        return window.location.pathname === "/yukino" || window.location.hash === "#/yukino";
      });
      useEffect(function () {
        function onRoute() {
          routeState[1](
            window.location.pathname === "/yukino" || window.location.hash === "#/yukino"
          );
        }
        window.addEventListener("hashchange", onRoute);
        window.addEventListener("popstate", onRoute);
        return function () {
          window.removeEventListener("hashchange", onRoute);
          window.removeEventListener("popstate", onRoute);
        };
      }, []);
      return routeState[0];
    }

    function YukinoApp() {
      var routeActive = useRoute();
      var forceState = useState(function () {
        return window.location.pathname === "/yukino" || window.location.hash === "#/yukino";
      });

      // DSH 自己的路由可能会把 /yukino 重定向回 /，这里一旦进入就保持展开，
      // 直到用户点「返回 DSH」才收起。
      useEffect(function () {
        if (routeActive && !forceState[0]) forceState[1](true);
      }, [routeActive, forceState]);

      var show = forceState[0] || routeActive;

      useEffect(function () {
        document.body.classList.toggle("yukino-route-active", show);
        return function () {
          if (!show) document.body.classList.remove("yukino-route-active");
        };
      }, [show]);

      if (!show) return null;

      return g.jsxs("div", {
        className: "yukino-route",
        "data-route": "yukino",
        children: [
          g.jsx("button", {
            type: "button",
            className: "yukino-route-back",
            onClick: function () {
              forceState[1](false);
              if (window.history.length > 1) {
                window.history.back();
              } else {
                window.location.href = "/";
              }
            },
            children: "← 返回 DSH",
          }),
          g.jsx("iframe", {
            className: "yukino-live2d-frame",
            src: "http://127.0.0.1:8000/",
            title: "Yukino",
            allow: "microphone; autoplay",
            scrolling: "no",
          }),
        ],
      });
    }

    function syncBodyClass() {
      try {
        var active = window.location.pathname === "/yukino" || window.location.hash === "#/yukino";
        document.body.classList.toggle("yukino-route-active", active);
      } catch (e) {}
    }

    /* 基础模式（紧凑面板）：DSH 右侧 1/4 屏竖屏，上部 Live2D、下部对话+打字。
     * 直接 DOM 注入（DSH 无 sidebar 插槽）；全屏 /yukino 路由激活时自动收起，
     * 避免两个 ws 会话同时占 s2s 唯一槽位。隐藏必须卸载 iframe（不能只 CSS hide）。 */
    function initCompactPanel() {
      var KEY = "yukino.panel.hidden";
      var panel = null;
      var toggleBtn = null;

      function isYukinoRoute() {
        return window.location.pathname === "/yukino" || window.location.hash === "#/yukino";
      }

      /* 向紧凑面板 iframe 发控制消息（新会话 / 载入历史）；initCompactPanel 作用域共享 */
      function postToFrame(msg) {
        var f = panel && panel.querySelector(".yukino-panel-frame");
        if (f && f.contentWindow) {
          try { f.contentWindow.postMessage(msg, "http://127.0.0.1:8000"); } catch (e) {}
        }
      }

      function makePanel() {
        var el = document.createElement("div");
        el.className = "yukino-panel";

        var head = document.createElement("div");
        head.className = "yukino-panel-head";

        var newBtn = document.createElement("button");
        newBtn.type = "button";
        newBtn.className = "yukino-panel-btn";
        newBtn.title = "新会话";
        newBtn.textContent = "＋";
        newBtn.addEventListener("click", function () {
          postToFrame({ type: "yukino.new-session" });
        });
        head.appendChild(newBtn);

        var hisBtn = document.createElement("button");
        hisBtn.type = "button";
        hisBtn.className = "yukino-panel-btn";
        hisBtn.title = "历史会话";
        hisBtn.textContent = "🕘";
        hisBtn.addEventListener("click", function () { toggleHistory(); });
        head.appendChild(hisBtn);

        var name = document.createElement("span");
        name.className = "yukino-panel-name";
        name.textContent = "雪乃";
        head.appendChild(name);

        var pop = document.createElement("button");
        pop.type = "button";
        pop.className = "yukino-panel-btn";
        pop.title = "跳出到 yukino（新窗口打开独立页）";
        pop.textContent = "⤢";
        pop.addEventListener("click", function () {
          unmount(false);  // 断开主界面雪乃（关闭 iframe ws，释放 s2s 槽位）
          // 等旧 ws 关闭、s2s 槽位释放后再开新窗口，避免单独路由连不上
          setTimeout(function () { window.open("http://127.0.0.1:8000/", "_blank"); }, 600);
        });
        head.appendChild(pop);

        var close = document.createElement("button");
        close.type = "button";
        close.className = "yukino-panel-btn yukino-panel-close";
        close.title = "隐藏面板";
        close.textContent = "✕";
        close.addEventListener("click", function () { unmount(true); });
        head.appendChild(close);

        el.appendChild(head);

        var frame = document.createElement("iframe");
        frame.className = "yukino-panel-frame";
        frame.src = "http://127.0.0.1:8000/?compact=1";
        frame.title = "Yukino";
        frame.allow = "microphone; autoplay";
        frame.setAttribute("scrolling", "no");
        el.appendChild(frame);
        return el;
      }

      /* 历史会话悬浮层：浮在面板左侧（不挤占主页面），列出 /api/history 会话，
       * 点击把该会话载入紧凑页主区（postMessage → iframe）。 */
      var historyEl = null;

      function removeHistory() {
        if (historyEl) { historyEl.remove(); historyEl = null; }
      }

      function makeHistory() {
        var el = document.createElement("div");
        el.className = "yukino-history";  // right 用 CSS 变量 --yukino-panel-w，与面板同宽对齐

        var head2 = document.createElement("div");
        head2.className = "yukino-history-head";
        var t = document.createElement("span");
        t.className = "yukino-history-title";
        t.textContent = "对话历史";
        var close = document.createElement("button");
        close.type = "button";
        close.className = "yukino-panel-btn yukino-panel-close";
        close.textContent = "✕";
        close.addEventListener("click", removeHistory);
        head2.appendChild(t);
        head2.appendChild(close);
        el.appendChild(head2);

        var list = document.createElement("div");
        list.className = "yukino-history-list";
        list.textContent = "加载中…";
        el.appendChild(list);

        fetch("http://127.0.0.1:8000/api/history")
          .then(function (r) { return r.json(); })
          .then(function (data) {
            var convs = (data && data.enabled) ? (data.conversations || []) : [];
            list.innerHTML = "";
            if (!convs.length) {
              list.innerHTML = '<div style="color:#b3a8b8;font-size:12px;padding:10px">还没有对话记录。</div>';
              return;
            }
            convs.forEach(function (c) {
              var item = document.createElement("div");
              item.className = "yukino-history-item";
              var t2 = document.createElement("div");
              t2.className = "yukino-history-item-title";
              t2.textContent = c.title || c.session_id;
              var m2 = document.createElement("div");
              m2.className = "yukino-history-item-meta";
              m2.textContent = (c.updated_at || "") + " · " + (c.message_count || 0) + " 条";
              item.appendChild(t2);
              item.appendChild(m2);
              item.addEventListener("click", function () {
                postToFrame({ type: "yukino.load-history", sessionId: c.session_id });
                removeHistory();
              });
              list.appendChild(item);
            });
          })
          .catch(function () {
            list.innerHTML = '<div style="color:#b38996;font-size:12px;padding:10px">加载失败（orchestrator 未运行？）</div>';
          });
        return el;
      }

      function toggleHistory() {
        if (historyEl) { removeHistory(); return; }
        historyEl = makeHistory();
        document.body.appendChild(historyEl);
      }

      function mount() {
        if (panel || isYukinoRoute()) return;
        panel = makePanel();
        document.body.appendChild(panel);
        document.body.classList.add("yukino-panel-open");  // #root 减宽，主内容左移让位
        syncToggle();
      }

      /* persist=true 记入 localStorage（用户主动隐藏）；false 仅临时收起（如跳出） */
      function unmount(persist) {
        if (persist) { try { localStorage.setItem(KEY, "1"); } catch (e) {} }
        removeHistory();
        if (panel) { panel.remove(); panel = null; }  // 卸载 iframe → ws 关闭 → 释放 s2s 槽位
        document.body.classList.remove("yukino-panel-open");
        syncToggle();
      }

      function syncToggle() {
        if (!toggleBtn) return;
        var onFull = isYukinoRoute();
        toggleBtn.style.display = onFull ? "none" : "flex";
        toggleBtn.classList.toggle("active", !!panel);  // CSS 控制 active 时按钮右移
      }

      function onRouteChange() {
        if (isYukinoRoute()) {
          unmount(false);               // 全屏路由接管，收起面板
        } else {
          var hidden = false;
          try { hidden = localStorage.getItem(KEY) === "1"; } catch (e) {}
          if (!hidden) mount();         // 返回 DSH 时恢复（除非用户主动隐藏过）
        }
        syncToggle();
      }

      toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.className = "yukino-panel-toggle";
      toggleBtn.title = "雪乃（基础模式）";
      toggleBtn.textContent = "❄";
      toggleBtn.addEventListener("click", function () {
        if (panel) unmount(true);
        else mount();
      });
      document.body.appendChild(toggleBtn);

      window.addEventListener("hashchange", onRouteChange);
      window.addEventListener("popstate", onRouteChange);

      var hidden = false;
      try { hidden = localStorage.getItem(KEY) === "1"; } catch (e) {}
      if (!hidden && !isYukinoRoute()) mount();
      syncToggle();
    }

    function apply(ctx) {
      syncBodyClass();
      window.addEventListener("hashchange", syncBodyClass);
      window.addEventListener("popstate", syncBodyClass);

      var style = document.createElement("style");
      style.dataset.plugin = "yukino-dsh";
      style.textContent = [
        ".yukino-route{position:fixed;inset:0;z-index:2147483000;background:#f6eef2}",
        "body.yukino-route-active .aionui-explorer-handle, body.yukino-route-active [class*='explorer-handle']{display:none !important}",
        "body.yukino-route-active{overflow:hidden}",
        ".yukino-route *{box-sizing:border-box}",
        ".yukino-live2d-frame{display:block;width:100%;height:100%;border:0;overflow:hidden;background:#f6eef2}",
        ".yukino-route-back{position:absolute;left:14px;top:56px;z-index:2147483001;padding:8px 12px;border:1px solid #e3d7de;border-radius:10px;background:#fff8fb;color:#c96f9b;font:700 12px/1.4 ui-rounded,'SF Pro Rounded','PingFang SC',system-ui,sans-serif;cursor:pointer}",
        // 基础模式：右侧紧凑面板 + 浮动开关按钮。
        // 面板参与 DSH 布局：展开时给 #root 减宽把主内容往左挤（同一层，非覆盖）。
        ":root{--yukino-panel-w:max(300px,min(25vw,420px))}",
        "body.yukino-panel-open #root{width:calc(100% - var(--yukino-panel-w))}",
        ".yukino-panel-toggle{position:fixed;right:14px;bottom:20px;z-index:2147482990;width:44px;height:44px;border-radius:50%;border:1px solid #e3d7de;background:#fff8fb;color:#c96f9b;font-size:20px;line-height:1;cursor:pointer;box-shadow:0 6px 20px rgba(180,140,165,.35);display:flex;align-items:center;justify-content:center;transition:right .25s ease}",
        ".yukino-panel-toggle.active{right:calc(var(--yukino-panel-w) + 14px)}",
        ".yukino-panel-toggle:hover{transform:translateY(-1px)}",
        ".yukino-panel{position:fixed;top:0;right:0;bottom:0;width:var(--yukino-panel-w);z-index:2147482995;display:flex;flex-direction:column;background:#fdf7fa;border-left:1px solid #eadfe6;box-shadow:-10px 0 30px rgba(180,140,165,.22)}",
        ".yukino-panel-head{display:flex;align-items:center;gap:8px;padding:8px 10px;background:#fff8fb;border-bottom:1px solid #eadfe6}",
        ".yukino-panel-name{flex:1;font:600 13px/1 ui-rounded,'SF Pro Rounded','PingFang SC',system-ui,sans-serif;color:#8a5a72}",
        ".yukino-panel-btn{width:28px;height:28px;border:1px solid #e3d7de;border-radius:8px;background:#fff;color:#c96f9b;font-size:14px;line-height:1;cursor:pointer}",
        ".yukino-panel-close{color:#b38996}",
        ".yukino-panel-frame{flex:1;width:100%;border:0;background:#f6eef2}",
        // 历史会话悬浮层：贴面板左侧，覆盖在 DSH 主页面之上（不挤占）
        ".yukino-history{position:fixed;top:0;bottom:0;right:var(--yukino-panel-w);width:300px;z-index:2147482996;display:flex;flex-direction:column;background:#fffafd;border-right:1px solid #eadfe6;box-shadow:8px 0 24px rgba(180,140,165,.18)}",
        ".yukino-history-head{display:flex;align-items:center;gap:8px;padding:8px 10px;background:#fff8fb;border-bottom:1px solid #eadfe6}",
        ".yukino-history-title{flex:1;font:600 13px/1 ui-rounded,'SF Pro Rounded','PingFang SC',system-ui,sans-serif;color:#8a5a72}",
        ".yukino-history-list{flex:1;overflow-y:auto;padding:6px}",
        ".yukino-history-item{padding:8px 10px;border-radius:8px;cursor:pointer}",
        ".yukino-history-item:hover{background:#f6edf2}",
        ".yukino-history-item-title{font-size:13px;color:#5c4450;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
        ".yukino-history-item-meta{font-size:11px;color:#b3a8b8;margin-top:2px}",
      ].join("\n");
      document.head.appendChild(style);

      initCompactPanel();

      // 任务完成监听：sessions.list 主动推送（事件驱动），不轮询、不进 React。
      ctx.effect(function () {
        return createTaskDoneWatcher(ctx);
      }, "yukino-dsh: task-done watch");

      ctx.slots.inject("shell.overlay", function () {
        return ctx.slots.register({
          name: "shell.overlay",
          id: "yukino-dsh",
          order: 80,
          label: "Yukino",
        }, YukinoApp);
      });
    }

    exports.apply = apply;
    exports.inject = ["slots", "sessions", "connection"];
    return module.exports;
  },
});
