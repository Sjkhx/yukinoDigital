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

    /* 文本块提取：assistant 的 blocks / user·steering·tool-result 的 content 里挑正文，
     * 跳过 reasoning（思考过程）与无 text 的 tool-call/image。 */
    function blockTexts(list) {
      var out = "";
      if (!Array.isArray(list)) return out;
      list.forEach(function (b) {
        if (!b) return;
        if (b.kind === "reasoning" || b.type === "reasoning") return;
        if (typeof b.text === "string" && b.text) out += b.text;
      });
      return out;
    }

    /* 节点正文：只取用户可见的文本 */
    function nodeText(n) {
      if (!n) return "";
      if (n.kind === "assistant") return blockTexts(n.blocks);
      if (n.kind === "user" || n.kind === "steering") return blockTexts(n.content);
      if (n.kind === "tool-result") {
        if (n.call && typeof n.call.name === "string") return "「" + n.call.name + "」";
        return "";
      }
      if (typeof n.text === "string" && n.text) return n.text;
      return "";
    }

    /* 只取任务最终总结：最后一条 assistant/user 文本，跳过 tool-result 等中间产物 */
    function finalSummary(snapshot) {
      var nodes = Array.isArray(snapshot && snapshot.nodes) ? snapshot.nodes : [];
      var skip = {
        "tool-result": 1, tool: 1, command: 1, context: 1,
        "model-retry": 1, "turn-error": 1, "turn-max-tokens": 1, compaction: 1,
      };
      for (var i = nodes.length - 1; i >= 0; i--) {
        var n = nodes[i];
        if (!n || skip[n.kind]) continue;
        var t = nodeText(n);
        if (t && (n.kind === "assistant" || n.kind === "user" || n.kind === "steering")) return t;
      }
      for (var j = nodes.length - 1; j >= 0; j--) {
        var t2 = nodeText(nodes[j]);
        if (t2) return t2;
      }
      return "";
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

    /* 任务完成监听：订阅 sessions.list，由 DSH 在列表变化时主动通知（事件驱动，非轮询）。
     * 完全在 apply 侧执行，不进 React 渲染路径，不影响 DSH 消息流。
     * 检测 running true→false 的会话，4 秒防抖合并（同一 project+title 只留最后一个），
     * flush 时再读会话快照，确保拿到的是最终总结。
     *
     * 性能约束：DSH 在流式输出时每个 host/mux frame 都会触发一次 list 订阅回调，
     * 而 ctx.sessions.list 是 cordis 服务代理，每次属性访问都要走 Proxy get（事件链 +
     * fiber 查找），成本很高。旧实现每次回调都通过 ctx.sessions.list 现取对象，高频
     * 下把主线程占满导致 DSH 卡死。这里把 list 引用缓存一次，回调只置标志、用
     * requestAnimationFrame 合并到每帧最多一次扫描。 */
    function createTaskDoneWatcher(ctx) {
      /* 一次性穿透 Proxy 取 list 引用，后续回调不再走 ctx.sessions 属性访问 */
      var list = ctx.sessions.list;
      var runningById = new Map();
      var pending = new Map();
      var timer = null;
      var scanQueued = false;

      function flush() {
        timer = null;
        var items = Array.from(pending.values());
        pending.clear();
        if (!items.length) return;
        items.sort(function (a, b) { return (b.at || 0) - (a.at || 0); });
        var item = items[0];
        var summary = "";
        try {
          /* 低频（4 秒防抖后）才走 proxy 解析 binding，不影响高频路径 */
          var binding = ctx.sessions.binding(item.sessionId);
          var session = binding && binding.session;
          if (session && typeof session.getSnapshot === "function") {
            summary = finalSummary(session.getSnapshot());
          }
        } catch (e) {
          console.warn("[yukino-dsh] read snapshot failed:", e);
        }
        postTaskDone({ project: item.project, title: item.title, summary: summary || "" });
      }

      function schedule() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(flush, 4000);
      }

      /* 扫描当前快照，检测 running true→false 边缘；用 for-in 避免 Object.keys 数组分配 */
      function scan() {
        scanQueued = false;
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

      /* 订阅回调极简：只排队一帧扫描。无论 DSH 推送多密集，每帧最多全扫一次 */
      function onListChange() {
        if (scanQueued) return;
        scanQueued = true;
        if (typeof requestAnimationFrame === "function") {
          requestAnimationFrame(scan);
        } else {
          setTimeout(scan, 0);
        }
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

      return list.subscribe(onListChange);
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
        ".yukino-route-back{position:absolute;left:14px;top:14px;z-index:2147483001;padding:8px 12px;border:1px solid #e3d7de;border-radius:10px;background:#fff8fb;color:#c96f9b;font:700 12px/1.4 ui-rounded,'SF Pro Rounded','PingFang SC',system-ui,sans-serif;cursor:pointer}",
      ].join("\n");
      document.head.appendChild(style);

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
    exports.inject = ["slots", "sessions"];
    return module.exports;
  },
});
