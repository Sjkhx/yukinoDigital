window.__ModuleLoader__.load({
  id: "yukino-dsh",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;

    var React = require("react");
    var g = require("react/jsx-runtime");

    var useEffect = React.useEffect;
    var useRef = React.useRef;
    var useState = React.useState;

    function nodeText(n) {
      if (!n) return "";
      if (typeof n === "string") return n;
      if (typeof n.text === "string" && n.text) return n.text;
      if (typeof n.content === "string" && n.content) return n.content;
      if (typeof n.message === "string" && n.message) return n.message;
      if (typeof n.summary === "string" && n.summary) return n.summary;
      if (n.call && typeof n.call.name === "string") return n.call.name;
      if (n.name && typeof n.name === "string") return n.name;
      return "";
    }

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

    /* 只取任务最终总结：优先最后一条 assistant/user 文本，跳过 tool-result 等中间产物 */
    function finalSummary(snapshot) {
      var nodes = Array.isArray(snapshot && snapshot.nodes) ? snapshot.nodes : [];
      for (var i = nodes.length - 1; i >= 0; i--) {
        var n = nodes[i];
        if (!n) continue;
        var kind = n.kind || "";
        if (kind === "tool-result" || kind === "tool" || kind === "command") continue;
        var t = nodeText(n);
        if (t && (kind === "assistant" || kind === "user" || kind === "steering")) return t;
      }
      for (var i2 = nodes.length - 1; i2 >= 0; i2--) {
        var t2 = nodeText(nodes[i2]);
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

    /* 所有 DSH 会话：running true->false 时进入待发送队列。
     * 同一任务经常会有多个会话先后结束（主会话 + 子代理），这里做 4 秒
     * 防抖合并：只保留同一 project+title 的最后一个，并在 flush 时重新读取
     * 会话节点，确保拿到的是最终总结，而不是中间的理解/工具节点。 */
    function useTaskDoneWatcher(sessions, resolveSession) {
      var notified = useRef(new Map());
      var pending = useRef(new Map());
      var timer = useRef(null);

      // 只关心 running 状态集合的变化：DSH 会话每次流式更新都会触发
      // useSessions 重渲染，如果每次都全量扫描会话节点会拖慢 DSH。
      // 这里把「哪些会话在跑、哪些停了」压缩成一个签名，只有签名变化才处理。
      var runningSig = useMemo(function () {
        var byId = (sessions && sessions.byId) || {};
        var running = [];
        var done = [];
        Object.keys(byId).forEach(function (id) {
          var s = byId[id];
          if (!s) return;
          if (s.running) running.push(id);
          else done.push(id);
        });
        running.sort();
        done.sort();
        return running.join(",") + "|" + done.join(",");
      }, [sessions]);

      function flush() {
        if (timer.current) clearTimeout(timer.current);
        timer.current = null;
        var items = Array.from(pending.current.values());
        pending.current.clear();
        if (!items.length) return;
        items.sort(function (a, b) { return (b.at || 0) - (a.at || 0); });
        var item = items[0];
        var binding = resolveSession ? resolveSession(item.sessionId) : null;
        var snapshot = binding && binding.getSnapshot ? binding.getSnapshot() : null;
        var summary = finalSummary(snapshot);
        postTaskDone({ project: item.project, title: item.title, summary: summary || "" });
      }

      useEffect(function () {
        var byId = (sessions && sessions.byId) || {};
        Object.keys(byId).forEach(function (id) {
          var s = byId[id];
          if (!s) return;
          var wasRunning = notified.current.get(id);
          if (wasRunning && !s.running) {
            var project = projectName(s);
            var title = sessionTitle(s);
            var key = project + "::" + title;
            pending.current.set(key, { sessionId: id, project: project, title: title, at: Date.now() });
            if (timer.current) clearTimeout(timer.current);
            timer.current = setTimeout(flush, 4000);
            notified.current.set(id, false);
          } else if (s.running) {
            notified.current.set(id, true);
          } else if (!wasRunning) {
            notified.current.set(id, false);
          }
        });
      }, [runningSig, resolveSession]);
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

    function YukinoApp(props) {
      var useSessions = props.useSessions;
      var resolveSession = props.resolveSession;
      var store = useSessions(function (x) { return x; });
      var sessions = store || { ids: [], byId: {} };
      var routeActive = useRoute();
      var forceState = useState(function () {
        return window.location.pathname === "/yukino" || window.location.hash === "#/yukino";
      });

      useTaskDoneWatcher(sessions, resolveSession);

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

      var resolveSession = function (id) {
        try {
          var b = ctx.sessions.binding(id);
          return b ? b.session : null;
        } catch (e) {
          return null;
        }
      };

      ctx.slots.inject("shell.overlay", function () {
        return ctx.slots.register({
          name: "shell.overlay",
          id: "yukino-dsh",
          order: 80,
          label: "Yukino",
          inject: function () {
            return { resolveSession: resolveSession };
          },
        }, YukinoApp);
      });
    }

    exports.apply = apply;
    exports.inject = ["slots", "sessions"];
    return module.exports;
  },
});
