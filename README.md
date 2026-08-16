# yukino-dsh

Yukino (Yukinoshita Yukino) standalone-route plugin for DSH.

- Route: `#/yukino` (append to DSH web URL). It does NOT float over the DSH main UI.
- Left tree: workspace → sessions → Yukino conversation nodes (read-only DSH session store).
- Right pane: iframe to the existing Yukino Live2D page `http://127.0.0.1:8000/`.
- When a DSH session transitions `running → done`, the browser speaks a short completion summary via Web Speech API and records it in the route.
- No context is injected into DSH. Sessions are only read.

## Install

Profile: `C:\Users\78723\.dsh\profiles\web`
- `package.json` already points to `file:///D:/Program Files/dsh/yukino-dsh-plugin`.
- A copy/symlink exists at `node_modules/yukino-dsh`.
- If you reinstall, run `pnpm install` in the profile directory (network permitting).
