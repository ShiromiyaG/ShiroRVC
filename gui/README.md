# ShiroRVC — Qt interface

A native front-end for the same backend the Gradio app and the CLI use. It is
an optional add-on: **deleting this directory leaves the rest of the
application working**, and nothing outside it imports it.

```
python -m gui          # or start-gui.bat / start-gui.sh
```

## The isolation rule

Two directions, both enforced by `tests/test_gui_isolation.py`:

1. **Nothing outside `gui/` imports `gui`.** That is what makes the directory
   deletable. The only files elsewhere that mention it are the launchers
   (`start-gui.*`), the packaging workflow, and that test.
2. **Only `gui/services/` imports the backend** (`core`, `rvc`, `tabs`). Views
   and widgets talk to `gui.services.engine` and nothing else. When the
   backend's signatures move, one directory has to follow them instead of
   twenty files.

## How it talks to the backend

The GUI process never imports torch. It spawns `gui/services/worker.py` as a
child, in the same environment, and speaks a line-delimited JSON protocol over
its stdin/stdout:

```
GUI ──stdin──►  {"id": 7, "cmd": "infer", "args": {...}}
GUI ◄─stdout──  <RS>{"id": 7, "type": "started"}
GUI ◄─stdout──  ordinary log output, forwarded verbatim
GUI ◄─stdout──  <RS>{"id": 7, "type": "result", "data": {...}}
```

`<RS>` is ASCII 0x1e, which never appears in log output, so no escaping is
needed in either direction.

Three things fall out of this, and all three are the reason for the design:

- **The window paints in well under a second**, because torch is loading in a
  different process.
- **A CUDA OOM or a native segfault does not take the UI with it.** The engine
  reports a dead worker and offers to restart it.
- **Models stay warm.** `core.import_voice_converter` is `lru_cache`d, so
  consecutive conversions reuse a loaded checkpoint — which a fresh subprocess
  per file could not do.

## Layout

| Path | Responsibility |
| --- | --- |
| `services/paths.py` | Where everything is, derived from this file's location |
| `services/catalog.py` | Dropdown contents. Filesystem and JSON only, no torch |
| `services/engine.py` | GUI-side client: owns the `QProcess`, routes replies |
| `services/worker.py` | Backend-side: the only code that imports `core` |
| `services/tbreader.py` | Incremental TFRecord/protobuf reader for the live chart |
| `services/prefs.py` | Persisted state, in `gui/.state/` so it stays portable |
| `widgets/` | Presentation only; never imports `services` |
| `widgets/chart.py` | Hand-painted line chart: log axis, legend, decimation |
| `widgets/metrics.py` | The training monitor: chart, presets, metric picker |
| `widgets/icons.py` | Stroked line icons on a 24×24 grid, plus the window icon |
| `views/` | One screen each; talks to the backend through `engine` |
| `theme.py` + `resources/style.qss` | `@token@` substitution, dark and light |

The version comes from the repository's `VERSION` file, read directly in
`gui/__init__.py` — the same file `app.py` and the release workflow read, so
the two interfaces cannot report different builds.

## Notes

- The live chart parses event files directly rather than importing
  `tensorboard`. Full parse of a 53 MB run: ~1.1 s. Incremental poll: ~1 ms.
- `widgets/chart.py` is hand-painted rather than pyqtgraph — it keeps the
  install smaller and follows the theme without a second styling system.
- Custom-painted widgets do not see QSS. They get the palette through
  `apply_theme(tokens)`, called from `MainWindow._apply_theme`.
- A run writes ~180 scalar tags. The picker groups them by prefix and hides the
  per-dimension diagnostics; presets in `widgets/metrics.py` cover the usual
  questions (health, adversarial balance, gradients, latent, schedule).
- Log scale is worth reaching for on any loss: reconstruction falls from ~60 to
  ~14 within the first 2% of a run, so a linear axis spends its whole range on
  the opening cliff.
- Pages cap their form column at `Page.max_width`. Without it a 0–1 slider gets
  1400 px of travel on a wide monitor.
- **Chart performance.** The grid and series render into a pixmap that is
  blitted on subsequent paints; only the legend and crosshair are live. Before
  that split, every mouse move re-ran decimation, smoothing and ~24k segments:
  146 ms per paint on a 20k-point six-series run — a 7 fps chart under the
  cursor. Now 2.5 ms hovering, 14 ms when new data arrives. Decimation is
  budgeted to one point per pixel of plot width, and series are drawn with
  `drawPolyline` on a prepared `QPolygonF` rather than a `QPainterPath` built
  a `lineTo` at a time.
- `Collapsible.set_expanded` freezes updates on the top-level window while it
  changes visibility and settles the layout, so the section repaints once
  instead of flashing through intermediate states.
- Pages get `on_hidden()` alongside `on_shown()`. The training page stops its
  metrics timer there unless a run is actually going.
- **The event reader is kept, and polled off the UI thread.** `RunReader`
  tails a file from where it left off — a poll is about a millisecond — but
  only if it survives between calls. `_attach_reader` runs on every show of
  the Training page *and* on every change of the model name, so rebuilding it
  each time meant parsing the whole run twice per tab switch: measured at
  0.73–0.88 s on a 36 MB pretrain, on the UI thread, which is what made
  switching to that tab look like a hang. Same run → same reader. A read for a
  genuinely new run runs in a `QThreadPool` job and reports back by signal,
  carrying its reader so a result for a run the user has since switched away
  from can be dropped.
- `widgets/progress.py` reads the trainer's `[PROGRESS]` lines. Rich renders
  *nothing* when stdout is not a terminal, so there is no bar to scrape —
  `rvc/train/train.py:_emit_machine_progress` prints a fixed format instead,
  and only when there is no terminal to clutter.
- **Depth is theme-dependent, on purpose.** Card shadows are attached only in
  the light theme. Measured: 25 shadowed cards take a full-window repaint from
  7.8 ms to 24.5 ms — a graphics effect forces its widget through an offscreen
  buffer on every paint — and on a near-black background a black shadow is
  invisible. Dark mode gets its elevation from the card gradient sitting
  lighter than the window, which costs nothing.
- **Compositor backdrop** (`native.py`): on Windows 11 the DWM will blur the
  desktop behind the window. Off by default and cycled from the sidebar. The
  window paints its own background in `MainWindow.paintEvent`, and turning the
  backdrop on just makes that fill transparent — no stylesheet swap (0.3 s of
  re-polish) and no `WA_TranslucentBackground`, which is creation-time and
  would mean rebuilding the window. The frame follows: the sidebar and status
  bar are `effects.ChromePanel`s, which paint their own fill and drop it to
  `BACKDROP_ALPHA` when a backdrop is on, for the same reason — an alpha that
  changes cannot come from the stylesheet. Cards stay fully opaque, so nothing
  anyone has to read sits directly on a wallpaper. Turning it *off* paints
  before it tells the DWM, not after: drop the extended frame while the window
  is still transparent and the compositor has a frame with nothing to draw,
  which it fills with black.
- **The backdrop is Windows-only, and that is the whole story off Windows.**
  Mica and Acrylic are things the DWM draws; X11 and Wayland have no portable
  equivalent to ask for, so `supports_backdrop()` is false, the sidebar button
  is hidden and the window is simply opaque. `native.py` must stay importable
  there regardless — `gui.app` imports it unconditionally — which is why it
  uses `ctypes.c_void_p` rather than `ctypes.wintypes.HWND`: importing
  `ctypes.wintypes` on Linux raises.
- **The themed sheet goes on the window, not on the `QApplication`.** Same
  widgets, same pixels, 0.3 s instead of 2.1 s — an application-wide sheet
  makes Qt re-resolve every rule against every widget in the process. Only
  what Qt parents to the desktop rather than to us needs to be application-wide,
  which is tooltips; `resources/global.qss` holds that, carries no literal
  colour, and is set once at startup.
- **A plain `QWidget` does not paint the background its QSS rule gives it**
  unless it has `WA_StyledBackground`; `ConsoleBar` and `TrainingProgress` set
  it. Miss it and nothing looks wrong for as long as an opaque window sits
  behind — until the backdrop goes on and that widget turns see-through.
- The native title bar follows the theme via `DWMWA_USE_IMMERSIVE_DARK_MODE`.
  Without it the frame stays light over a dark window, which is the clearest
  sign an application is not really themed.
- `widgets/navlist.py` paints the sidebar selection itself so it can *slide*
  between rows. QSS has no transitions, so a stylesheet-driven highlight
  teleports.
- **A `QHBoxLayout` that cannot fit its children overlaps them.** It does not
  clip, scroll or elide — at 1130 px the metrics panel's window/smoothing row
  was painted across the chart's legend. Rows that carry more than a couple of
  controls use `widgets/flow.py`'s `FlowRow`, which wraps onto a second line,
  and the chart and tag tree carry minimum heights small enough for the panel
  to fit a laptop window. `test_gui_behavior.py` asserts both: the panel's
  minimum height, and that no two of its rows share pixels at 330x560.
- **A run outlives the page that started it.** Training is a process in the
  backend, so switching to Inference mid-run changes nothing about it — but
  the card reporting it lives on the Training page. `widgets/dock.py` floats
  that same widget (reparented, not duplicated, so no state is copied) over
  the bottom-right of whatever page is showing, and hands it back when the
  user returns or the run ends. It carries its own Stop, because it is the
  only training control on screen at that point.
- **The card grows in when a run starts and collapses away when it ends** —
  its *height* is animated, not its opacity, because a fade needs a graphics
  effect and that slot is taken by the card shadow the light theme adds. The
  finished card is held for `TrainingPage.FINISHED_HOLD_MS` first: the final
  epoch count and elapsed time are shown nowhere else. The dock repositions on
  its panel's resize, so a card collapsing while it floats takes the dock (and
  its shadow) down with it.
- **The card travels between its two homes rather than teleporting.** A widget
  cannot be animated across a reparent — it is in one layout or the other, and
  the change is instant — so `dock.fly` plays the move with a scaled snapshot
  while the real widget is hidden at the destination. It hangs off
  `stateChanged`, not `finished`: a flight cut short by a second tab switch
  never emits `finished`, and the widget it stood in for would stay hidden.
- **Results are not cleared by navigation.** An `AudioPlayer` keeps its file
  until the next conversion replaces it or the ✕ beside the volume is pressed.
  Comparing two takes means leaving the page and coming back.
- **The window always opens on Inference**, never on the last view used.
- **Icons.** The window and taskbar icon comes from `assets/logo.png`, trimmed
  of its ~11% transparent margin and rendered at each size the shell asks for —
  a bare `QIcon(path)` throws away a quarter of the pixels at 16 px. The
  sidebar mark is a *drawn* shiba (`icons.py`), because the logo is a detailed
  illustration that turns to mush at 20 px and would not match the stroked nav
  icons beside it. On Windows, `SetCurrentProcessExplicitAppUserModelID` is
  needed as well, or the taskbar attributes the process to the interpreter and
  shows the Python icon regardless.
- `widgets/scrollguard.py` takes the wheel away from combo boxes, spin boxes
  and sliders and gives it to the scroll area behind them. Scrolling past a
  control must never edit it — on a form this long the damage is invisible
  until the output comes out wrong.
- Defaults come from `catalog.INFERENCE_DEFAULTS`, which mirrors the Gradio
  tabs *per context*: single-file, batch and TTS genuinely ship different
  numbers upstream. Output filenames follow `output_path_fn` as well, so both
  interfaces write to the same place and the Gradio "clear `_output` files"
  button still finds them.
- The theme is written the moment it is switched, not at shutdown. Dark is the
  first-run default.
- Rarely-touched controls live in a `Collapsible`, mirroring the Gradio tabs'
  `gr.Accordion`. Sections start closed — one that opens itself is not hiding
  anything. Conditional controls (`torch_compile_mode`, the gradient clipping
  schedule) are revealed by their own toggle, as upstream does.
- `tests/test_gui_coverage.py` parses `core.py` with `ast` and asserts the
  forms set **every** parameter the backend takes. A control that is merely
  missing produces no error at runtime — the argument just keeps its default —
  so it has to be caught structurally.
- The metrics panel picks its run independently of the model-name field. That
  field steers it until the user chooses a run by hand; after that the choice
  sticks, so configuring the next run cannot yank the chart off the one being
  watched. Starting a training run overrides both.
