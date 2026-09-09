# Crowd Master — handoff

Everything a new session needs. Written 2026-09-09.

**Repo:** `github.com/YoussefKabbary/Crowd-Intelligence-System` (not yet pushed —
see [Publishing](#publishing))
**Local:** `C:\Users\DELL\Desktop\crowd-master-v2`, branch `main`, 12 commits
**Python:** `./venv/Scripts/python.exe` — **not** the system python, which lacks
torchvision and will fail at import

---

## 1. What this is

Real-time crowd analytics for CCTV: person detection and tracking, gate
entry/exit counting, per-zone occupancy, anomaly alerts with saved clips,
forecasting, REST API, session reports. YOLOv8 + ByteTrack + PyTorch + OpenCV.
Targets prayer halls, and generalises to malls, stations, venues.

Single file (`crowd_master_v2.py`, ~4,400 lines) plus `identity.py`, `tests/`,
`tools/`.

---

## 2. State of play

| Phase | Status |
|---|---|
| 1 — critical audit fixes | done, committed |
| 2 — zones, accuracy harness, event clips | done, committed |
| 3 — identity matching | on branch `phase3-identity`, **not in the v2 release** |
| Publishing | prepared, **not pushed** |

Runs clean: 999 frames, 42.6 ms/frame, no exceptions.

---

## 3. Measurements — the important part

Every number below was measured on this machine (laptop RTX 3050, CUDA 13.0,
torch 2.11.0+cu130, ultralytics 8.4.99). **Do not re-derive these; do not design
against wishes that contradict them.**

### Performance, before and after the audit

| | v5.2 | now |
|---|---|---|
| steady-state inference | 95–232 ms | **43–72 ms** |
| detection throughput | 4–10 /s | **~20 /s** |
| first useful frame | 10.4 s | **0.6 s** |
| startup to first detection | 26 s | **11 s** |

Timings vary a lot with machine load — the same binary measured 43 ms idle and
over 130 ms with busy background processes. Treat any single figure as
indicative.

### What the pixels support (identity work)

Measured on the project's 640×360 CCTV clip:

| signal | measured | needed | verdict |
|---|---|---|---|
| inter-ocular (eye to eye) | **3.2 px** median, 6.6 p90 | ~90 px | face recognition impossible — 28× short |
| person height | **94 px** median, 28% ≥128 px | ≥96 px for ReID | ReID viable |
| full gait cycle available | **44%** of tracks | — | gait marginal |

Face recognition would need a dedicated camera: **≤1.2 m at 1080p**, or
**≤5 m at 4K with a narrow lens**. That is a deployment decision, not a code
change.

Gait's documented worst case is loose clothing — directly relevant to a prayer
hall. Treat it as corroboration, never identification.

### Tracker identity stability

ID churn = distinct IDs issued ÷ average people on screen. 1.0 is perfect.

| tracker | ms/frame | churn | mean unbroken run |
|---|---|---|---|
| bytetrack | 19.2 | **8.47** | 37.3 frames |
| botsort | 34.8 | 8.11 | 38.2 |
| botsort + ReID (detector features) | 21.2 | 10.47 | 28.8 |
| botsort + ReID (`yolo26s-reid.onnx`) | 232 (CPU) | 10.39 | 28.7 |

**Both ReID variants measured worse.** Identity stability is the real
bottleneck for gate counting and for "follow this person". Verify any claimed
improvement by measurement — this is where intuition has been wrong twice.

### Accuracy

**Unmeasured on real footage.** Every number above is speed. `tests/measure_accuracy.py`
exists but needs hand-counted frames. On synthetic footage the `accurate` signal
scored MAE 0.07; that is not evidence about real crowds.

**This is the single biggest gap in the project.**

---

## 4. Known limits, stated plainly

**Gate counting depends on tracker identity.** Dependable when people cross one
at a time or clearly separated. Not dependable for dense two-way flow: with
three walkers crossing together, one ID was credited with two crossings while a
third walker went uncounted, and a single ID's x jumped 375 px between
consecutive frames. BoT-SORT behaved the same. Closing this needs
re-identification that actually works, not threshold tuning.

Do **not** "fix" it by lowering `MIN_CROSS_FRAMES` or raising `MAX_JUMP_PX` —
those guards are what suppress phantom counts.

`tests/test_gate_counting.py` reflects this: sequential crossings are asserted,
the simultaneous case is measured and printed but not asserted, because
asserting on it makes the suite fail for reasons unrelated to the gate
(toggling cuDNN autotuning was enough to flip it).

---

## 5. What was fixed in the audit

Full detail in `AUDIT.md`. The ones that mattered:

- **Sliced inference destroyed all tracking** — returned no track IDs, which
  silently killed gate counting, box smoothing and the centre tracker.
- **NMS deleted real detections twice over** — xyxy boxes passed to a function
  that reads xywh, plus `score_threshold=scores.min()` dropping the weakest
  detection every frame. 20 scattered people → 15 kept instead of 19.
- **TTA emitted inverted boxes** (`x1 > x2`) and never ran in the GPU config.
- **Online learning cost 40–60% of frame rate and contributed 0** in every row
  of every session log. Off by default now.
- **Panic detection was fed a constant grey image** — optical flow on it is
  identically zero, so the alerts were mathematically unreachable.
- **Forecaster returned the current count** for all four horizons. Rewritten.
- **API bound 0.0.0.0 with no auth.** Now loopback + `X-API-Key`.

Three more were found *by writing the gate test*, not by reading code:
adaptive resolution collapsing to 320×180 exactly when slicing engaged; slicing
sharing the tracked model's predictor and disturbing tracker state; reassigned
tracker IDs registering as crossings.

---

## 6. Environment gotchas

- **HTTPS-inspecting antivirus (Avast) breaks every Python download.**
  `certifi` lacks its root cert while the Windows store has it. Fixed with
  `truststore` (installed). Without it, model downloads fail with
  `CERTIFICATE_VERIFY_FAILED`.
- **`onnxruntime` is CPU-only** as installed. ReID measured 232 ms/frame.
  For real use: `pip uninstall onnxruntime && pip install onnxruntime-gpu`.
- **VS Code must use the venv interpreter.** `Ctrl+Shift+P` → *Python: Select
  Interpreter* → `.\venv\Scripts\python.exe`. The script now fails with an
  explanatory message rather than a bare `ModuleNotFoundError`.
- **Machine is memory-constrained** (~0.5–1 GB free of 7.7 GB). The gate test
  loads models once for the whole suite because per-case loading OOM'd.

---

## 7. How to run

```bash
cd C:\Users\DELL\Desktop\crowd-master-v2
venv\Scripts\python.exe crowd_master_v2.py
```

Headless / batch:
```bash
CROWD_MASTER_HEADLESS=1 CROWD_MASTER_MAX_FRAMES=600 \
CROWD_MASTER_VIDEO=clip.mp4 CROWD_MASTER_DATA_DIR=out \
venv/Scripts/python.exe crowd_master_v2.py
```

Tests:
```bash
venv\Scripts\python.exe tests\test_gate_counting.py     # ~5 min, needs GPU free
venv\Scripts\python.exe tests\measure_accuracy.py clip.mp4 labels.csv
```

All configuration is environment variables — see README.

---

## 8. Next steps, in order

### Now — highest value per hour

1. **Hand-count 200 frames** (`tools/extract_frames.py`), then run
   `tests/measure_accuracy.py`. About an hour of work. Until this exists, no
   change can be judged on accuracy, only on speed. Everything below is
   guesswork without it.

2. **Push to GitHub.** Prepared but not pushed — see [Publishing](#publishing).

### Phase 3 — identity matching

`identity.py` is written, compiles, and its capability gate is verified. It
lives on the `phase3-identity` branch and is deliberately not part of the v2
release: it is not yet verified against a measured false-accept rate, and it
should not ship until it is.

```bash
git checkout phase3-identity
```

3. Wire `IdentityEngine` into `DetectionWorker`: measure capability per tracked
   person, call `check()`, surface candidates in the HUD and `/api/status`.
4. Operator review flow: list pending candidates, confirm/reject, show the
   reference image beside the live crop.
5. Notify on a **confirmed** match only, never on a candidate. Pluggable
   channel, start with a local webhook.
6. `tests/test_identity.py`: prove the gate refuses face on low-res input, that
   an enrolled person is matched by ReID on a clip, and that nothing is emitted
   below threshold.
7. **Measure the false-accept rate** with a watchlist of people *not* in the
   video. Report it honestly. If it is high, say so rather than lowering the
   bar.

Non-negotiable in this phase: never assert an identity, never act
autonomously, always audit-log, return nothing when the pixels do not support a
method. A false positive here flags a person as wanted — a different class of
consequence from miscounting a crowd. Thresholds in `identity.py` are
placeholders and must be calibrated per camera before anyone acts on output.

Face matching needs `pip install insightface`, deliberately not preinstalled.

### Later — worth doing, not urgent

- **Drop the pose model for heads.** Running `yolov8s-pose` purely for head
  centres costs 17–34 ms/frame, roughly half the budget. Derive the head from
  the upper body box, or train a small head detector.
- **Delete the CNN regressor, online learner and auto-retrainer** (~400 lines,
  disabled, contributed 0) unless you intend to revisit them.
- **Per-zone thresholds** instead of one global one. A doorway is crowded at
  10; a prayer hall is not crowded until 200.
- **Split the file into a package.** Purely structural; no functional gain.
- **Multi-camera and WebSocket.** Both assume a single camera works reliably
  first — which depends on step 1.

---

## 9. Publishing

The local repo was `git init`'d fresh on 2026-09-04 and shares **no history**
with the GitHub repo. `git push --force` would destroy the existing repo.

Safe route:

```bash
git remote add origin https://github.com/YoussefKabbary/Crowd-Intelligence-System.git
git fetch origin
git branch -m main audit-and-fixes-v5.3
git push -u origin audit-and-fixes-v5.3
```

Then open a PR and review before merging.

**Before publishing, note:** the existing public README and two LinkedIn posts
carry claims this project's own measurements do not support — "98% accurate
people counting", "34 FPS on RTX 3060", "< 5ms with FP16", plus forecasting and
online learning as working features. The measured figures are above. The new
README replaces those claims; the LinkedIn posts are separate and still stand.

`docs/screenshot.jpg` shows identifiable people from the test footage. Replace
it if that video is private or a client's.

---

## 10. File map

| Path | What |
|---|---|
| `crowd_master_v2.py` | the application |
| `identity.py` | watchlist matching, capability-gated (phase 3, not wired) |
| `trackers/botsort_reid.yaml` | BoT-SORT with ReID; measured worse, kept for reference |
| `tests/test_gate_counting.py` | gate regression test |
| `tests/make_gate_clip.py` | synthetic clip generator, sequential + simultaneous |
| `tests/measure_accuracy.py` | accuracy vs hand-counted frames |
| `tools/extract_frames.py` | pulls frames out for hand-counting |
| `AUDIT.md` | the code review, with measurements |
| `README.md` | user-facing docs |
