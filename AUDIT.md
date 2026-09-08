# Crowd Master — audit and fixes (2026-09-04)

Full review of v5.2, everything reproduced by running the code on the target
machine (RTX 3050 Laptop, CUDA 13.0, torch 2.11.0+cu130, ultralytics 8.4.99).
The result is v5.3.

Full report: <https://claude.ai/code/artifact/ecab8e47-47ed-4332-8281-c561e5a08321>

---

## Headline results

| | v5.2 | v5.3 |
|---|---|---|
| Steady-state inference | 95–232 ms | **43–72 ms** |
| Detection throughput | 4–10 /s | **~19 /s** |
| First useful frame | 10.4 s | **0.6 s** |
| Track IDs with slicing on | `None` — gates dead | **present** |
| NMS recall (20 scattered people) | 15 kept | **19 kept** |
| Anomaly score | always `0.000` | **live** |
| Forecast horizons | all identical | **distinct** |
| CSV cadence | gaps of 2–21 s | **even 2 s** |
| Gate counting, sequential | 0 counted | **3 of 3, both modes** |
| Gate counting, simultaneous | 0 counted | 2-3 of 3 — see below |

---

## Critical

**CM-01 — sliced inference destroyed all tracking.** `USE_SLICED=True` (on in
the GPU variant) routed detection through `SlicedInference.detect`, which returns
no track IDs. That silently killed gate entry/exit counting, box EMA smoothing,
the centre tracker and head-to-body association. The code then re-ran
`.track()` whenever gates existed, discarding the sliced result entirely — so
slicing cost 83–131 ms per frame and bought nothing.

*Fixed:* the tracked pass is now unconditional and is always the source of IDs.
Slicing and TTA are extra recall merged on top; boxes that do not correspond to
a tracked person are carried with id `-1` and excluded from gate logic. Slicing
only engages once `body_count >= DENSE_FALLBACK_THR`, so an empty corridor does
not pay for it.

**CM-02 / CM-03 — NMS deleted real detections, two ways.**
`SlicedInference._nms` passed `xyxy` boxes to `cv2.dnn.NMSBoxes`, which reads its
input as `xywh`; every box was treated as far larger than it is, so detections
that do not overlap at all suppressed one another. It also set
`score_threshold=scores.min()`, discarding the least-confident detection in every
frame — typically the distant or partly occluded person.

*Measured:* 20 scattered people → 15 kept instead of 19.
*Fixed:* replaced with `torchvision.ops.nms`, which takes `xyxy` and runs on GPU.

**CM-04 — TTA produced inverted boxes.**

```python
fb[:, 0] = self.cfg.INFER_W - fb[:, 2]
fb[:, 2] = self.cfg.INFER_W - fb[:, 0].copy()   # reads the value just written
```

Yielded `x1 > x2`. It also used the fixed `INFER_W` rather than the adaptive
width. And TTA never ran in the GPU config at all, because the guard was
`if self.use_tta and not self.cfg.USE_SLICED` with both enabled.

*Fixed:* both edges derived from the original array, adaptive width used, and
TTA composes with slicing instead of being mutually exclusive.

---

## High

**CM-05 — the CNN branch always returned 0.** Present in every row of every
session log. The saved checkpoint predicts 0.098 on real images. Not an
architecture problem — the same network trains fine on the project's own replay
buffer — but `ONLINE_BATCH=4` at `lr=3e-5` every 0.5 s is roughly one gradient
step per second, nowhere near convergence. Its labels are the detector's own
output, so the ceiling is "imitate YOLO" regardless.

*Fixed:* the final `nn.ReLU()` on the regression head (zero gradient wherever it
predicts 0) is now `nn.Softplus()`; measured MAE 0.30 → 0.26 with faster
convergence. The subsystem is off by default — see CM-06.

**CM-06 — online learning cost 40–60% of frame rate for nothing.**

| configuration | ms/frame | FPS |
|---|---|---|
| YOLO + pose only | 50.1 | 20.0 |
| + CNN, trainer at load | 77.9 | 12.8 |
| + CNN, trainer warming up | 129.2 | 7.7 |
| trainer stopped | 50.5 | 19.8 |

`body.track` alone went 24 ms → 81 ms under trainer load.

*Fixed:* `ENABLE_ONLINE_LEARNING` defaults off; a `NullOnlineLearner` keeps the
interface. Set `CROWD_MASTER_ONLINE_LEARNING=1` to restore. The replay buffer
(0.56 GB at full capacity) is not even loaded when disabled, and the model is no
longer overwritten on exit by runs that trained nothing.

**CM-07 — panic/stampede detection could never fire.** `AnalyticsThread` passed
`np.full((100,100),128)` to both detectors. Farneback optical flow on a constant
image is identically zero, so `RUNNING/PANIC`, `REVERSE FLOW` and
`CHAOTIC MOTION` were unreachable, and `DriftDetector`'s brightness term was
always 0.

*Fixed:* the worker publishes the real downscaled frame as `scene_gray`.
Anomaly scores are now non-zero in normal operation.

**CM-08 — the forecaster echoed the current count.** It needed 360 samples
before predicting anything (40–70 s of every session), and its training targets
clamped so all four horizons learned the same value. `steps` were documented as
seconds but history was appended per analytics tick.

*Fixed:* rewritten around a true 1 Hz history with a robust (Theil–Sen) trend and
damped extrapolation. Horizons are now distinct; with fewer than 8 s of history
it reports the current level rather than inventing a slope.

**CM-09 — API exposed on the network with no auth.** `host="0.0.0.0"`,
`allow_origins=["*"]`, no key.
*Fixed:* binds `127.0.0.1`, CORS empty unless configured, `X-API-Key` supported,
and it refuses to start off-loopback without a key.

**CM-10 — two locks over one model.** `OnlineLearner` and `AutoRetrainer` each
had their own `_lock` guarding the same weights.
*Fixed:* one lock, created in `run()` and shared.

**CM-11 — audio never resumed after pause.** `pause()` set `_paused=True`; the P
key called `seek()`, which no-ops while paused; `resume()` was never called from
anywhere.
*Fixed:* `seek(sec, resume=True)`.

---

## Medium

### Found by writing the gate test

Three defects that only surfaced once there was a test with a known right answer.

**CM-26 — adaptive resolution collapsed exactly when detail mattered most.** The
heuristic measured the wall-clock rate of the whole detection loop, including the
optional slicing pass. Slicing costs 83–131 ms, so engaging it dragged the
measured rate under target and the inference size walked down from 960×540
toward the 320×180 floor — shrinking the image precisely when the scene was dense
enough to need slicing, and dropping gate crossings as a result. The test caught
this as a persistent undercount (2 of 3 people) whenever slicing was forced on.
*Fixed:* only frames where the plain tracked pass ran alone feed the frame-rate
estimate, and the estimate needs 8 samples before it acts.

**CM-27 — auxiliary passes disturbed tracker state.** Ultralytics keeps one
predictor, and its tracker state, per `YOLO` object. Slicing called
`model(tile)` on the same object between `model.track()` calls, re-initialising
that predictor.
*Fixed:* slicing and TTA run on a second instance of the same weights (~22 MB
of VRAM); the tracked model is never touched by them.

**CM-28 — ID reassignment counted as a crossing.** A tracker ID freed by someone
leaving can be handed to someone entering on the other side of the line; the
side-change test read that as a crossing, producing phantom exits during a
one-way flow.
*Fixed:* a track whose anchor moves more than `MAX_JUMP_PX` (220 px) between
analysed frames is treated as a new track and re-seeded rather than counted.

**CM-29 — enabling optical flow starved the detector.** Fixing CM-07 (feeding
real frames to the anomaly detector) introduced a regression of its own: dense
Farneback flow is CPU-bound and holds the GIL, and running it on every analysed
frame pushed inference from ~40 ms to over 700 ms and climbing.
*Fixed:* flow runs on a fixed 5 Hz cadence, comparing frames one interval apart.
Count-based alerts still evaluate every tick.

### Other medium items

| # | Issue | Fix |
|---|---|---|
| 12 | GPU variant was mojibake with `77` prefixed to line 1, voiding the coding declaration; the launcher ran the other file | merged into `crowd_master_v2.py`, duplicate deleted |
| 13 | Log path ignored `CROWD_MASTER_DATA_DIR` | honours it |
| 14 | Dashboard drawn before `pz.apply()` so zoom magnified the HUD; gates drawn twice | HUD drawn after the transform; duplicate removed |
| 15 | `r` bound three times, `c` documented as volume but bound to snapshot, no volume-down | `R` context-sensitive, `V`/`C` volume, `S` snapshot |
| 16 | Gate dicts keyed by track ID, never pruned | `Gate.prune()` above 4000 entries |
| 17 | `MIN_CROSS_FRAMES=1` counted single-frame jitter as a crossing | raised to 3 |
| 18 | Replay buffer 588 KB/sample, 0.56 GB at capacity | not loaded unless online learning is on |
| 19 | Adaptive resolution changed `_cur_w`, TTA used `cfg.INFER_W` | uses `_cur_w` |
| 20 | CSV gated on `frame_id % 30` over sparse ids → gaps of 2–21 s | wall-clock `LOG_EVERY_SEC` |
| 21 | No `requirements.txt` | added, pinned |
| 22 | Zone counts went stale when the scene emptied | reset each frame |
| 23 | `highgui_available()` created and destroyed a probe window every frame | cached |
| 24 | 10.4 s before the first detection (cuDNN autotune) | warm-up on both the main and worker threads; now 0.6 s |
| 25 | No git history, no tests | repo initialised, gate regression test added |

---

## Measurements worth keeping

**`yolov8s` is faster than `yolov8n` here** — 18.6 ms vs 25.0 ms at 960×540 FP16.
The nano model is too small to saturate the GPU. Upgrading the models was the
right call; the problem was the slicing and TTA enabled alongside them.

**Running the body and pose models back-to-back is nearly free** — 37.2 ms
interleaved versus 36.4 ms for the body model alone. The GPU pipelines them.

**BoT-SORT is not worth it on this build** — 33.5 ms vs 15.9 ms for ByteTrack,
for 7 distinct IDs versus 8 over 60 frames. Its global motion compensation throws
on OpenCV 5.0 (`GMC failed, falling back to identity`), so its main advantage is
unavailable. ByteTrack remains the default; `CROWD_MASTER_TRACKER` switches.

---

## Still outstanding

Deliberately not done, and why:

- **Package split.** Still one 4,000-line file. Splitting it is the right next
  step but it is a large mechanical change with real regression risk, and it
  buys nothing functional today.
- **Ground-truth accuracy harness.** The gate test proves counting logic on
  synthetic footage; there is still no labelled real clip to measure detection
  MAE against. That needs ~200 hand-counted frames from your own cameras — a
  data task, not a code task, and the one thing that would let you tell whether
  a future change helps or hurts accuracy.
- **Gate accuracy in dense flow — measured, and it is the real limit.** The
  suite now runs the crowded case and prints the result instead of asserting on
  it. With three walkers crossing together at the same speed, ByteTrack swaps
  their IDs: a trace showed one ID credited with two crossings while a third
  walker was never counted, and a single ID's reported x jumping 375 px between
  consecutive frames. BoT-SORT behaves the same. Turning cuDNN autotuning off
  was enough to flip the total between 2 and 3 — and the run that produced 3 was
  right only by cancellation, not because it counted correctly.

  So the honest statement is: gate counting is dependable when people cross one
  at a time or clearly separated, and is not dependable for dense simultaneous
  flow. Closing that gap needs appearance-based re-identification, not threshold
  tuning. Until then the asserted tests cover what the system can actually
  guarantee, and the crowded case is reported so a regression there is visible
  without failing the suite for unrelated reasons.
- **Head detection via a full pose model.** Running `yolov8s-pose` purely to get
  head centres costs ~17–34 ms/frame. Deriving the head from the upper part of
  the body box, or training a dedicated head detector, would reclaim most of it.
- **Multi-camera.** Single-camera assumptions are baked into `run()`.
- **WebSocket / event clip recording.** API is still poll-only.
- **The CNN regressor itself.** Softplus head aside, it needs real labels to be
  worth its inference cost. Left disabled rather than deleted.
