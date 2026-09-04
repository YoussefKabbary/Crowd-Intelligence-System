# Crowd Master

Real-time crowd analytics for video files and live cameras: person detection and
tracking, gate (door) entry/exit counting, zone occupancy, heatmaps, anomaly
alerts, and a per-session PDF/CSV report.

Built on YOLOv8 (detection + pose) with ByteTrack, PyTorch, and OpenCV.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then either double-click `run_crowd_master.cmd`, or:

```bash
venv\Scripts\python.exe crowd_master_v2.py
```

With no `CROWD_MASTER_VIDEO` set, a launcher window opens so you can browse for
a video or pick a camera.

## Keys

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `Q` | quit | `G` | gate draw mode (click + drag) |
| `P` | pause | `TAB` | select next gate |
| `N` | stats panel | `R` | reverse selected gate / reset view |
| `B` | boxes | `X` | delete selected gate / clear all |
| `H` | heads | `ESC` | deselect gate |
| `M` | heatmap | `S` | save snapshot |
| `Z` | zones | `V` / `C` | volume up / down |
| `F` | fullscreen | `+` / `-` | playback speed |
| `I` | image enhance | `]` / `[` | zoom |
| `T` | test-time augmentation | `W A D` + arrows | pan |
| `O` | detection on/off | `.` / `,` | seek ±5 s |
| `K` | controls overlay | `E` | export CNN to ONNX |
| `Ctrl+R` | trigger retrain | | |

Gates are saved to `DATA/gates.json` and reloaded automatically.

## Configuration

Everything is overridable by environment variable — no code edits needed.

| Variable | Default | Meaning |
|---|---|---|
| `CROWD_MASTER_VIDEO` | `test_video.mp4` | video file; skips the launcher |
| `CROWD_MASTER_INPUT_MODE` | `file` | `file` \| `webcam` \| `rtsp` |
| `CROWD_MASTER_CAMERA_INDEX` | `0` | webcam index |
| `CROWD_MASTER_CAMERA_URL` | — | RTSP URL |
| `CROWD_MASTER_DATA_DIR` | `./DATA` | all outputs, logs and reports |
| `CROWD_MASTER_DEVICE` | auto | `cuda` \| `cpu` |
| `CROWD_MASTER_FP16` | `1` | half precision on GPU |
| `CROWD_MASTER_BODY_MODEL` | `yolov8s.pt` | body detector weights |
| `CROWD_MASTER_HEAD_MODEL` | `yolov8s-pose.pt` | pose model used for heads |
| `CROWD_MASTER_TRACKER` | `bytetrack.yaml` | or `botsort.yaml` |
| `CROWD_MASTER_ONLINE_LEARNING` | `0` | see "Online learning" below |
| `CROWD_MASTER_LOG_EVERY_SEC` | `2` | CSV sampling interval |
| `CROWD_MASTER_HEADLESS` | `0` | no window, no key waits |
| `CROWD_MASTER_MAX_FRAMES` | `0` | stop after N frames (0 = unlimited) |
| `CROWD_MASTER_LOOP` | `1` (`0` headless) | loop the video at end |
| `CROWD_MASTER_API_HOST` | `127.0.0.1` | see "API" below |
| `CROWD_MASTER_API_PORT` | `5055` | |
| `CROWD_MASTER_API_KEY` | — | required to bind off-loopback |
| `CROWD_MASTER_API_ORIGINS` | — | comma-separated CORS origins |

### Model choice

On the RTX 3050 this project targets, `yolov8s` is **faster** than `yolov8n`
(18.6 ms vs 25.0 ms at 960×540 FP16) — the nano model is too small to keep the
GPU busy. `s` is therefore the default for both detectors.

### Online learning

The CNN crowd regressor and its online trainer are **off by default**. Measured
on this hardware they cost 40–60% of the frame rate (20.0 FPS → 7.7–12.8 FPS)
while contributing a count of 0 in every logged session, because they train on
the detector's own output rather than on ground truth. Enable with
`CROWD_MASTER_ONLINE_LEARNING=1` if you want to work on them.

### API

`GET /api/status` returns live counts, gate totals, forecasts and anomalies;
`GET /api/health` is a liveness probe.

The server binds **loopback only** by default. To expose it on the network you
must set both `CROWD_MASTER_API_HOST` and `CROWD_MASTER_API_KEY` — it refuses to
start off-loopback without a key, and clients then send `X-API-Key`.

## Headless / batch

```bash
CROWD_MASTER_HEADLESS=1 CROWD_MASTER_MAX_FRAMES=400 \
CROWD_MASTER_VIDEO=clip.mp4 CROWD_MASTER_DATA_DIR=out \
venv/Scripts/python.exe crowd_master_v2.py
```

Runs with no window, stops on its own, and writes the CSV, TXT and PDF reports.

## Tests

```bash
venv/Scripts/python.exe tests/test_gate_counting.py
```

Generates a clip with a known number of people crossing a known line and asserts
the gate reports exactly that many entries. This is the regression test for the
v5.2 bug where enabling sliced inference silently zeroed every gate count.

## Outputs

Everything lands in `DATA/` (or `CROWD_MASTER_DATA_DIR`):

| File | Contents |
|---|---|
| `crowd_log.csv` | time series: counts, forecasts, anomaly and drift scores |
| `crowd_report.txt` | session report — peaks, crowded periods, gate totals |
| `crowd_report.pdf` | the same report, formatted |
| `crowd_master.log` | run log |
| `gates.json` | gate geometry, reloaded on next run |
| `drift_events.csv` | scene-drift events |

## Notes

See `AUDIT.md` for the 2026-09-04 code review: what was broken, how it was
measured, and what is still outstanding.
