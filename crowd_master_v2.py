# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  CROWD MASTER  |  Enterprise AI Surveillance Platform  v5.3         ║
║  ─────────────────────────────────────────────────────────────────  ║
║  NEW IN v5.3  (audit fixes — see AUDIT.md):                          ║
║    ✓ Tracking is never lost: slicing/TTA now augment the tracked    ║
║      pass instead of replacing it. Gate counting works again.       ║
║    ✓ NMS rewritten — it was fed xyxy boxes as xywh and discarded    ║
║      the weakest detection every frame (~20% undercount).           ║
║    ✓ TTA flip un-mirror fixed (used to emit boxes with x1 > x2).    ║
║    ✓ Online learning OFF by default: cost 40-60% of frame rate      ║
║      and contributed a CNN count of 0 in every logged session.      ║
║    ✓ Anomaly/drift detectors now see the real frame, not a          ║
║      constant grey image — panic/stampede alerts can fire at all.   ║
║    ✓ Forecaster rewritten: true 1 Hz history + robust trend.        ║
║    ✓ API binds loopback and supports an API key.                    ║
║    ✓ Headless/batch mode for scripted runs and tests.               ║
║                                                                      ║
║  MERGED: crowd_master_v2_gpu_max.py is folded in and deleted.        ║
║                                                                      ║
║  ARCHITECTURE: 6-Thread async pipeline                               ║
║    VideoPlayerThread  → exact-FPS playback, independent            ║
║    DetectionWorker    → YOLO + head + sliced + TTA (async)          ║
║    AnalyticsThread    → drift / forecast / anomaly                  ║
║    OnlineLearner      → continuous CNN fine-tuning                  ║
║    AutoRetrainer      → nightly deep retrain                        ║
║    Main Thread        → display only (zero lag)                     ║
║                                                                      ║
║  GATE / DOOR SYSTEM:                                                 ║
║    G  → enter gate draw mode (click+drag to place gates)            ║
║    X  → clear all gates                                             ║
║    Gates saved to gates.json, auto-reloaded on next run             ║
║                                                                      ║
║  KEYS:                                                               ║
║    Q quit   P pause   N stats   B boxes   H heads   M heatmap       ║
║    Z zones  F full    I enhance  T TTA    E ONNX    Ctrl+R train    ║
║    G gate-draw   TAB select-gate   R reverse   ESC deselect         ║
║    X delete-selected  (X clears all when nothing selected)          ║
║    O  toggle detection on/off (tracking keeps running)              ║
║    +/-  speed   V/C volume up/down   ]/[ zoom                       ║
║    S  save snapshot     WAD + arrow keys pan (down = down arrow)    ║
║    .  seek+5s   ,  seek-5s   (audio starts automatically)           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2, csv, os, sys, time, logging, math, threading, pickle
import warnings, random, queue, json, subprocess, tempfile, shutil
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as T
    import torchvision.models as models
    from torch.utils.data import DataLoader, TensorDataset
    from ultralytics import YOLO
    from PIL import Image
except ImportError as _exc:
    # Almost always this means the script was launched with the wrong Python —
    # the system interpreter instead of the project's venv (VS Code's Run button
    # uses whichever interpreter is selected, which is not necessarily this one).
    _venv_py = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
    _msg = [
        "",
        "=" * 68,
        f"  Crowd Master cannot start: {_exc.name!r} is not installed",
        "  in the Python that launched it.",
        "=" * 68,
        f"  Running under : {sys.executable}",
    ]
    if _venv_py.exists():
        _msg += [
            f"  Should be     : {_venv_py}",
            "",
            "  Run it with the project's environment instead:",
            "",
            f'      cd "{Path(__file__).resolve().parent}"',
            "      venv\\Scripts\\python.exe crowd_master_v2.py",
            "",
            "  Or just double-click run_crowd_master.cmd.",
            "",
            "  In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter'",
            "              -> .\\venv\\Scripts\\python.exe",
        ]
    else:
        _msg += [
            "",
            "  No venv found next to this script. Create one:",
            "",
            "      python -m venv venv",
            "      venv\\Scripts\\python.exe -m pip install torch==2.11.0 "
            "torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130",
            "      venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        ]
    _msg += ["=" * 68, ""]
    print("\n".join(_msg), file=sys.stderr)
    sys.exit(1)

# ── Tkinter GUI launcher ─────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# ── GPU performance tuning ───────────────────────────────────
torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]         = "TRUE"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ★ FIX: this was hardcoded to <script dir>/DATA and ignored CROWD_MASTER_DATA_DIR,
# so the log landed somewhere other than every other output of the same run.
_LOG_DIR  = Path(os.getenv("CROWD_MASTER_DATA_DIR", Path(__file__).parent / "DATA"))
_LOG_FILE = str(_LOG_DIR / "crowd_master.log")
Path(_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("CrowdMaster")

_HIGHGUI_AVAILABLE = None
_HIGHGUI_WARNING_LOGGED = False
_HIGHGUI_LAST_CHECK = 0.0
_HIGHGUI_RECHECK_SEC = 5.0


def highgui_available() -> bool:
    """Return True when this OpenCV build can create desktop windows.

    ★ FIX: this used to create and destroy a probe window on *every* call, and
    safe_wait_key() calls it once per frame. A positive result is now cached for
    good; a negative one is re-checked at most every _HIGHGUI_RECHECK_SEC.
    """
    global _HIGHGUI_AVAILABLE, _HIGHGUI_WARNING_LOGGED, _HIGHGUI_LAST_CHECK
    if _HIGHGUI_AVAILABLE is True:
        return True
    now = time.monotonic()
    if (_HIGHGUI_AVAILABLE is False
            and now - _HIGHGUI_LAST_CHECK < _HIGHGUI_RECHECK_SEC):
        return False
    _HIGHGUI_LAST_CHECK = now

    test_win = "__crowd_master_highgui_test__"
    try:
        cv2.namedWindow(test_win, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)          # pump the event queue so the window actually paints
        cv2.destroyWindow(test_win)
        cv2.waitKey(1)          # ...and again so the destroy actually takes
        _HIGHGUI_AVAILABLE = True
    except cv2.error as exc:
        _HIGHGUI_AVAILABLE = False
        if not _HIGHGUI_WARNING_LOGGED:
            logger.warning(
                "OpenCV HighGUI is unavailable (%s). Install a GUI-enabled OpenCV wheel "
                "with: pip uninstall -y opencv-python-headless opencv-python; "
                "pip install opencv-python",
                str(exc).splitlines()[0],
            )
            _HIGHGUI_WARNING_LOGGED = True
    return _HIGHGUI_AVAILABLE


def safe_wait_key(delay: int = 1) -> int:
    if not highgui_available():
        time.sleep(max(delay, 1) / 1000.0)
        return -1
    try:
        return cv2.waitKey(delay)
    except cv2.error:
        return -1


# ══════════════════════════════════════════════════════════════
#  CONFIG  — single source of truth for all settings
# ══════════════════════════════════════════════════════════════
class Config:
    # ── Paths - Cross-Platform with Environment Variables ──────
    # Set VIDEO_PATH environment variable or put test_video.mp4 in same folder
    _base_dir = Path(__file__).parent
    _data_dir = Path(os.getenv("CROWD_MASTER_DATA_DIR", _base_dir / "DATA"))
    
    # Create data directory if it doesn't exist
    _data_dir.mkdir(parents=True, exist_ok=True)
    
    # Default video path - can be overridden with CROWD_MASTER_VIDEO env var
    _default_video = _base_dir / "test_video.mp4"
    VIDEO_PATH  = str(Path(os.getenv("CROWD_MASTER_VIDEO", _default_video)))
    INPUT_MODE  = os.getenv("CROWD_MASTER_INPUT_MODE", "file")  # file | webcam | rtsp
    CAMERA_INDEX = int(os.getenv("CROWD_MASTER_CAMERA_INDEX", "0"))
    CAMERA_URL   = os.getenv("CROWD_MASTER_CAMERA_URL", "")
    CAMERA_BUFFER_SIZE = 1
    SHOW_CONTROLS = True
    
    LOG_FILE    = str(_data_dir / "crowd_log.csv")
    REPORT_FILE = str(_data_dir / "crowd_report.txt")
    PDF_REPORT  = str(_data_dir / "crowd_report.pdf")
    MODEL_SAVE  = str(_data_dir / "best_crowd_model.pth")
    BUFFER_SAVE = str(_data_dir / "replay_buffer.pkl")
    DRIFT_LOG   = str(_data_dir / "drift_events.csv")
    ONNX_PATH   = str(_data_dir / "crowd_model.onnx")
    GATES_FILE  = str(_data_dir / "gates.json")

    # ── Models ─────────────────────────────────────────────────
    # yolov8s benchmarks FASTER than yolov8n on the RTX 3050 (18.6 vs 25.0 ms
    # @960x540 FP16) — the nano model is too small to saturate the GPU.
    BODY_MODEL  = os.getenv("CROWD_MASTER_BODY_MODEL", "yolov8s.pt")
    HEAD_MODEL  = os.getenv("CROWD_MASTER_HEAD_MODEL", "yolov8s-pose.pt")

    # ── Tracker ────────────────────────────────────────────────
    # Measured on this machine (yolov8s @960x540, 60 frames, 6 people):
    #   bytetrack  15.9 ms/frame, 8 distinct IDs
    #   botsort    33.5 ms/frame, 7 distinct IDs
    # botsort is marginally more stable but twice as slow, and its global motion
    # compensation throws on this OpenCV 5.0 build ("GMC failed, falling back to
    # identity") so its main advantage is not actually available. bytetrack it
    # is; set CROWD_MASTER_TRACKER=botsort.yaml to reconsider on a newer OpenCV.
    TRACKER     = os.getenv("CROWD_MASTER_TRACKER", "bytetrack.yaml")

    # ── Inference resolution ────────────────────────────────────
    INFER_W      = 960
    INFER_H      = 540
    DETECT_EVERY = 1         # ★ FIX: was 3 → faster detection per frame

    # ── Adaptive confidence ────────────────────────────────────
    BODY_CONF_BASE = 0.25
    BODY_IOU       = 0.45
    HEAD_CONF_BASE = 0.20
    HEAD_IOU       = 0.35
    CROWD_THRESHOLD = 15

    # ── Sliced inference ────────────────────────────────────────
    # ★ FIX: slicing used to REPLACE the tracked detection pass, which silently
    # destroyed all track IDs (and with them every gate count). It is now an
    # augmentation layered on top of tracking, and it only engages once the
    # scene is genuinely dense — a 2x2 slice costs 83-131 ms, so paying it on an
    # empty corridor is pure waste.
    USE_SLICED    = True
    SLICE_ROWS    = 2
    SLICE_COLS    = 2
    SLICE_OVERLAP = 0.20

    # ── Test-Time Augmentation ─────────────────────────────────
    # Off by default: a horizontal-flip pass costs a full extra inference for a
    # marginal recall gain. Toggle at runtime with the T key.
    USE_TTA = False

    # ── Detection fusion ───────────────────────────────────────
    FUSION_DIST         = 55
    HEAD_PERSIST_FRAMES = 5

    # ── Dense-scene fallback threshold ─────────────────────────
    DENSE_FALLBACK_THR  = 40

    # ── Gate / Door system ─────────────────────────────────────
    GATE_LINE_THICKNESS = 3
    GATE_CROSS_DIST     = 70  # ★ FIX: was 40 → wider zone catches more crossings
    GATE_COLORS         = [
        (0, 220, 255), (0, 255, 100), (255, 150, 0),
        (200, 0, 255), (0, 100, 255), (255, 50, 50),
    ]

    # ── Display ────────────────────────────────────────────────
    HEAD_DOT_RADIUS = 2
    HEAD_DOT_COLOR  = (0, 0, 220)
    HEAD_DOT_BORDER = (255, 255, 255)
    BOX_COLOR_LOW   = (0, 200,   0)
    BOX_COLOR_MED   = (0, 165, 255)
    BOX_COLOR_HIGH  = (0,   0, 255)

    # ── Playback ───────────────────────────────────────────────
    DEFAULT_SPEED = 1.0
    MIN_SPEED     = 0.1
    MAX_SPEED     = 8.0
    SPEED_STEP    = 0.25

    # ── Zoom ─────────────────────────────────────────────────
    DEFAULT_ZOOM = 1.0
    MIN_ZOOM     = 0.3
    MAX_ZOOM     = 10.0
    ZOOM_STEP    = 0.25
    PAN_STEP     = 60
    ZOOM_INTERP  = cv2.INTER_LANCZOS4

    # ── Online learning ────────────────────────────────────────
    # ★ DISABLED BY DEFAULT. Measured cost: 20.0 FPS -> 7.7-12.8 FPS, because the
    # training thread competes with detection for the same GPU (body.track alone
    # went 24 ms -> 81 ms under trainer load).
    # Measured benefit: zero. The CNN column is 0 in every row of every session
    # log, and its labels are YOLO's own output, so the best it can ever learn is
    # to imitate the detector it is supposed to be helping.
    # Set CROWD_MASTER_ONLINE_LEARNING=1 to re-enable.
    ENABLE_ONLINE_LEARNING = os.getenv("CROWD_MASTER_ONLINE_LEARNING", "0") == "1"
    ONLINE_LR           = 3e-5
    ONLINE_BATCH        = 32
    ONLINE_EVERY_SEC    = 2
    BUFFER_SIZE         = 1000
    MIN_BUFFER_TO_TRAIN = 16

    # ── Drift ──────────────────────────────────────────────────
    DRIFT_WINDOW    = 30
    DRIFT_THRESHOLD = 0.40

    # ── Forecasting (Transformer) ──────────────────────────────
    FORECAST_STEPS = [5, 30, 60, 300]

    # ── Zones ──────────────────────────────────────────────────
    ZONES = {
        "Entrance Door": (0.17, 0.15, 0.30, 0.60),
        "prayer hall":    (0.0,  0.7,  1.0,  1.0),
        "Prayer Hall": (0.32,  0.1,  0.9,  0.6),
        "Corridor B":  (0.0,  0.1,  0.1,  0.7),
        "IMAM Membar":  (0.9,  0.1,  1.0,  0.7)
    }

    # ── Anomaly ────────────────────────────────────────────────
    SPEED_THRESH        = 30.0
    DENSITY_JUMP_THRESH = 8

    # ── Heatmap ────────────────────────────────────────────────
    HEATMAP_DECAY  = 0.96
    HEATMAP_RADIUS = 69

    # ── REST API ────────────────────────────────────────────────
    # ★ FIX: used to bind 0.0.0.0 with allow_origins=["*"] and no auth, exposing
    # live occupancy and gate traffic to every device on the network.
    # Binds to loopback only unless you explicitly opt out, and requires a key
    # whenever it is bound to anything else.
    API_ENABLED = True
    API_PORT    = int(os.getenv("CROWD_MASTER_API_PORT", "5055"))
    API_HOST    = os.getenv("CROWD_MASTER_API_HOST", "127.0.0.1")
    API_KEY     = os.getenv("CROWD_MASTER_API_KEY", "")
    API_ORIGINS = [o for o in
                   os.getenv("CROWD_MASTER_API_ORIGINS", "").split(",") if o]

    # ── Auto retrain ───────────────────────────────────────────
    RETRAIN_HOUR   = 3
    RETRAIN_EPOCHS = 40

    LOG_EVERY = 30            # legacy frame-modulo cadence, no longer used
    LOG_EVERY_SEC = float(os.getenv("CROWD_MASTER_LOG_EVERY_SEC", "2"))

    # ── Headless / batch mode ──────────────────────────────────
    # CROWD_MASTER_HEADLESS=1     -> never open a window, never wait for a key
    # CROWD_MASTER_MAX_FRAMES=N   -> quit automatically after N displayed frames
    # Together these make the app scriptable and testable in CI.
    HEADLESS   = os.getenv("CROWD_MASTER_HEADLESS", "0") == "1"
    MAX_FRAMES = int(os.getenv("CROWD_MASTER_MAX_FRAMES", "0"))
    # Files loop by default so a demo clip keeps playing; a batch run wants the
    # process to end when the footage does.
    LOOP_VIDEO = os.getenv("CROWD_MASTER_LOOP", "0" if HEADLESS else "1") == "1"
    
    # ── Device - Auto-detect with environment override ────────
    _device_env = os.getenv("CROWD_MASTER_DEVICE", "").lower()
    if _device_env in ("cuda", "cpu"):
        DEVICE = _device_env
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── FP16 (half precision) — 2× faster on GPU, no accuracy loss ──
    # Automatically enabled when CUDA is available.
    # Set CROWD_MASTER_FP16=0 to force disable.
    USE_FP16 = (DEVICE == "cuda") and (os.getenv("CROWD_MASTER_FP16", "1") != "0")

    # ── Adaptive resolution ───────────────────────────────────
    # If FPS drops below target, inference resolution is reduced.
    # Set to 0 to disable adaptive scaling.
    TARGET_FPS       = 25       # FPS below this triggers downscale
    INFER_W_MIN      = 320      # minimum adaptive width
    INFER_H_MIN      = 180      # minimum adaptive height

    # ── EMA smoothing alpha for track centers ─────────────────
    TRACK_EMA_ALPHA  = 0.70     # 0.7 = fast response, smooth motion

    def model_save_path(self) -> str:
        return self.MODEL_SAVE


# ══════════════════════════════════════════════════════════════
#  GATE
# ══════════════════════════════════════════════════════════════

def resolve_video_source(cfg: Config):
    mode = str(getattr(cfg, "INPUT_MODE", "file")).lower().strip()
    if mode == "webcam":
        return int(getattr(cfg, "CAMERA_INDEX", 0))
    if mode == "rtsp":
        return str(getattr(cfg, "CAMERA_URL", "")).strip()
    source = str(getattr(cfg, "VIDEO_PATH", "")).strip()
    return int(source) if source.isdigit() else source


def is_live_source(cfg: Config) -> bool:
    return str(getattr(cfg, "INPUT_MODE", "file")).lower().strip() in ("webcam", "rtsp")


def source_label(cfg: Config) -> str:
    mode = str(getattr(cfg, "INPUT_MODE", "file")).lower().strip()
    if mode == "webcam":
        return f"Webcam {getattr(cfg, 'CAMERA_INDEX', 0)}"
    if mode == "rtsp":
        url = str(getattr(cfg, "CAMERA_URL", "")).strip()
        return "IP Camera" if len(url) > 42 else f"IP Camera {url}"
    return Path(str(getattr(cfg, "VIDEO_PATH", ""))).name
class Gate:
    # ★ FIX: was 1, which meant a single frame of box jitter across the line
    # registered as a crossing. A person genuinely walking through a gate is
    # inside the band for several frames; noise is not.
    MIN_CROSS_FRAMES = 3
    RESET_DIST_MULT  = 1.8   # ★ FIX: was 2.5 → re-trigger sooner
    MAX_TRACKED_IDS  = 4000  # bound on the per-track bookkeeping dicts
    # Largest believable movement of one person between two analysed frames.
    # Anything beyond this is a tracker ID reassignment, not motion.
    MAX_JUMP_PX      = 220

    def __init__(self, gate_id: int, x1: int, y1: int,
                 x2: int, y2: int, color: Tuple,
                 direction_mode: str = "auto",
                 in_side: str = "positive"):
        self.gate_id       = gate_id
        self.x1, self.y1   = x1, y1
        self.x2, self.y2   = x2, y2
        self.color          = color
        self.direction_mode = direction_mode
        self.in_side        = in_side
        self.entry          = 0
        self.exit           = 0

        self._side_mem:   Dict[int, int] = {}
        self._zone_frames: Dict[int, int] = {}
        self._crossed:    set = set()
        self._last_pt:    Dict[int, Tuple[int, int]] = {}

        self._dx  = x2 - x1
        self._dy  = y2 - y1
        self._len = math.hypot(self._dx, self._dy) + 1e-9

    def to_dict(self) -> dict:
        return {
            "id":             self.gate_id,
            "x1": self.x1,   "y1": self.y1,
            "x2": self.x2,   "y2": self.y2,
            "direction_mode": self.direction_mode,
            "in_side":        self.in_side,
        }

    @staticmethod
    def from_dict(d: dict, color: Tuple) -> "Gate":
        g = Gate(d["id"], d["x1"], d["y1"], d["x2"], d["y2"], color,
                 direction_mode=d.get("direction_mode", "auto"),
                 in_side=d.get("in_side", "positive"))
        return g

    def reverse_direction(self):
        self.in_side = "negative" if self.in_side == "positive" else "positive"
        logger.info(f"Gate {self.gate_id}: direction reversed → in_side={self.in_side}")

    def _raw_side(self, px: int, py: int) -> int:
        cross = (px - self.x1) * self._dy - (py - self.y1) * self._dx
        return 1 if cross >= 0 else -1

    def _entry_side(self) -> int:
        if self.in_side == "negative":
            return -1
        return 1

    def _dist(self, px: int, py: int) -> float:
        return abs((px - self.x1) * self._dy -
                   (py - self.y1) * self._dx) / self._len

    def update_multi(self,
                     tid: int,
                     body_cx: Optional[int], body_cy: Optional[int],
                     head_cx: Optional[int], head_cy: Optional[int],
                     fused_cx: Optional[int], fused_cy: Optional[int],
                     cross_dist: int = 55) -> Optional[str]:
        cx, cy = None, None
        anchor_used = "none"
        for (ax, ay, lbl) in [
            (fused_cx, fused_cy, "fused"),
            (body_cx,  body_cy,  "body"),
            (head_cx,  head_cy,  "head"),
        ]:
            if ax is not None and ay is not None:
                cx, cy, anchor_used = ax, ay, lbl
                break

        if cx is None:
            return None

        # ── Reject teleporting tracks ─────────────────────────
        # A tracker ID can be reassigned to a different person (one leaves the
        # frame as another enters). The new occupant may be on the opposite side
        # of the line, which the side-change test below reads as a crossing —
        # observed as a phantom "exit" while three people all walked in the same
        # direction. A real person cannot jump this far between frames, so treat
        # a large jump as a new track and re-seed rather than count it.
        prev_pt = self._last_pt.get(tid)
        self._last_pt[tid] = (cx, cy)
        if prev_pt is not None:
            if math.hypot(cx - prev_pt[0], cy - prev_pt[1]) > self.MAX_JUMP_PX:
                self._side_mem[tid]    = self._raw_side(cx, cy)
                self._zone_frames[tid] = 0
                self._crossed.discard(tid)
                return None

        dist    = self._dist(cx, cy)
        reset_d = cross_dist * self.RESET_DIST_MULT

        if dist > cross_dist:
            self._side_mem[tid]    = self._raw_side(cx, cy)
            self._zone_frames[tid] = 0
            if dist > reset_d:
                self._crossed.discard(tid)
            return None

        current = self._raw_side(cx, cy)
        prev    = self._side_mem.get(tid)

        self._zone_frames[tid] = self._zone_frames.get(tid, 0) + 1

        if prev is None:
            self._side_mem[tid] = current
            return None

        if self._zone_frames.get(tid, 0) < self.MIN_CROSS_FRAMES:
            return None

        if current != prev and tid not in self._crossed:
            self._side_mem[tid] = current
            self._crossed.add(tid)
            is_entry = (current == self._entry_side())
            if is_entry:
                self.entry += 1
                return "entry"
            else:
                self.exit += 1
                return "exit"

        self._side_mem[tid] = current
        return None

    def update(self, tid: int, cx: int, cy: int,
               cross_dist: int = 55) -> Optional[str]:
        return self.update_multi(tid, cx, cy, None, None, None, None,
                                 cross_dist)

    def prune(self, live_ids: set):
        """Drop bookkeeping for track IDs that no longer exist.

        ★ FIX: _side_mem / _zone_frames / _crossed were keyed by tracker ID and
        never cleaned. Tracker IDs increase forever, so a camera left running
        leaked memory for the life of the process.
        """
        if len(self._side_mem) <= self.MAX_TRACKED_IDS:
            return
        for d in (self._side_mem, self._zone_frames, self._last_pt):
            for k in [k for k in d if k not in live_ids]:
                d.pop(k, None)
        self._crossed &= live_ids

    @property
    def total(self) -> int:
        return self.entry + self.exit

    def draw(self, frame: np.ndarray, scale: float = 1.0,
             selected: bool = False):
        x1 = int(self.x1 * scale); y1 = int(self.y1 * scale)
        x2 = int(self.x2 * scale); y2 = int(self.y2 * scale)

        thickness = 5 if selected else 3
        cv2.line(frame, (x1,y1), (x2,y2), self.color, thickness, cv2.LINE_AA)
        if selected:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 1, cv2.LINE_AA)

        mx = (x1+x2)//2; my = (y1+y2)//2
        nx = -self._dy / self._len
        ny =  self._dx / self._len
        sign = 1 if self._entry_side() == 1 else -1
        ax = int(mx + sign*nx*32*scale)
        ay = int(my + sign*ny*32*scale)
        cv2.arrowedLine(frame, (mx,my), (ax,ay), (0,255,0), 2, tipLength=0.35)

        mode_tag = " [REV]" if self.in_side == "negative" else ""
        label = f"G{self.gate_id} In:{self.entry} Out:{self.exit}{mode_tag}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        lx = mx - tw//2; ly = my - 14
        cv2.rectangle(frame, (lx-3, ly-th-3), (lx+tw+3, ly+3), (0,0,0), -1)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  GATE MANAGER
# ══════════════════════════════════════════════════════════════
class GateManager:
    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.gates: List[Gate] = []
        self._draw_mode   = False
        self._drawing     = False
        self._start_pt    = None
        self._preview     = None
        self._win_name    = ""
        self._scale       = 1.0
        self.selected_idx = -1
        self._load()

    def _color(self, idx: int) -> Tuple:
        return self.cfg.GATE_COLORS[idx % len(self.cfg.GATE_COLORS)]

    def _load(self):
        p = self.cfg.GATES_FILE
        if not os.path.exists(p):
            return
        try:
            with open(p) as f:
                data = json.load(f)
            self.gates = [Gate.from_dict(d, self._color(i))
                          for i, d in enumerate(data)]
            logger.info(f"Loaded {len(self.gates)} gate(s) from {p}")
        except Exception as e:
            logger.warning(f"Gate load failed: {e}")

    def save(self):
        p = self.cfg.GATES_FILE
        try:
            with open(p, "w") as f:
                json.dump([g.to_dict() for g in self.gates], f, indent=2)
            logger.info(f"Saved {len(self.gates)} gate(s) to {p}")
        except Exception as e:
            logger.warning(f"Gate save failed: {e}")

    def toggle_draw_mode(self, win_name: str, display_scale: float):
        self._draw_mode = not self._draw_mode
        self._win_name  = win_name
        self._scale     = display_scale
        if self._draw_mode:
            logger.info("Gate draw mode ON — click+drag to draw a gate line.")
            cv2.setMouseCallback(win_name, self._mouse_cb)
        else:
            logger.info("Gate draw mode OFF — gates saved.")
            cv2.setMouseCallback(win_name, lambda *a: None)
            self.save()

    def clear_all(self):
        self.gates         = []
        self.selected_idx  = -1
        self.save()
        logger.info("All gates cleared.")

    def select_next(self):
        if not self.gates:
            return
        self.selected_idx = (self.selected_idx + 1) % len(self.gates)
        logger.info(f"Gate selected: G{self.gates[self.selected_idx].gate_id}")

    def reverse_selected(self):
        if 0 <= self.selected_idx < len(self.gates):
            self.gates[self.selected_idx].reverse_direction()
            self.save()

    def delete_selected(self):
        if 0 <= self.selected_idx < len(self.gates):
            gid = self.gates[self.selected_idx].gate_id
            del self.gates[self.selected_idx]
            self.selected_idx = min(self.selected_idx, len(self.gates)-1)
            self.save()
            logger.info(f"Gate G{gid} deleted.")

    def _mouse_cb(self, event, x, y, flags, param):
        ox = int(x / self._scale)
        oy = int(y / self._scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing  = True
            self._start_pt = (ox, oy)
            self._preview  = (ox, oy)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._preview = (ox, oy)

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            if self._start_pt and self._preview:
                x1, y1 = self._start_pt
                x2, y2 = self._preview
                if math.hypot(x2-x1, y2-y1) > 20:
                    gid = len(self.gates)
                    self.gates.append(
                        Gate(gid, x1, y1, x2, y2, self._color(gid)))
                    self.selected_idx = gid
                    logger.info(f"Gate {gid} created: ({x1},{y1})→({x2},{y2})")
            self._start_pt = self._preview = None

    def draw_preview(self, display_frame: np.ndarray, scale: float = 1.0):
        for i, g in enumerate(self.gates):
            g.draw(display_frame, scale, selected=(i == self.selected_idx))

        if self._draw_mode:
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, 0), (w, 30), (0, 0, 170), -1)
            cv2.putText(display_frame,
                        "GATE DRAW MODE  —  Click+drag to place gate. Press G to finish.",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255,255,255), 1, cv2.LINE_AA)

            if self._drawing and self._start_pt and self._preview:
                sx = int(self._start_pt[0] * scale)
                sy = int(self._start_pt[1] * scale)
                ex = int(self._preview[0]  * scale)
                ey = int(self._preview[1]  * scale)
                cv2.line(display_frame, (sx,sy), (ex,ey),
                         self._color(len(self.gates)), 2, cv2.LINE_AA)

    def update_tracking(self,
                        body_boxes_orig: np.ndarray,
                        body_ids: Optional[np.ndarray],
                        head_centers: Optional[List[Tuple[int,int]]] = None,
                        fused_centers: Optional[Dict[int, Tuple[int,int]]] = None):
        if body_ids is None or len(body_boxes_orig) == 0:
            return

        cross_d = self.cfg.GATE_CROSS_DIST
        live_ids = {int(t) for t in np.asarray(body_ids).tolist()}
        for gate in self.gates:
            gate.prune(live_ids)

        for i, (box, tid) in enumerate(zip(body_boxes_orig, body_ids)):
            bcx = int((box[0]+box[2]) / 2)
            bcy = int((box[1]+box[3]) / 2)

            fc  = fused_centers.get(int(tid)) if fused_centers else None
            fcx = int(fc[0]) if fc else None
            fcy = int(fc[1]) if fc else None

            hcx = hcy = None
            if head_centers:
                best_d = float("inf")
                for (hx, hy) in head_centers:
                    d = math.hypot(hx - bcx, hy - bcy)
                    if d < best_d:
                        best_d = d; hcx = int(hx); hcy = int(hy)

            for gate in self.gates:
                gate.update_multi(
                    int(tid),
                    body_cx=bcx,  body_cy=bcy,
                    head_cx=hcx,  head_cy=hcy,
                    fused_cx=fcx, fused_cy=fcy,
                    cross_dist=cross_d,
                )

    @property
    def total_entry(self) -> int:
        return sum(g.entry for g in self.gates)

    @property
    def total_exit(self) -> int:
        return sum(g.exit for g in self.gates)

    @property
    def draw_mode(self) -> bool:
        return self._draw_mode


# ══════════════════════════════════════════════════════════════
#  AUDIO PLAYER
# ══════════════════════════════════════════════════════════════
class NullAudioPlayer:
    _vol: float = 1.0
    def pause(self):               pass
    def resume(self):              pass
    def seek(self, sec: float, resume: bool = False): pass
    def set_volume(self, v):       self._vol = max(0.0, min(1.0, float(v)))
    def start(self):               pass
    def stop(self):                pass


class AudioPlayerThread(threading.Thread):
    """
    ★ FIX v2: 
      - _find_ffplay() searches PATH + common Windows install paths
      - -volume flag added so V/C keys actually change volume
      - set_volume() restarts ffplay immediately at new volume
    """
    def __init__(self, video_path: str, fps: float = 30.0):
        super().__init__(name="AudioPlayer", daemon=True)
        self._path      = video_path
        self._fps       = fps
        self._lock      = threading.Lock()
        self._stop_ev   = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._paused    = False
        self._start_sec = 0.0
        self._play_t    = 0.0
        self._vol       = 1.0
        self._ffplay    = self._find_ffplay()          # ★ FIX
        self._available = self._ffplay is not None
        if not self._available:
            logger.warning("Audio: ffplay not found. Install ffmpeg to enable audio. Download: https://ffmpeg.org or run: winget install ffmpeg")

    @staticmethod
    def _find_ffplay() -> Optional[str]:
        """Search PATH then common Windows install locations."""
        found = shutil.which("ffplay")
        if found:
            return found
        # Common Windows locations
        for p in [
            r"C:\ffmpeg\bin\ffplay.exe",
            r"C:\Program Files\ffmpeg\bin\ffplay.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffplay.exe",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffplay.exe"),
        ]:
            if os.path.isfile(p):
                return p
        return None

    def set_volume(self, v: float):
        new_vol = max(0.0, min(1.0, float(v)))
        self._vol = new_vol
        # ★ FIX: restart with new volume so change takes effect immediately
        with self._lock:
            if not self._paused and self._proc is not None:
                pos = self._current_sec()
                self._kill_proc()
                self._launch(pos)

    def pause(self):
        with self._lock:
            if self._paused:
                return
            self._start_sec = self._current_sec()
            self._paused    = True
            self._kill_proc()

    def resume(self):
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            self._launch(self._start_sec)

    def seek(self, sec: float, resume: bool = False):
        """Jump to `sec`. With resume=True this also lifts a pause.

        ★ FIX: the P key called seek() to restart audio after a pause, but seek()
        bailed out while _paused was True and resume() was never called from
        anywhere — so audio died permanently at the first pause.
        """
        with self._lock:
            self._start_sec = max(0.0, sec)
            if resume:
                self._paused = False
            if not self._paused:
                self._kill_proc()
                self._launch(self._start_sec)

    def stop(self):
        self._stop_ev.set()
        with self._lock:
            self._kill_proc()

    def _current_sec(self) -> float:
        if self._play_t <= 0:
            return self._start_sec
        elapsed = time.perf_counter() - self._play_t
        return self._start_sec + elapsed

    def _kill_proc(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._play_t = 0.0

    def _launch(self, start_sec: float):
        if not self._available:
            return
        vol_int = max(0, min(100, int(self._vol * 100)))
        try:
            cmd = [
                self._ffplay,          # ★ FIX: use found path, not bare "ffplay"
                "-nodisp", "-autoexit",
                "-loglevel", "quiet",
                "-ss", f"{start_sec:.2f}",
                "-volume", str(vol_int),  # ★ FIX: actual volume control
                self._path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._play_t = time.perf_counter()
            logger.info(f"Audio: playing {start_sec:.1f}s vol={vol_int}% PID={self._proc.pid}")
        except Exception as e:
            logger.warning(f"Audio: ffplay launch failed — {e}")
            self._proc = None

    def run(self):
        if not self._available:
            return  # warning already logged in __init__
        with self._lock:
            self._launch(0.0)

        while not self._stop_ev.is_set():
            time.sleep(0.5)
            with self._lock:
                if self._paused or self._proc is None:
                    continue
                ret = self._proc.poll()
                if ret is not None:
                    self._start_sec = 0.0
                    self._launch(0.0)

        with self._lock:
            self._kill_proc()


# ══════════════════════════════════════════════════════════════
#  DETECTION RESULT
# ══════════════════════════════════════════════════════════════
class DetectionResult:
    def __init__(self):
        self._lock          = threading.Lock()
        self.body_boxes     = np.zeros((0, 4))
        self.body_ids       = None
        self.body_confs     = None
        self.body_count     = 0
        self.head_centers   = []
        self.head_count     = 0
        self.orphan_heads   = []
        self.raw_fused      = 0
        self.final_count    = 0
        self.density_est    = 0
        self.avg_conf       = 0.0
        self.adaptive_conf  = 0.25
        self.entry_count    = 0
        self.exit_count     = 0
        self.forecasts      = {5: 0, 30: 0, 60: 0, 300: 0}
        self.crowd_thr      = 15
        self.status         = "Normal"
        self.anomalies      = []
        self.anomaly_score  = 0.0
        self.drift_score    = 0.0
        self.cnn_count      = 0
        self.online_loss    = 0.0
        self.online_updates = 0
        self.mae_cnn        = 0.0
        self.mae_lstm       = 0.0
        self.inference_ms   = 0.0
        self.display_fps    = 0.0
        self.frame_id       = 0
        self.accurate_count = 0
        self.smooth_boxes: Dict[int, np.ndarray] = {}
        # Downscaled grayscale of the last analysed frame. AnalyticsThread needs
        # a real image for optical-flow anomaly detection and for the brightness
        # term of drift detection; it used to be handed a constant.
        self.scene_gray: Optional[np.ndarray] = None

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            s = DetectionResult.__new__(DetectionResult)
            for attr in vars(self):
                if attr.startswith("_"):
                    continue
                val = getattr(self, attr)
                if isinstance(val, np.ndarray): val = val.copy()
                elif isinstance(val, list):     val = list(val)
                elif isinstance(val, dict):     val = dict(val)
                setattr(s, attr, val)
            s._lock = threading.Lock()
            return s


# ══════════════════════════════════════════════════════════════
#  PAN+ZOOM VIEW
# ══════════════════════════════════════════════════════════════
class PanZoomView:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.zoom   = cfg.DEFAULT_ZOOM
        self._pan_x = 0
        self._pan_y = 0
        self._fw    = 1920
        self._fh    = 1080

    def zoom_in(self):
        self.zoom = min(self.cfg.MAX_ZOOM,
                        round(self.zoom + self.cfg.ZOOM_STEP, 2))
        self._clamp()

    def zoom_out(self):
        self.zoom = max(self.cfg.MIN_ZOOM,
                        round(self.zoom - self.cfg.ZOOM_STEP, 2))
        if self.zoom <= 1.0:
            self._pan_x = self._pan_y = 0
        else:
            self._clamp()

    def pan(self, dx: int, dy: int):
        if self.zoom <= 1.0:
            return
        self._pan_x += dx
        self._pan_y += dy
        self._clamp()

    def reset(self):
        self.zoom = self.cfg.DEFAULT_ZOOM
        self._pan_x = self._pan_y = 0

    def _clamp(self):
        if self.zoom <= 1.0:
            self._pan_x = self._pan_y = 0
            return
        cw = self._fw / self.zoom
        ch = self._fh / self.zoom
        mx = max(0, int((self._fw - cw) / 2))
        my = max(0, int((self._fh - ch) / 2))
        self._pan_x = max(-mx, min(mx, self._pan_x))
        self._pan_y = max(-my, min(my, self._pan_y))

    def _crop_rect(self, w: int, h: int) -> Tuple[int, int, int, int]:
        cw = max(1, int(w / self.zoom))
        ch = max(1, int(h / self.zoom))
        cx = w // 2 + self._pan_x
        cy = h // 2 + self._pan_y
        x1 = max(0, min(cx - cw//2, w - cw))
        y1 = max(0, min(cy - ch//2, h - ch))
        return x1, y1, cw, ch

    def apply(self, frame: np.ndarray, enhance: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        self._fw = w; self._fh = h
        self._clamp()

        if self.zoom == 1.0:
            if enhance:
                return self._enhance_pass(frame, strength=0.5)
            return frame

        if self.zoom > 1.0:
            x1, y1, cw, ch = self._crop_rect(w, h)
            roi = frame[y1:y1+ch, x1:x1+cw]
            up = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LANCZOS4)

            if self.zoom >= 2.0:
                d     = 7 if self.zoom >= 4.0 else 5
                sigma = 55
                up = cv2.bilateralFilter(up, d=d, sigmaColor=sigma, sigmaSpace=sigma)

            us_str   = min(0.65, (self.zoom - 1.0) * 0.28)
            us_ksize = 5 if self.zoom >= 3.0 else 3
            up = self._unsharp(up, strength=us_str, ksize=us_ksize)

            if enhance:
                up = self._enhance_pass(up,
                    strength=min(1.2, 0.5 + (self.zoom - 1.0) * 0.25))
            return up

        nw = max(1, int(w * self.zoom))
        nh = max(1, int(h * self.zoom))
        small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        ox = (w - nw) // 2; oy = (h - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = small
        return canvas

    @staticmethod
    def _unsharp(img: np.ndarray, strength: float, ksize: int = 3) -> np.ndarray:
        if strength < 0.05:
            return img
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)

    @staticmethod
    def _enhance_pass(img: np.ndarray, strength: float = 0.8) -> np.ndarray:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip  = min(3.5, 1.6 + strength * 1.1)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l     = clahe.apply(l)
            img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except Exception:
            pass
        try:
            us_str = min(0.55, strength * 0.60)
            img    = PanZoomView._unsharp(img, strength=us_str, ksize=3)
        except Exception:
            pass
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1.0 + strength * 0.12), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        except Exception:
            pass
        return img


# ══════════════════════════════════════════════════════════════
#  SLICED INFERENCE
# ══════════════════════════════════════════════════════════════
class SlicedInference:
    def __init__(self, rows: int = 2, cols: int = 2, overlap: float = 0.20):
        self.rows    = rows
        self.cols    = cols
        self.overlap = overlap

    def detect(self, model, frame: np.ndarray,
               conf: float, iou: float, device: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        step_y = int(h / self.rows * (1 - self.overlap / (self.rows+1)))
        step_x = int(w / self.cols * (1 - self.overlap / (self.cols+1)))
        tile_h  = int(h / self.rows * (1 + self.overlap))
        tile_w  = int(w / self.cols * (1 + self.overlap))

        all_boxes: List[np.ndarray] = []
        all_confs: List[np.ndarray] = []

        for r in range(self.rows):
            for c in range(self.cols):
                y1 = min(r * step_y, h - tile_h)
                x1 = min(c * step_x, w - tile_w)
                y2 = min(y1 + tile_h, h)
                x2 = min(x1 + tile_w, w)
                tile = frame[y1:y2, x1:x2]
                res = model(tile, classes=[0], conf=conf, iou=iou, verbose=False)
                if res[0].boxes is not None and len(res[0].boxes) > 0:
                    boxes = res[0].boxes.xyxy.cpu().numpy().copy()
                    confs = res[0].boxes.conf.cpu().numpy()
                    boxes[:, 0] += x1; boxes[:, 2] += x1
                    boxes[:, 1] += y1; boxes[:, 3] += y1
                    all_boxes.append(boxes)
                    all_confs.append(confs)

        if not all_boxes:
            return np.zeros((0, 4)), np.zeros((0,))

        merged_b = np.vstack(all_boxes)
        merged_c = np.concatenate(all_confs)
        keep = self._nms(merged_b, merged_c, iou)
        return merged_b[keep], merged_c[keep]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
        """Non-max suppression over xyxy boxes.

        ★ FIX (two bugs, both silently deleting real people):
          1. This fed xyxy boxes to cv2.dnn.NMSBoxes, which interprets its input
             as [x, y, w, h]. Every box was treated as far larger than it is, so
             detections that do not overlap at all suppressed each other.
             Measured: 20 scattered people -> 15 kept instead of 19.
          2. score_threshold=scores.min() told NMS to drop everything at or below
             the weakest score, so the least-confident detection in the frame —
             usually the distant or partly occluded person we most want — was
             discarded every single frame.
        torchvision.ops.nms takes xyxy directly and runs on the GPU.
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)
        b = torch.as_tensor(np.ascontiguousarray(boxes), dtype=torch.float32)
        s = torch.as_tensor(np.ascontiguousarray(scores), dtype=torch.float32)
        keep = torchvision.ops.nms(b, s, float(iou_thr))
        return keep.cpu().numpy().astype(int)


# ══════════════════════════════════════════════════════════════
#  ADAPTIVE CONFIDENCE
# ══════════════════════════════════════════════════════════════
class AdaptiveConfidence:
    def __init__(self, base: float = 0.25):
        self.base    = base
        self.history = deque(maxlen=30)
        self.current = base

    def update(self, count: int, frame_area: int) -> float:
        self.history.append(count)
        avg     = float(np.mean(self.history)) if self.history else count
        density = avg / max(frame_area / 10000, 1)
        if density > 2.0:
            self.current = max(0.12, self.base - 0.13)
        elif density > 1.0:
            self.current = max(0.18, self.base - 0.07)
        else:
            self.current = self.base
        return self.current


# ══════════════════════════════════════════════════════════════
#  DENSITY ESTIMATOR
# ══════════════════════════════════════════════════════════════
class DensityEstimator:
    def __init__(self, crowd_thr: int = 15):
        self.crowd_thr = crowd_thr
        self._history  = deque(maxlen=10)

    def estimate(self, gray: np.ndarray, yolo_count: int) -> int:
        # ★ FIX: old formula gave 173 with only 7 people (supermarket shelves).
        # New: cap rough estimate at yolo_count*2, only blend for truly dense crowds.
        edges  = cv2.Canny(gray, 60, 180)
        ratio  = float(edges.sum()) / (gray.size * 255)
        h, w   = gray.shape
        rough  = min(int(ratio * h * w / 250), max(1, yolo_count * 2))
        self._history.append(rough)
        smooth = int(np.mean(self._history))
        # Only blend when YOLO itself says crowd is very large (truly dense)
        if yolo_count >= self.crowd_thr * 2 and smooth > yolo_count * 1.5:
            return max(yolo_count, int(yolo_count * 0.75 + smooth * 0.25))
        return yolo_count


# ══════════════════════════════════════════════════════════════
#  DETECTION FUSION
# ══════════════════════════════════════════════════════════════
class DetectionFusion:
    def __init__(self, dist_thresh: int = 55):
        self.dist_thresh = dist_thresh

    def fuse(self, body_boxes: np.ndarray, body_confs: Optional[np.ndarray],
             head_centers: List[Tuple[int,int]], head_count: int,
             body_count: int) -> Tuple[int, List[Tuple[int,int]], float]:
        orphans:  List[Tuple[int,int]] = []
        avg_conf = float(np.mean(body_confs)) if (
            body_confs is not None and len(body_confs) > 0) else 0.5

        if len(body_boxes) == 0:
            return head_count, list(head_centers), avg_conf

        for (hx, hy) in head_centers:
            matched = False
            for box in body_boxes:
                x1, y1, x2, y2 = box
                bcx = (x1 + x2) / 2
                bcy = (y1 + y2) / 2
                dist   = math.hypot(hx - bcx, hy - bcy)
                in_box = (x1-10 <= hx <= x2+10) and (y1-10 <= hy <= y1 + (y2-y1)*0.45)
                if dist < self.dist_thresh or in_box:
                    matched = True
                    break
            if not matched:
                orphans.append((hx, hy))

        base_final = body_count + len(orphans)
        if avg_conf < 0.30:
            final = max(base_final, head_count)
        else:
            final = base_final

        return final, orphans, avg_conf


# ══════════════════════════════════════════════════════════════
#  HEAD DETECTOR
# ══════════════════════════════════════════════════════════════
class HeadDetector:
    HEAD_KPS = [0, 1, 2, 3, 4]

    class _TrackedHead:
        __slots__ = ("x","y","vx","vy","ttl","conf","alpha")
        def __init__(self, x, y, conf=1.0, alpha=0.45):
            self.x  = float(x); self.y  = float(y)
            self.vx = 0.0;      self.vy = 0.0
            self.ttl  = 0
            self.conf = conf
            self.alpha = alpha

        def predict(self):
            return self.x + self.vx * 0.8, self.y + self.vy * 0.8

        def update(self, nx, ny):
            a = self.alpha
            new_vx = nx - self.x
            new_vy = ny - self.y
            self.vx = 0.5 * self.vx + 0.5 * new_vx
            self.vy = 0.5 * self.vy + 0.5 * new_vy
            self.x = a * nx + (1 - a) * self.x
            self.y = a * ny + (1 - a) * self.y
            self.ttl = 0

        def center(self) -> Tuple[int, int]:
            return int(self.x), int(self.y)

    def __init__(self, model_path: str, device: str,
                 conf: float = 0.18, iou: float = 0.30,
                 persist_frames: int = 6):
        self.model          = YOLO(model_path).to(device)
        self.conf           = conf
        self.iou            = iou
        self.persist_frames = persist_frames
        self._tracks: List["HeadDetector._TrackedHead"] = []
        self._MATCH_DIST   = 50
        self._MIN_CONF_GHOST = 0.15

    def detect(self, frame_bgr: np.ndarray,
               use_tta: bool = False) -> Tuple[List[Tuple[int,int]], int]:
        raw = self._detect_raw(frame_bgr)
        if use_tta:
            w   = frame_bgr.shape[1]
            raw2= self._detect_raw(cv2.flip(frame_bgr, 1))
            raw2= [(w - x, y, c) for (x, y, c) in raw2]
            raw = self._dedup_with_conf(raw + raw2)

        matched_track_ids = set()
        unmatched_dets    = []

        for (dx, dy, dc) in raw:
            best_tid  = -1
            best_dist = self._MATCH_DIST * (1.5 if len(self._tracks) > 20 else 1.0)

            for i, t in enumerate(self._tracks):
                px, py = t.predict()
                dist = math.hypot(dx - px, dy - py)
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = i

            if best_tid >= 0:
                self._tracks[best_tid].update(dx, dy)
                self._tracks[best_tid].conf = dc
                matched_track_ids.add(best_tid)
            else:
                unmatched_dets.append((dx, dy, dc))

        new_tracks = []
        for i, t in enumerate(self._tracks):
            if i in matched_track_ids:
                new_tracks.append(t)
            else:
                t.ttl += 1
                if t.ttl <= self.persist_frames:
                    t.x += t.vx * 0.5
                    t.y += t.vy * 0.5
                    t.vx *= 0.5; t.vy *= 0.5
                    t.conf *= 0.85
                    if t.conf >= self._MIN_CONF_GHOST:
                        new_tracks.append(t)

        for (dx, dy, dc) in unmatched_dets:
            nh = HeadDetector._TrackedHead(dx, dy, conf=dc, alpha=0.45)
            new_tracks.append(nh)

        self._tracks = new_tracks
        centers = [t.center() for t in self._tracks]
        return centers, len(centers)

    def _detect_raw(self, frame: np.ndarray) -> List[Tuple[int,int,float]]:
        res = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        out: List[Tuple[int,int,float]] = []
        if res[0].keypoints is None or res[0].keypoints.xy is None:
            return out
        kps_all = res[0].keypoints.xy.cpu().numpy()
        kps_c   = (res[0].keypoints.conf.cpu().numpy()
                   if res[0].keypoints.conf is not None else None)
        person_c = (res[0].boxes.conf.cpu().numpy()
                    if res[0].boxes is not None else None)
        for p_idx, kps in enumerate(kps_all):
            pts: List[Tuple[float,float,float]] = []
            for ki in self.HEAD_KPS:
                x, y = kps[ki]
                if x < 1 and y < 1:
                    continue
                kp_conf = kps_c[p_idx][ki] if kps_c is not None else 1.0
                if kp_conf < 0.18:
                    continue
                pts.append((float(x), float(y), float(kp_conf)))
            if pts:
                cx  = float(np.mean([p[0] for p in pts]))
                cy  = float(np.mean([p[1] for p in pts]))
                conf= float(np.mean([p[2] for p in pts]))
                if person_c is not None and p_idx < len(person_c):
                    conf = (conf + float(person_c[p_idx])) / 2.0
                out.append((int(cx), int(cy), conf))
        return out

    @staticmethod
    def _dedup_with_conf(dets: List[Tuple[int,int,float]],
                          dist: int = 22) -> List[Tuple[int,int,float]]:
        dets = sorted(dets, key=lambda d: d[2], reverse=True)
        out: List[Tuple[int,int,float]] = []
        for (x, y, c) in dets:
            if not any(math.hypot(x-ox, y-oy) < dist for (ox,oy,_) in out):
                out.append((x, y, c))
        return out


# ══════════════════════════════════════════════════════════════
#  CNN REGRESSOR
# ══════════════════════════════════════════════════════════════
class CrowdCNN(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        base          = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        # ★ FIX: the head used to end in nn.ReLU(). A count regressor whose final
        # activation is ReLU has zero gradient everywhere it predicts 0, so once
        # it lands there it cannot climb back out. Softplus keeps the output
        # non-negative while staying differentiable everywhere.
        # Measured on the project's own replay buffer (82 samples, label mean
        # 3.13): MAE 0.30 with the old head vs 0.26 with this one, and this one
        # converged in a third of the steps.
        self.head     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,  64), nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(64,    1), nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)))


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER FORECASTER
# ══════════════════════════════════════════════════════════════
class TransformerForecaster(nn.Module):
    def __init__(self, seq_len: int = 60, n_ahead: int = 4,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(1, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head        = nn.Linear(d_model, n_ahead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transformer(self.embed(x))[:, -1, :])


class CrowdForecaster:
    """Forecast occupancy N seconds ahead from a 1 Hz occupancy history.

    ★ REWRITTEN. The previous implementation could not forecast anything:

      1. It needed seq_len + max(steps) = 360 samples before it would predict at
         all, and until then it returned the *current* count verbatim. At the
         measured 5-10 detections/sec that is the first 40-70 seconds of every
         session filled with fake numbers.
      2. Its training target was norm[min(len-1, len-seq_len+s-1)]. For
         seq_len=60 and s=300 that index runs past the end of the array and
         clamps to len-1, so all four horizons were trained against the same
         value — they learned to predict "the current count".
      3. `steps` are documented as seconds but history was appended once per
         analytics tick, not once per second, so a "300s" horizon was really
         about 30-60 seconds.

    Every row of every session log showed p5 == p30 == p60 == p300 as a result.

    This version samples at a true 1 Hz, fits a robust (Theil-Sen style) linear
    trend over a recent window, and extrapolates with damping so a short burst
    does not project into an implausible number. It is exact about what it does
    not know: with less than MIN_FIT seconds of history it reports the current
    level rather than inventing a slope.
    """

    MIN_FIT     = 8       # seconds of history before a trend is meaningful
    WINDOW_SEC  = 90      # lookback used for the fit
    DAMPING     = 0.55    # how fast the trend decays over the forecast horizon

    def __init__(self, steps: List[int], device: str):
        self.steps      = steps
        self.device     = device
        self.history: deque = deque(maxlen=self.WINDOW_SEC)   # (t_seconds, count)
        self.last_preds = {s: 0 for s in steps}
        self._last_t    = -1.0

    def update(self, count: int) -> Dict[int, int]:
        now = time.monotonic()
        # 1 Hz sampling — independent of how fast the analytics thread ticks
        if self._last_t < 0 or now - self._last_t >= 1.0:
            self.history.append((now, float(count)))
            self._last_t = now

        if len(self.history) < self.MIN_FIT:
            for s in self.steps:
                self.last_preds[s] = int(count)
            return self.last_preds

        t = np.array([p[0] for p in self.history], dtype=np.float64)
        y = np.array([p[1] for p in self.history], dtype=np.float64)
        t = t - t[-1]                      # seconds relative to now (<= 0)

        slope = self._robust_slope(t, y)
        level = float(np.median(y[-self.MIN_FIT:]))

        for s in self.steps:
            # Damped extrapolation: an effective horizon shorter than the
            # nominal one, so a 5-minute projection is not just 60x a 5-second
            # trend. Converges to `level + slope * (1/DAMPING)` as s grows.
            eff  = (1.0 - math.exp(-self.DAMPING * s / 60.0)) * (60.0 / self.DAMPING)
            pred = level + slope * eff
            self.last_preds[s] = max(0, int(round(pred)))
        return self.last_preds

    @staticmethod
    def _robust_slope(t: np.ndarray, y: np.ndarray) -> float:
        """Median of pairwise slopes — resistant to the odd dropped detection."""
        n = len(t)
        if n < 2:
            return 0.0
        if n > 40:                        # subsample so this stays O(1)-ish
            idx = np.linspace(0, n - 1, 40).astype(int)
            t, y = t[idx], y[idx]
            n = len(t)
        i, j = np.triu_indices(n, k=1)
        dt = t[j] - t[i]
        ok = np.abs(dt) > 1e-6
        if not ok.any():
            return 0.0
        return float(np.median((y[j][ok] - y[i][ok]) / dt[ok]))


# ══════════════════════════════════════════════════════════════
#  DEEP ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════
class DeepAnomalyDetector:
    def __init__(self, speed_thresh: float, jump_thresh: int):
        self.speed_thresh = speed_thresh
        self.jump_thresh  = jump_thresh
        self.prev_gray    = None
        self._last_flow_t = 0.0
        self.count_hist   = deque(maxlen=20)
        self.flow_hist    = deque(maxlen=20)
        self.alerts: List[str] = []
        self.flow_mag     = 0.0
        self.anomaly_score = 0.0

    # Farneback dense optical flow is CPU-bound and holds the GIL, which starves
    # the detection thread. Running it on every analysed frame pushed inference
    # from ~40 ms to over 700 ms. Crowd motion does not change meaningfully
    # faster than this, so it runs on a fixed cadence instead.
    FLOW_MIN_INTERVAL = 0.2       # seconds — 5 Hz

    def update(self, gray: np.ndarray, count: int) -> Tuple[List[str], float]:
        self.alerts = []
        self.count_hist.append(count)

        now = time.monotonic()
        due = (now - getattr(self, "_last_flow_t", 0.0)) >= self.FLOW_MIN_INTERVAL
        if self.prev_gray is not None and due:
            self._last_flow_t = now
            sc   = cv2.resize(gray, (320, 180))
            sp   = cv2.resize(self.prev_gray, (320, 180))
            flow = cv2.calcOpticalFlowFarneback(
                sp, sc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            self.flow_mag = float(mag.mean())
            self.flow_hist.append(self.flow_mag)

            if self.flow_mag > self.speed_thresh:
                self.alerts.append("RUNNING/PANIC")
            if float(flow[...,1].mean()) < -6:
                self.alerts.append("REVERSE FLOW")
            if len(self.flow_hist) >= 5:
                if float(np.std(list(self.flow_hist)[-5:])) > self.speed_thresh * 0.8:
                    self.alerts.append("CHAOTIC MOTION")

        # Only refresh the reference frame when flow actually ran, so the pair
        # being compared is always one FLOW_MIN_INTERVAL apart.
        if self.prev_gray is None or due:
            self.prev_gray = gray.copy()

        if len(self.count_hist) >= 5:
            d3 = self.count_hist[-1] - self.count_hist[-4]
            d1 = self.count_hist[-1] - self.count_hist[-2]
            if d3 >= self.jump_thresh:
                self.alerts.append(f"DENSITY SURGE +{d3}")
            if d1 < -self.jump_thresh:
                self.alerts.append(f"RAPID EVACUATION {d1}")

        fn = min(1.0, self.flow_mag / max(self.speed_thresh, 1))
        self.anomaly_score = 0.6*fn + 0.4*min(1.0, len(self.alerts)/3.0)
        return self.alerts, self.anomaly_score


# ══════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int, save_path: str = ""):
        self.capacity  = capacity
        self.save_path = save_path
        self.buffer    = deque(maxlen=capacity)
        self._lock     = threading.Lock()
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, "rb") as f:
                    self.buffer = deque(pickle.load(f), maxlen=capacity)
                logger.info(f"Buffer loaded: {len(self.buffer)} samples")
            except Exception as e:
                logger.warning(f"Buffer load failed: {e}")

    def add(self, img_tensor: torch.Tensor, count: float):
        with self._lock:
            self.buffer.append((img_tensor.cpu(), count))

    def sample(self, n: int) -> Optional[Tuple]:
        with self._lock:
            if len(self.buffer) < n:
                return None
            batch = random.sample(list(self.buffer), n)
        imgs   = torch.stack([b[0] for b in batch])
        counts = torch.tensor([[b[1]] for b in batch], dtype=torch.float32)
        return imgs, counts

    def save(self):
        if not self.save_path:
            return
        try:
            with self._lock:
                data = list(self.buffer)
            with open(self.save_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Buffer save failed: {e}")

    def __len__(self) -> int:
        return len(self.buffer)


CNN_TF = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
CNN_TF_AUG = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.25, 0.25, 0.1),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ══════════════════════════════════════════════════════════════
#  ONLINE LEARNER
# ══════════════════════════════════════════════════════════════
class NullOnlineLearner:
    """Stand-in used when ENABLE_ONLINE_LEARNING is off (the default).

    The real learner cost 20.0 -> 7.7-12.8 FPS on the RTX 3050 while its CNN
    column read 0 in every row of every session log, so it is disabled unless
    explicitly switched on. Same interface, no threads, no GPU work.
    """
    def __init__(self, *_args, **_kwargs):
        self.updates   = 0
        self.last_loss = 0.0

    def push_frame(self, *_a, **_kw):  return None
    def inference(self, *_a, **_kw):   return 0
    def stop(self):                    return None


class OnlineLearner:
    def __init__(self, model: CrowdCNN, buffer: ReplayBuffer, cfg: Config,
                 model_lock: Optional[threading.Lock] = None):
        self.model_ref = model
        self.buffer    = buffer
        self.cfg       = cfg
        self.device    = cfg.DEVICE
        # ★ FIX: this lock must be SHARED with AutoRetrainer. They previously
        # held two independent locks over the same weights, so a nightly retrain
        # could mutate the model while inference was reading it.
        self._lock     = model_lock if model_lock is not None else threading.Lock()
        self._stop     = threading.Event()
        self._queue    = deque(maxlen=80)
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=cfg.ONLINE_LR, weight_decay=1e-5)
        self.criterion = nn.HuberLoss()
        self.updates   = 0
        self.last_loss = 0.0
        self._thread   = threading.Thread(target=self._loop,
                                          name="OnlineLearner", daemon=True)
        self._thread.start()

    def push_frame(self, bgr: np.ndarray, count: int, conf: float):
        if conf < self.cfg.BODY_CONF_BASE:
            return
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._queue.append((img, float(count)))

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(0.5)
            if not self._queue:
                continue
            while self._queue:
                img, count = self._queue.popleft()
                self.buffer.add(CNN_TF_AUG(img), count)
            if len(self.buffer) < self.cfg.MIN_BUFFER_TO_TRAIN:
                continue
            batch = self.buffer.sample(self.cfg.ONLINE_BATCH)
            if batch is None:
                continue
            imgs, counts = batch
            imgs = imgs.to(self.device)
            counts = counts.to(self.device)
            with self._lock:
                self.model_ref.train()
                self.optimizer.zero_grad()
                preds = self.model_ref(imgs)
                loss  = self.criterion(preds, counts)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_ref.parameters(), 1.0)
                self.optimizer.step()
                self.model_ref.eval()
            self.updates   += 1
            self.last_loss  = loss.item()
            if self.updates % 200 == 0:
                save_path = self.cfg.model_save_path()
                torch.save(self.model_ref.state_dict(), save_path)
                self.buffer.save()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def inference(self, bgr: np.ndarray) -> int:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        x   = CNN_TF(img).unsqueeze(0).to(self.device)
        with self._lock:
            self.model_ref.eval()
            with torch.no_grad():
                c = self.model_ref(x).item()
        return max(0, int(round(c)))


# ══════════════════════════════════════════════════════════════
#  VIDEO PLAYER THREAD
# ══════════════════════════════════════════════════════════════
class VideoPlayerThread(threading.Thread):
    def __init__(self, cap, fps: float, cfg: Config):
        super().__init__(name="VideoPlayer", daemon=True)
        self.cap          = cap
        self.fps          = fps
        self.cfg          = cfg
        self.speed        = cfg.DEFAULT_SPEED
        self.paused       = False
        self._stop        = threading.Event()
        self._seek_to     = -1
        self._seek_lock   = threading.Lock()
        self.frame_id     = 0
        # frame_id restarts at 0 every time the clip loops, so it cannot be used
        # to bound a batch run. frames_served counts every frame ever delivered.
        self.frames_served = 0
        self.ended         = threading.Event()
        self.loop          = bool(getattr(cfg, "LOOP_VIDEO", True))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.display_queue = queue.Queue(maxsize=2)
        self.detect_queue  = queue.Queue(maxsize=1)

    @property
    def frame_delay(self) -> float:
        return 1.0 / (max(self.speed, 0.01) * self.fps)

    def seek(self, frame_num: int):
        with self._seek_lock:
            self._seek_to = max(0, min(frame_num, self.total_frames - 1))

    def seek_seconds(self, seconds: float):
        self.seek(self.frame_id + int(seconds * self.fps))

    def run(self):
        deadline = time.perf_counter()
        while not self._stop.is_set():
            with self._seek_lock:
                if self._seek_to >= 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                    self.frame_id = self._seek_to
                    self._seek_to = -1
                    deadline = time.perf_counter()

            if self.paused:
                time.sleep(0.015)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.loop:
                    self.ended.set()
                    break
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_id = 0
                deadline = time.perf_counter()
                continue

            self.frame_id += 1
            self.frames_served += 1
            deadline += self.frame_delay
            st = deadline - time.perf_counter()
            if st > 0:
                time.sleep(st)
            else:
                deadline = time.perf_counter()

            try:
                self.display_queue.put_nowait((self.frame_id, frame))
            except queue.Full:
                try: self.display_queue.get_nowait()
                except queue.Empty: pass
                try: self.display_queue.put_nowait((self.frame_id, frame))
                except queue.Full: pass

            if self.frame_id % self.cfg.DETECT_EVERY == 0:
                try:
                    self.detect_queue.put_nowait((self.frame_id, frame))
                except queue.Full:
                    try: self.detect_queue.get_nowait()
                    except queue.Empty: pass
                    try: self.detect_queue.put_nowait((self.frame_id, frame))
                    except queue.Full: pass

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════
#  DETECTION WORKER
#  ★ BUG FIX: cnn_c is now computed BEFORE accurate_count
#    (was previously computed AFTER, causing NameError on frame 1
#     which silently crashed the thread → all stats stayed at 0,
#     B/H/M keys appeared dead because nothing was detected)
# ══════════════════════════════════════════════════════════════
class DetectionWorker(threading.Thread):
    def __init__(self, detect_queue, result: DetectionResult,
                 body_yolo, head_det: HeadDetector,
                 fusion: DetectionFusion,
                 slicer: SlicedInference,
                 density_est: DensityEstimator,
                 adapt_conf: AdaptiveConfidence,
                 gate_mgr: GateManager,
                 learner: OnlineLearner,
                 cfg: Config, frame_h: int,
                 warmup_frame: Optional[np.ndarray] = None,
                 aux_yolo=None):
        super().__init__(name="DetectionWorker", daemon=True)
        self.warmup_frame = warmup_frame
        # ★ Separate model instance for the slicing / TTA passes.
        # Ultralytics keeps ONE predictor (and its tracker state) per YOLO
        # object. Calling model(...) for a slice tile between model.track()
        # calls re-initialises that predictor and disturbs track persistence,
        # which showed up as dropped gate crossings whenever slicing engaged.
        # The auxiliary passes never touch the tracked model.
        self.aux_yolo = aux_yolo if aux_yolo is not None else body_yolo
        self.detect_queue = detect_queue
        self.result       = result
        self.body_yolo    = body_yolo
        self.head_det     = head_det
        self.fusion       = fusion
        self.slicer       = slicer
        self.density_est  = density_est
        self.adapt_conf   = adapt_conf
        self.gate_mgr     = gate_mgr
        self.learner      = learner
        self.cfg          = cfg
        self.frame_h      = frame_h
        self._stop        = threading.Event()
        self._head_cache  = ([], 0)
        self._head_tick   = 0
        self._last_online = time.time()
        self._smooth      = deque(maxlen=5)
        # tracked-only body counts, drives AdaptiveConfidence
        self._prev_body   = deque(maxlen=5)
        self.use_tta      = cfg.USE_TTA
        self._infer_times = deque(maxlen=60)
        self._box_ema: Dict[int, np.ndarray] = {}
        self._BOX_ALPHA = 0.88   # ★ FIX: was 0.60 → faster box following
        # Detection toggle (O key)
        self.detection_enabled = True
        # Center tracker for smooth, center-based tracking
        self.center_tracker = CenterTracker(
            alpha=cfg.TRACK_EMA_ALPHA, max_age=30)
        # Adaptive resolution tracking
        self._fps_window = deque(maxlen=20)
        self._last_had_extras = False
        self._last_fps_check = time.time()
        self._cur_w = cfg.INFER_W
        self._cur_h = cfg.INFER_H

    def run(self):
        self._warm_up()
        while not self._stop.is_set():
            try:
                frame_id, frame = self.detect_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]

            # ── Adaptive resolution based on real FPS ────────
            # If FPS is below target, shrink inference size.
            # ★ FIX: this used to react to the wall-clock rate of the whole loop,
            # including the optional slicing and TTA passes. Slicing costs
            # 83-131 ms, so engaging it dragged the measured FPS under target and
            # the resolution collapsed from 960x540 towards the 320x180 floor —
            # shrinking the image precisely when the scene was dense enough to
            # need the detail, and dropping gate crossings as a result.
            # Only frames where the plain tracked pass ran alone are used to
            # judge the frame rate, and the step-down is gentler than the
            # step-up is (fast to recover, slow to degrade).
            now_t = time.perf_counter()
            if not self._last_had_extras:
                self._fps_window.append(now_t)
                if len(self._fps_window) >= 8:
                    real_fps = (len(self._fps_window) - 1) / max(
                        self._fps_window[-1] - self._fps_window[0], 1e-6)
                    if real_fps < self.cfg.TARGET_FPS * 0.8:
                        self._cur_w = max(self.cfg.INFER_W_MIN,
                                          self._cur_w - 32)
                        self._cur_h = max(self.cfg.INFER_H_MIN,
                                          self._cur_h - 18)
                    elif real_fps > self.cfg.TARGET_FPS * 1.1:
                        self._cur_w = min(self.cfg.INFER_W,
                                          self._cur_w + 32)
                        self._cur_h = min(self.cfg.INFER_H,
                                          self._cur_h + 18)

            small  = cv2.resize(frame, (self._cur_w, self._cur_h))
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sx     = orig_w / self._cur_w
            sy     = orig_h / self._cur_h

            # ── Skip detection if toggled off (O key) ────────
            if not self.detection_enabled:
                # Still update center tracker with no new data
                self.center_tracker.update(np.zeros((0,4)), None)
                time.sleep(0.02)
                continue

            # ★ Drive adaptive confidence from the TRACKED body count, not the
            # fused total. Feeding it the fused count coupled the extra
            # recall passes back into the primary detector: slicing inflated the
            # count, which lowered the confidence threshold, which changed what
            # the tracked pass found on the next frame. That made gate results
            # depend on whether slicing happened to be engaged.
            prev_count = self._prev_body[-1] if self._prev_body else 0
            conf = self.adapt_conf.update(prev_count, orig_w * orig_h)

            # ── BODY DETECTION ───────────────────────────────
            # ★ FIX: the tracked pass ALWAYS runs and is always the source of
            # track IDs. Slicing and TTA used to *replace* it, which returned
            # body_ids=None and silently destroyed gate counting, box EMA and the
            # centre tracker. They are now extra recall layered on top: their
            # boxes are merged in, and any box that does not correspond to a
            # tracked person is carried as an untracked extra (id -1).
            res = self.body_yolo.track(
                small, classes=[0], persist=True, verbose=False,
                tracker=self.cfg.TRACKER,
                conf=conf, iou=self.cfg.BODY_IOU)
            bi = np.zeros((0, 4), dtype=np.float32); bc = None; body_ids = None
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                bi = res[0].boxes.xyxy.cpu().numpy()
                bc = res[0].boxes.conf.cpu().numpy()
                if res[0].boxes.id is not None:
                    body_ids = res[0].boxes.id.cpu().numpy().astype(int)

            body_count = len(bi)
            self._prev_body.append(body_count)   # before extras are merged in

            # ── EXTRA RECALL PASSES (slicing / TTA) ──────────
            # Slicing is expensive (83-131 ms for a 2x2), so it only engages once
            # the scene is actually dense enough to need it.
            extra_boxes: List[np.ndarray] = []
            extra_confs: List[np.ndarray] = []

            want_slice = (self.cfg.USE_SLICED
                          and body_count >= self.cfg.DENSE_FALLBACK_THR)
            if want_slice:
                sb, sc_ = self.slicer.detect(
                    self.aux_yolo, small,
                    conf, self.cfg.BODY_IOU, self.cfg.DEVICE)
                if len(sb):
                    extra_boxes.append(sb)
                    extra_confs.append(sc_ if sc_ is not None
                                       else np.full(len(sb), conf, np.float32))

            if self.use_tta:
                fr2 = cv2.flip(small, 1)
                r2  = self.aux_yolo(fr2, classes=[0], conf=conf,
                                    iou=self.cfg.BODY_IOU, verbose=False)
                if r2[0].boxes is not None and len(r2[0].boxes) > 0:
                    src = r2[0].boxes.xyxy.cpu().numpy()
                    # ★ FIX: the old un-mirror overwrote fb[:,0] and then read it
                    # back to compute fb[:,2], producing boxes with x1 > x2. It
                    # also used the fixed cfg.INFER_W instead of the adaptive
                    # width, so coordinates drifted whenever the resolution
                    # scaled down. Both sides are now derived from the original.
                    fb = src.copy()
                    fb[:, 0] = self._cur_w - src[:, 2]
                    fb[:, 2] = self._cur_w - src[:, 0]
                    extra_boxes.append(fb)
                    extra_confs.append(r2[0].boxes.conf.cpu().numpy())

            self._last_had_extras = bool(extra_boxes)
            if extra_boxes:
                eb = np.vstack(extra_boxes).astype(np.float32)
                ec = np.concatenate(extra_confs).astype(np.float32)
                # Keep only extras that do not already correspond to a tracked box.
                if body_count > 0:
                    novel = ~self._overlaps_any(eb, bi, self.cfg.BODY_IOU)
                    eb, ec = eb[novel], ec[novel]
                if len(eb):
                    keep = SlicedInference._nms(eb, ec, self.cfg.BODY_IOU)
                    eb, ec = eb[keep], ec[keep]
                if len(eb):
                    bi = np.vstack([bi, eb]).astype(np.float32)
                    bc = np.concatenate(
                        [bc if bc is not None else np.zeros(0, np.float32), ec])
                    if body_ids is not None:
                        # untracked extras get id -1 so downstream code can tell
                        # them apart and never feeds them to the gate counter
                        body_ids = np.concatenate(
                            [body_ids, np.full(len(eb), -1, dtype=int)])
                    body_count = len(bi)

            # ── Scale to original coords ──────────────────────
            bo = bi.copy()
            if len(bo) > 0:
                bo[:, 0] *= sx; bo[:, 2] *= sx
                bo[:, 1] *= sy; bo[:, 3] *= sy

            # (The old "re-run with tracker if gates exist" fallback is gone —
            #  the tracked pass above is now unconditional, so there is nothing
            #  left to recover from.)

            # ── BOX EMA SMOOTHING ─────────────────────────────
            if body_ids is not None and len(bo) > 0:
                smoothed_bo = bo.copy().astype(np.float32)
                alpha = self._BOX_ALPHA
                for i, tid in enumerate(body_ids):
                    tid = int(tid)
                    if tid < 0:
                        continue        # untracked extra — nothing to smooth against
                    if tid in self._box_ema:
                        smoothed_bo[i] = (alpha * bo[i] +
                                          (1.0 - alpha) * self._box_ema[tid])
                    self._box_ema[tid] = smoothed_bo[i]
                active_ids = {int(t) for t in body_ids.tolist() if int(t) >= 0}
                for k in list(self._box_ema.keys()):
                    if k not in active_ids:
                        del self._box_ema[k]
                bo = smoothed_bo.astype(bo.dtype)

            # ── HEAD DETECTION — every frame (★ FIX: removed % 2 skip) ──
            self._head_tick += 1
            hc_i, hcnt = self.head_det.detect(small, use_tta=False)
            hc_o = [(int(x*sx), int(y*sy)) for (x, y) in hc_i]
            self._head_cache = (hc_o, hcnt)

            # ── GATE CROSSING ─────────────────────────────────
            # Update center tracker — this is the source of truth
            # for smooth, center-based positions (no box trailing).
            # Untracked extras (id -1) are excluded: they have no identity, so
            # they cannot be followed across a gate line.
            if body_ids is not None and len(bo) > 0:
                tracked = body_ids >= 0
                bo_t, ids_t = bo[tracked], body_ids[tracked]
            else:
                bo_t, ids_t = bo, body_ids
            smooth_centers = self.center_tracker.update(bo_t, ids_t)

            fused_map: Dict[int, Tuple[int,int]] = {}
            if ids_t is not None and len(bo_t) > 0:
                for _i, _tid in enumerate(ids_t):
                    # Use smoothed center if available, else fall back to box center
                    _tid_int = int(_tid)
                    sc = smooth_centers.get(_tid_int)
                    if sc:
                        fused_map[_tid_int] = sc
                    else:
                        fused_map[_tid_int] = (
                            int((bo_t[_i][0]+bo_t[_i][2])/2),
                            int((bo_t[_i][1]+bo_t[_i][3])/2)
                        )

            # Update head offsets in center tracker for prediction
            if ids_t is not None and len(hc_o) > 0:
                for _i, _tid in enumerate(ids_t):
                    sc = smooth_centers.get(int(_tid))
                    if sc:
                        # Find nearest head to this body center
                        best_d = 60; best_h = None
                        for (hx, hy) in hc_o:
                            d = math.hypot(hx - sc[0], hy - sc[1])
                            if d < best_d:
                                best_d = d; best_h = (hx, hy)
                        if best_h:
                            self.center_tracker.update_head_offset(
                                int(_tid), best_h[0], best_h[1])
            self.gate_mgr.update_tracking(
                bo_t, ids_t,
                head_centers=list(hc_o),
                fused_centers=fused_map,
            )

            # ── FUSION ───────────────────────────────────────
            final, orphans, fconf = self.fusion.fuse(
                bo, bc, hc_o, hcnt, body_count)

            # ── DENSITY FALLBACK ─────────────────────────────
            dens = self.density_est.estimate(gray, final)

            # ── TEMPORAL SMOOTHING ────────────────────────────
            self._smooth.append(final)
            smoothed = int(round(np.mean(self._smooth)))

            # ══════════════════════════════════════════════════
            #  ★ FIX: CNN inference BEFORE accurate_count
            #    Previously cnn_c was defined AFTER the block
            #    that used it → NameError → thread crash → all 0
            # ══════════════════════════════════════════════════
            cnn_c = self.learner.inference(small)

            # ── ACCURATE COUNT — 4-signal fusion ─────────────
            # ★ FIX: density weight cut from 10-14% → 3% to stop
            #        inflated ACCURATE counts (e.g. 24 with only 7 bodies).
            if fconf >= 0.45:
                _wb, _wh, _wd, _wc = 0.55, 0.35, 0.03, 0.07
            elif fconf >= 0.30:
                _wb, _wh, _wd, _wc = 0.48, 0.44, 0.03, 0.05
            else:
                _wb, _wh, _wd, _wc = 0.40, 0.53, 0.03, 0.04
            acc_raw = (_wb * body_count + _wh * hcnt +
                       _wd * dens       + _wc * cnn_c)
            accurate = max(int(round(acc_raw)), max(body_count, hcnt))

            # ── ONLINE LEARNING ───────────────────────────────
            if time.time() - self._last_online >= self.cfg.ONLINE_EVERY_SEC:
                avg_c = float(np.mean(bc)) if bc is not None and len(bc) else fconf
                self.learner.push_frame(small, smoothed, avg_c)
                self._last_online = time.time()

            infer_ms = (time.perf_counter() - t0) * 1000
            self._infer_times.append(infer_ms)

            self.result.update(
                frame_id       = frame_id,
                body_boxes     = bo,
                body_ids       = body_ids,
                body_confs     = bc,
                body_count     = body_count,
                head_centers   = hc_o,
                head_count     = hcnt,
                orphan_heads   = orphans,
                raw_fused      = final,
                final_count    = smoothed,
                accurate_count = accurate,
                density_est    = dens,
                avg_conf       = fconf,
                adaptive_conf  = conf,
                cnn_count      = cnn_c,
                online_loss    = self.learner.last_loss,
                online_updates = self.learner.updates,
                inference_ms   = infer_ms,
                # Real frame for the anomaly / drift detectors. Fixed 320x180 so
                # optical flow cost stays constant as the adaptive resolution
                # moves, and copied so analytics never reads a buffer we reuse.
                scene_gray     = cv2.resize(gray, (320, 180)).copy(),
            )

    def _warm_up(self):
        """Pay the one-off CUDA/cuDNN setup cost before the first real frame.

        Warming up from the main thread only got the first live inference from
        10.4 s down to 4.1 s — some of the setup is per-thread, so the rest has
        to be paid here, on the thread that actually runs detection.
        """
        if self.warmup_frame is None:
            return
        try:
            t0 = time.perf_counter()
            w = cv2.resize(self.warmup_frame, (self._cur_w, self._cur_h))
            for _ in range(2):
                self.body_yolo.track(
                    w, classes=[0], persist=False, verbose=False,
                    tracker=self.cfg.TRACKER,
                    conf=self.cfg.BODY_CONF_BASE, iou=self.cfg.BODY_IOU)
                self.head_det.detect(w, use_tta=False)
            self.head_det._tracks.clear()
            self.center_tracker = CenterTracker(
                alpha=self.cfg.TRACK_EMA_ALPHA, max_age=30)
            logger.info("DetectionWorker warm-up: %.1fs",
                        time.perf_counter() - t0)
        except Exception as e:
            logger.warning(f"DetectionWorker warm-up skipped: {e}")

    @staticmethod
    def _overlaps_any(cand: np.ndarray, ref: np.ndarray,
                      iou_thr: float = 0.45) -> np.ndarray:
        """Boolean mask: does each candidate box overlap any reference box?

        Used to keep only the genuinely new detections from the slicing/TTA
        passes, so an already-tracked person is never counted twice.
        """
        if len(cand) == 0:
            return np.zeros(0, dtype=bool)
        if len(ref) == 0:
            return np.zeros(len(cand), dtype=bool)
        c = torch.as_tensor(np.ascontiguousarray(cand), dtype=torch.float32)
        r = torch.as_tensor(np.ascontiguousarray(ref), dtype=torch.float32)
        iou = torchvision.ops.box_iou(c, r)
        return (iou.max(dim=1).values >= float(iou_thr)).cpu().numpy()

    def avg_infer_ms(self) -> float:
        return float(np.mean(self._infer_times)) if self._infer_times else 0.0

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════
#  DRIFT DETECTOR
# ══════════════════════════════════════════════════════════════
class DriftDetector:
    def __init__(self, window: int = 30, threshold: float = 0.40,
                 log_path: str = "drift_events.csv"):
        self.window    = window
        self.threshold = threshold
        self.bh        = deque(maxlen=window*2)
        self.ch        = deque(maxlen=window*2)
        self.drift_active = False
        self.drift_score  = 0.0
        self.drift_events = 0
        exists = os.path.exists(log_path)
        self._f = open(log_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if not exists:
            self._w.writerow(["ts", "type", "score", "detail"])

    def update(self, gray: np.ndarray, count: int) -> bool:
        self.bh.append(float(gray.mean()))
        self.ch.append(count)
        if len(self.bh) < self.window:
            return False
        mid = self.window
        b0 = np.mean(list(self.bh)[:mid]); b1 = np.mean(list(self.bh)[mid:])
        c0 = np.mean(list(self.ch)[:mid]); c1 = np.mean(list(self.ch)[mid:])
        bc = abs(b1-b0)/(b0+1e-5); cc = abs(c1-c0)/(c0+1e-5)
        raw = max(bc, cc)
        self.drift_score = float(raw) if (raw == raw and raw != float("inf")) else 0.0
        if self.drift_score > self.threshold:
            if not self.drift_active:
                self.drift_events += 1
                self._w.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    "brightness" if bc > cc else "density",
                    f"{self.drift_score:.3f}",
                    f"b={bc:.2f} c={cc:.2f}"])
                self._f.flush()
            self.drift_active = True
            return True
        self.drift_active = False
        return False

    def close(self):
        self._f.close()


# ══════════════════════════════════════════════════════════════
#  ADAPTIVE THRESHOLD
# ══════════════════════════════════════════════════════════════
class AdaptiveThreshold:
    PRAYER_WINDOWS = [(4,6),(11,13),(14,16),(17,19),(19,22)]

    def __init__(self, base_crowd: int, base_alert: int):
        self.bc = base_crowd
        self.ba = base_alert
        self.ch = deque(maxlen=300)
        self.is_ramadan = False

    def update(self, count: int) -> Tuple[int, int]:
        self.ch.append(count)
        now = datetime.now(); f = 1.0
        for h0, h1 in self.PRAYER_WINDOWS:
            if h0 <= now.hour < h1:
                f *= 0.80; break
        if now.weekday() == 4: f *= 0.85
        if self.is_ramadan:    f *= 0.75
        if len(self.ch) >= 60:
            trend = (np.mean(list(self.ch)[-10:]) -
                     np.mean(list(self.ch)[-60:-50]))
            if trend > 3: f *= 0.90
        return max(5, int(self.bc*f)), max(10, int(self.ba*f))


# ══════════════════════════════════════════════════════════════
#  FEEDBACK LOOP
# ══════════════════════════════════════════════════════════════
class FeedbackLoop:
    def __init__(self):
        self.yh = deque(maxlen=100)
        self.ch = deque(maxlen=100)
        self.ph = deque(maxlen=100)
        self.mae_cnn  = 0.0
        self.mae_lstm = 0.0

    def update(self, yolo: int, cnn: int, lstm_5s: int):
        self.yh.append(yolo); self.ch.append(cnn)
        if len(self.ph) >= 5:
            self.mae_lstm = abs(self.ph[-5] - yolo)
        self.ph.append(lstm_5s)
        if len(self.yh) >= 5:
            self.mae_cnn = float(np.mean([
                abs(c-y) for c,y in
                zip(list(self.ch)[-20:], list(self.yh)[-20:])]))


# ══════════════════════════════════════════════════════════════
#  CSV LOGGER
# ══════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path: str):
        exists = os.path.exists(path)
        self._file   = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not exists:
            self._writer.writerow([
                "ts", "body", "head", "fused", "smooth", "cnn", "dens",
                "orphans", "gate_entry", "gate_exit", "status",
                "p5", "p30", "p60", "p300",
                "drift", "anomaly", "anom_score",
                "thr", "conf", "loss", "ms",
            ])

    def write(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ══════════════════════════════════════════════════════════════
#  PEAK TIME TRACKER
# ══════════════════════════════════════════════════════════════
class PeakTimeTracker:
    def __init__(self):
        self.sc: Dict[int, list] = defaultdict(list)

    def update(self, sec: int, count: int):
        self.sc[sec].append(count)

    def _avg(self, sec: int) -> float:
        v = self.sc.get(sec, [0])
        return sum(v) / len(v)

    def build_report(self, threshold: int, report_path: str,
                     session_stats: dict, cfg: Config,
                     gate_mgr: GateManager):
        if not self.sc:
            return

        max_sec    = max(self.sc.keys())
        all_c      = [v for vals in self.sc.values() for v in vals]
        lines: List[str] = []

        lines.append("=" * 70)
        lines.append("   CROWD MASTER  |  ENTERPRISE SESSION REPORT  v5.0")
        lines.append("=" * 70)
        lines.append(f"  Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Video       : {Path(cfg.VIDEO_PATH).name}")
        lines.append(f"  Duration    : {str(timedelta(seconds=max_sec))}")
        lines.append(f"  Threshold   : {threshold} people")
        lines.append(f"  Device      : {cfg.DEVICE.upper()}")
        lines.append(f"  Body model  : {cfg.BODY_MODEL}")
        lines.append(f"  Head model  : {cfg.HEAD_MODEL}")
        lines.append(f"  Tracker     : {cfg.TRACKER}")
        lines.append(f"  Sliced inf. : {'YES' if cfg.USE_SLICED else 'NO'}")
        lines.append("")

        if gate_mgr.gates:
            lines.append("── GATE / DOOR STATISTICS ─────────────────────────────────────")
            for g in gate_mgr.gates:
                net = g.entry - g.exit
                lines.append(
                    f"  Gate {g.gate_id:2d}  |  "
                    f"Entry: {g.entry:5d}  "
                    f"Exit: {g.exit:5d}  "
                    f"Net: {net:+d}")
            lines.append(f"  {'─'*50}")
            lines.append(f"  TOTAL Entry : {gate_mgr.total_entry}")
            lines.append(f"  TOTAL Exit  : {gate_mgr.total_exit}")
            lines.append(f"  Net inside  : {gate_mgr.total_entry - gate_mgr.total_exit:+d}")
            lines.append("")

        lines.append("── SESSION PERFORMANCE ────────────────────────────────────────")
        lines.append(f"  Frames processed  : {session_stats.get('frames', 0)}")
        lines.append(f"  Avg infer ms      : {session_stats.get('avg_infer_ms', 0):.1f}")
        lines.append(f"  Display FPS (avg) : {session_stats.get('display_fps', 0):.1f}")
        lines.append(f"  Online updates    : {session_stats.get('updates', 0)}")
        lines.append(f"  Drift events      : {session_stats.get('drifts', 0)}")
        lines.append(f"  CNN MAE vs YOLO   : {session_stats.get('mae_cnn', 0):.2f}")
        lines.append(f"  Forecaster MAE 5s : {session_stats.get('mae_lstm', 0):.2f}")
        lines.append("")

        lines.append("── CROWD STATISTICS ───────────────────────────────────────────")
        lines.append(f"  Peak count  : {int(max(all_c))} people")
        lines.append(f"  Average     : {np.mean(all_c):.1f} people")
        lines.append(f"  Median      : {np.median(all_c):.1f} people")
        lines.append(f"  Std dev     : {np.std(all_c):.1f}")
        lines.append(f"  95th pct    : {np.percentile(all_c, 95):.1f}")
        lines.append("")

        lines.append("── CROWDED PERIODS ────────────────────────────────────────────")
        in_c = False; ps = 0; pb: list = []; periods: list = []
        for sec in range(max_sec + 1):
            avg = self._avg(sec)
            if avg >= threshold:
                if not in_c:
                    in_c = True; ps = sec; pb = []
                pb.append(avg)
            else:
                if in_c:
                    periods.append((ps, sec-1, float(np.mean(pb))))
                    in_c = False
        if in_c:
            periods.append((ps, max_sec, float(np.mean(pb))))

        if not periods:
            lines.append("  No crowded periods detected.")
        else:
            for i, (s, e, avg) in enumerate(periods, 1):
                dur = e - s + 1
                lines.append(
                    f"  Period {i:02d} | "
                    f"{str(timedelta(seconds=s))} -> "
                    f"{str(timedelta(seconds=e))} | "
                    f"Avg:{avg:.1f} | Duration:{dur}s")

        lines.append("")
        lines.append("── TOP 5 BUSIEST MOMENTS ──────────────────────────────────────")
        top5 = sorted(self.sc.keys(),
                      key=lambda s: self._avg(s), reverse=True)[:5]
        for rank, sec in enumerate(top5, 1):
            lines.append(
                f"  #{rank}  {str(timedelta(seconds=sec))}  |  "
                f"{self._avg(sec):.0f} people avg")

        lines.append("")
        lines.append("── BUSIEST MINUTE ─────────────────────────────────────────────")
        minute_avgs: Dict[int, list] = defaultdict(list)
        for sec, vals in self.sc.items():
            minute_avgs[sec // 60].extend(vals)
        if minute_avgs:
            bm  = max(minute_avgs, key=lambda m: np.mean(minute_avgs[m]))
            avg = float(np.mean(minute_avgs[bm]))
            lines.append(f"  Minute {bm+1}  ->  avg {avg:.1f} people")

        lines.append("")
        lines.append(f"  Total crowded periods : {len(periods)}")
        lines.append("")

        lines.append("── HOURLY CROWD BREAKDOWN ─────────────────────────────────────")
        hourly: Dict[int,list] = defaultdict(list)
        for sec,vals in self.sc.items():
            hourly[sec//3600].extend(vals)
        for hr in sorted(hourly.keys()):
            avg_h = np.mean(hourly[hr])
            bar_len = int(avg_h / max(int(max(all_c)), 1) * 30)
            bar = "#" * bar_len
            lines.append(f"  Hour {hr:02d}  | {bar:<30} | avg {avg_h:.0f}")

        lines.append("")
        lines.append("── DETECTION QUALITY SUMMARY ──────────────────────────────────")
        lines.append(f"  Avg CNN MAE vs YOLO : {session_stats.get('mae_cnn',0):.2f}")
        lines.append(f"  Forecast MAE (5s)   : {session_stats.get('mae_lstm',0):.2f}")
        avg_ms = session_stats.get('avg_infer_ms', 0)
        fps_est = 1000.0/max(avg_ms,1)
        lines.append(f"  Avg inference ms    : {avg_ms:.1f}  (~{fps_est:.1f} det/s)")
        lines.append("")
        lines.append("=" * 70)

        for line in lines:
            print(line)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Report -> {report_path}")


# ══════════════════════════════════════════════════════════════
#  ANALYTICS THREAD
# ══════════════════════════════════════════════════════════════
class AnalyticsThread(threading.Thread):
    def __init__(self, result: DetectionResult,
                 drift_det: DriftDetector,
                 forecaster: CrowdForecaster,
                 anomaly_det: DeepAnomalyDetector,
                 adapt_thr: AdaptiveThreshold,
                 feedback: FeedbackLoop,
                 csv_log: CSVLogger,
                 peak_track: PeakTimeTracker,
                 gate_mgr: GateManager,
                 cfg: Config, fps: float):
        super().__init__(name="Analytics", daemon=True)
        self.result      = result
        self.drift_det   = drift_det
        self.forecaster  = forecaster
        self.anomaly_det = anomaly_det
        self.adapt_thr   = adapt_thr
        self.feedback    = feedback
        self.csv_log     = csv_log
        self.peak_track  = peak_track
        self.gate_mgr    = gate_mgr
        self.cfg         = cfg
        self.fps         = fps
        self._stop       = threading.Event()
        self._last_fid   = -1
        self._infer_hist = deque(maxlen=100)
        self._fps_hist   = deque(maxlen=30)
        self._last_log_t = 0.0

    def run(self):
        # ★ FIX: this used to pass a constant np.full((100,100),128) to both
        # detectors. Farneback optical flow on an unchanging image is identically
        # zero, so RUNNING/PANIC, REVERSE FLOW and CHAOTIC MOTION could never
        # fire, and DriftDetector's brightness term was always 0. The detection
        # worker now publishes the real downscaled grayscale frame.
        while not self._stop.is_set():
            time.sleep(0.08)
            snap = self.result.snapshot()
            if snap.frame_id == self._last_fid or snap.frame_id is None:
                continue
            self._last_fid = snap.frame_id
            scene_gray = snap.scene_gray
            if scene_gray is None:
                scene_gray = np.full((180, 320), 128, dtype=np.uint8)

            count = snap.final_count or 0
            crowd_thr, _ = self.adapt_thr.update(count)
            forecasts     = self.forecaster.update(count)
            self.feedback.update(count, snap.cnn_count or 0,
                                  forecasts.get(5, 0))
            status = "Crowded" if count >= crowd_thr else "Normal"
            sec    = int((snap.frame_id or 0) / max(self.fps, 1))
            self.peak_track.update(sec, count)

            if snap.inference_ms:
                self._infer_hist.append(snap.inference_ms)
            if snap.display_fps:
                self._fps_hist.append(snap.display_fps)

            drift_result           = self.drift_det.update(scene_gray, count)
            anomalies, anom_score  = self.anomaly_det.update(scene_gray, count)

            # ★ FIX: this was `frame_id % LOG_EVERY == 0`, but analytics only sees
            # the frame ids that actually completed detection, so most multiples
            # of 30 were never observed. Measured gaps in one 60-second session:
            # 21s, 3s, 2s, 2s, 6s, 10s. Logging on a wall-clock interval gives an
            # evenly sampled series that can be charted and compared.
            now_log = time.monotonic()
            if now_log - self._last_log_t >= self.cfg.LOG_EVERY_SEC:
                self._last_log_t = now_log
                self.csv_log.write([
                    datetime.now().strftime("%H:%M:%S"),
                    snap.body_count, snap.head_count,
                    snap.raw_fused, count, snap.cnn_count,
                    snap.density_est, len(snap.orphan_heads),
                    self.gate_mgr.total_entry, self.gate_mgr.total_exit,
                    status,
                    forecasts.get(5,0), forecasts.get(30,0),
                    forecasts.get(60,0), forecasts.get(300,0),
                    int(drift_result), ";".join(anomalies),
                    f"{anom_score:.3f}", crowd_thr,
                    f"{snap.adaptive_conf:.3f}",
                    f"{snap.online_loss:.4f}",
                    f"{snap.inference_ms:.1f}",
                ])

            self.result.update(
                forecasts     = forecasts,
                crowd_thr     = crowd_thr,
                status        = status,
                anomalies     = anomalies,
                anomaly_score = anom_score,
                drift_score   = self.drift_det.drift_score,
                mae_cnn       = self.feedback.mae_cnn,
                mae_lstm      = self.feedback.mae_lstm,
            )

    def avg_infer_ms(self) -> float:
        return float(np.mean(self._infer_hist)) if self._infer_hist else 0.0

    def avg_display_fps(self) -> float:
        return float(np.mean(self._fps_hist)) if self._fps_hist else 0.0

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════
#  AUTO RETRAINER
# ══════════════════════════════════════════════════════════════
class AutoRetrainer(threading.Thread):
    def __init__(self, model: CrowdCNN, buffer: ReplayBuffer, cfg: Config,
                 model_lock: Optional[threading.Lock] = None):
        super().__init__(name="AutoRetrainer", daemon=True)
        self.model  = model
        self.buffer = buffer
        self.cfg    = cfg
        self._stop  = threading.Event()
        self._trigger = threading.Event()
        # ★ FIX: shared with OnlineLearner — see the note there.
        self._lock  = model_lock if model_lock is not None else threading.Lock()
        self.last_retrain  = None
        self.retrain_count = 0
        self.status        = "idle"

    def trigger(self):
        self._trigger.set()

    def run(self):
        while not self._stop.is_set():
            time.sleep(30)
            now  = datetime.now()
            auto = (now.hour == self.cfg.RETRAIN_HOUR and
                    (self.last_retrain is None or
                     self.last_retrain.date() < now.date()))
            if auto or self._trigger.is_set():
                self._trigger.clear()
                self._retrain()

    def _retrain(self):
        if len(self.buffer) < self.cfg.MIN_BUFFER_TO_TRAIN * 4:
            return
        self.status = "retraining"
        batch = self.buffer.sample(min(len(self.buffer), 512))
        if batch is None:
            self.status = "idle"; return
        imgs, counts = batch
        device = self.cfg.DEVICE
        crit   = nn.HuberLoss()
        opt    = optim.AdamW(self.model.parameters(),
                             lr=self.cfg.ONLINE_LR * 5, weight_decay=1e-4)
        sch    = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.RETRAIN_EPOCHS)
        loader = DataLoader(TensorDataset(imgs, counts),
                            batch_size=16, shuffle=True)
        with self._lock:
            for ep in range(1, self.cfg.RETRAIN_EPOCHS + 1):
                self.model.train(); ep_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = crit(self.model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item()
                sch.step()
            self.model.eval()
        save_path = self.cfg.model_save_path()
        torch.save(self.model.state_dict(), save_path)
        self.last_retrain  = datetime.now()
        self.retrain_count += 1
        self.status        = "idle"

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════
#  CROWD HEATMAP
# ══════════════════════════════════════════════════════════════
class CrowdHeatmap:
    def __init__(self, w: int, h: int, decay: float = 0.96, radius: int = 45):
        self.w = w; self.h = h
        self.decay  = decay
        self.radius = radius
        self.heat   = np.zeros((h, w), dtype=np.float32)
        r = radius
        g = cv2.getGaussianKernel(2*r+1, r/2.5)
        self._kernel = (g * g.T).astype(np.float32)
        self._kernel /= self._kernel.max()

    def update(self, centers: List[Tuple[int,int]]):
        self.heat *= self.decay
        r = self.radius
        for (cx, cy) in centers:
            sx, sy = int(cx), int(cy)
            x1 = max(0, sx-r); x2 = min(self.w, sx+r+1)
            y1 = max(0, sy-r); y2 = min(self.h, sy+r+1)
            kx1 = x1-(sx-r); kx2 = kx1+(x2-x1)
            ky1 = y1-(sy-r); ky2 = ky1+(y2-y1)
            if x2 > x1 and y2 > y1:
                self.heat[y1:y2, x1:x2] += self._kernel[ky1:ky2, kx1:kx2]

    def render(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        if self.heat.max() < 0.001:
            return frame
        norm    = np.clip(self.heat / (self.heat.max()+1e-5), 0, 1)
        colored = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_JET)
        mask    = np.stack([(norm > 0.05).astype(np.float32)]*3, axis=-1)
        return (colored.astype(np.float32)*alpha*mask +
                frame.astype(np.float32)*(1-alpha*mask)).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  LOADING SCREEN
# ══════════════════════════════════════════════════════════════
class LoadingScreen:
    def __init__(self, title: str, steps: List[str]):
        self.title   = title
        self.steps   = steps
        self.total   = len(steps)
        self.current = 0
        self._w, self._h = 820, 310
        self._win    = "Loading"
        self._enabled = highgui_available()
        if self._enabled:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, self._w, self._h)
            self._draw(0, "Initializing...")
        else:
            logger.info("Loading: Initializing...")

    def step(self, label: str = ""):
        self.current += 1
        pct = int(self.current / self.total * 100)
        lbl = label or (self.steps[self.current-1]
                        if self.current-1 < len(self.steps) else "")
        self._draw(pct, lbl)
        if self._enabled:
            safe_wait_key(1)

    def _draw(self, pct: int, label: str):
        img = np.full((self._h, self._w, 3), (18,18,18), dtype=np.uint8)
        cv2.putText(img, self.title, (self._w//2-250, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200,200,200), 2, cv2.LINE_AA)
        cv2.putText(img, f"{pct}%", (self._w//2-28, 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,220,255), 2, cv2.LINE_AA)
        bx1,by1,bx2,by2 = 55,153, self._w-55, 188
        cv2.rectangle(img, (bx1,by1), (bx2,by2), (50,50,50), -1)
        fx = bx1 + int((bx2-bx1)*pct/100)
        if fx > bx1:
            bar = np.zeros((by2-by1, fx-bx1, 3), dtype=np.uint8)
            for x in range(fx-bx1):
                t = x / max(fx-bx1-1, 1)
                bar[:,x] = [int(255*(1-t)), int(220*t), 0]
            img[by1:by2, bx1:fx] = bar
        cv2.rectangle(img, (bx1,by1), (bx2,by2), (80,80,80), 1)
        cv2.putText(img, label,
                    (self._w//2 - min(len(label)*7, 370), 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150,150,150), 1, cv2.LINE_AA)
        cv2.putText(img, f"Step {self.current}/{self.total}",
                    (self._w-165, 268), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (100,100,100), 1, cv2.LINE_AA)
        if self._enabled:
            cv2.imshow(self._win, img)
        else:
            logger.info("Loading %d%% - %s", pct, label)

    def done(self):
        self._draw(100, "Ready!")
        if self._enabled:
            safe_wait_key(700)
            cv2.destroyWindow(self._win)


# ══════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════
def start_api(result: DetectionResult, gate_mgr: GateManager, cfg: Config):
    if not cfg.API_ENABLED:
        return
    host = str(getattr(cfg, "API_HOST", "127.0.0.1"))
    key  = str(getattr(cfg, "API_KEY", ""))
    if host not in ("127.0.0.1", "localhost", "::1") and not key:
        logger.error(
            "API refused to start: API_HOST=%s exposes live occupancy data to the "
            "network but no CROWD_MASTER_API_KEY is set. Set a key, or leave "
            "API_HOST at 127.0.0.1.", host)
        return

    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        app = FastAPI(title="Crowd Master API v5")
        origins = list(getattr(cfg, "API_ORIGINS", []) or [])
        if origins:
            app.add_middleware(CORSMiddleware, allow_origins=origins)

        def _auth(x_api_key: Optional[str]):
            if key and x_api_key != key:
                raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

        @app.get("/api/status")
        def status(x_api_key: Optional[str] = Header(default=None)):
            _auth(x_api_key)
            s = result.snapshot()
            return {
                "final_count":   s.final_count,
                "body_count":    s.body_count,
                "head_count":    s.head_count,
                "density_est":   s.density_est,
                "gate_entry":    gate_mgr.total_entry,
                "gate_exit":     gate_mgr.total_exit,
                "gates":         [{"id":g.gate_id,"entry":g.entry,"exit":g.exit}
                                   for g in gate_mgr.gates],
                "status":        s.status,
                "crowd_thr":     s.crowd_thr,
                "forecasts":     s.forecasts,
                "anomalies":     s.anomalies,
                "anomaly_score": s.anomaly_score,
                "drift_score":   s.drift_score,
                "inference_ms":  s.inference_ms,
                "adaptive_conf": s.adaptive_conf,
            }

        @app.get("/api/health")
        def health():
            return {"status": "ok", "device": cfg.DEVICE}

        def _run():
            uvicorn.run(app, host=host, port=cfg.API_PORT, log_level="error")
        threading.Thread(target=_run, daemon=True, name="FastAPI").start()
        logger.info(f"API: http://{host}:{cfg.API_PORT}/api/status"
                    f"{'  (X-API-Key required)' if key else ''}")
    except ImportError:
        logger.info("FastAPI not installed — API disabled.")


# ══════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ══════════════════════════════════════════════════════════════
def draw_boxes(frame: np.ndarray, boxes: np.ndarray,
               ids, confs, crowd_thr: int, cfg: Config):
    n = len(boxes)
    c = (cfg.BOX_COLOR_LOW  if n < crowd_thr//2 else
         cfg.BOX_COLOR_MED  if n < crowd_thr    else
         cfg.BOX_COLOR_HIGH)
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
        tid  = int(ids[i])     if ids   is not None and i < len(ids)   else i
        conf = float(confs[i]) if confs is not None and i < len(confs) else 0.0
        label = f"id:{tid} {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ly = max(th+4, y1)
        cv2.rectangle(frame, (x1,ly-th-4), (x1+tw+4,ly), c, -1)
        cv2.putText(frame, label, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
    return frame


def draw_heads(frame: np.ndarray,
               head_centers: List[Tuple[int,int]],
               orphan_centers: List[Tuple[int,int]],
               cfg: Config):
    r = cfg.HEAD_DOT_RADIUS
    for (hx,hy) in head_centers:
        cv2.circle(frame, (int(hx),int(hy)), r+2, cfg.HEAD_DOT_BORDER, -1)
        cv2.circle(frame, (int(hx),int(hy)), r,   cfg.HEAD_DOT_COLOR,  -1)
    for (hx,hy) in orphan_centers:
        cv2.circle(frame, (int(hx),int(hy)), r+5, (0,255,255), -1)
        cv2.circle(frame, (int(hx),int(hy)), r+3, (0,200,255), -1)
        cv2.putText(frame, "+", (int(hx)-5, int(hy)+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)
    return frame


def draw_zones(frame: np.ndarray, zones_dict: dict,
               counts_dict: dict, fw: int, fh: int):
    COLORS = [(200,200,60),(60,200,120),(60,120,200),(200,60,120),(120,60,200)]
    for i, (name,(x1p,y1p,x2p,y2p)) in enumerate(zones_dict.items()):
        px1=int(x1p*fw); py1=int(y1p*fh)
        px2=int(x2p*fw); py2=int(y2p*fh)
        col = COLORS[i % len(COLORS)]
        ov  = frame.copy()
        cv2.rectangle(ov, (px1,py1), (px2,py2), col, -1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (px1,py1), (px2,py2), col, 1)
        cv2.putText(frame, f"{name}:{counts_dict.get(name,0)}",
                    (px1+4, py1+16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════
#  STATS DASHBOARD
# ══════════════════════════════════════════════════════════════
def draw_dashboard(frame: np.ndarray,
                   snap: DetectionResult,
                   gate_mgr: GateManager,
                   show_stats: bool,
                   retrain_status: str,
                   pz: PanZoomView,
                   speed: float,
                   paused: bool,
                   use_tta: bool,
                   use_sliced: bool,
                   enhance_mode: bool = False):
    h, w = frame.shape[:2]

    hint1 = "Q quit  P pause  +/- speed  V/C volume  ]/[ zoom  WASD/arrows pan  . +5s  , -5s"
    hint2 = "N stats  B boxes  H heads  Z zones  M heatmap  G gate  TAB select  R reverse  ESC desel  X del  I enhance  T TTA  F full"
    cv2.putText(frame, hint1, (10,h-26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,100,100), 1, cv2.LINE_AA)
    cv2.putText(frame, hint2, (10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,100,100), 1, cv2.LINE_AA)

    hud = (f"{'[||] ' if paused else ''}"
           f"Spd:{speed:.1f}x  Zm:{pz.zoom:.1f}x"
           f"  {'[ENH]' if enhance_mode else ''}"
           f"  {'TTA' if use_tta else ''}"
           f"  {'SLI' if use_sliced else ''}")
    (tw,th),_ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    hx = w - tw - 10
    ov = frame.copy()
    cv2.rectangle(ov, (hx-5,5), (hx+tw+5,26), (0,0,0), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    hc = (0,180,255) if not paused else (0,100,255)
    cv2.putText(frame, hud, (hx,21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, hc, 1, cv2.LINE_AA)

    if not show_stats:
        clr = (50,220,80) if snap.status == "Normal" else (40,50,220)
        cv2.putText(frame, f"People: {snap.final_count}  [{snap.status}]",
                    (10,38), cv2.FONT_HERSHEY_SIMPLEX, 0.80, clr, 2, cv2.LINE_AA)
        acc = getattr(snap, "accurate_count", snap.final_count)
        cv2.putText(frame, f"Accurate: {acc}",
                    (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2, cv2.LINE_AA)
        return frame

    PW=240; PH=360; PX=8; PY=8
    panel = frame.copy()
    cv2.rectangle(panel, (PX,PY), (PX+PW,PY+PH), (10,10,10), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    sc = snap.status == "Normal"
    cv2.rectangle(frame, (PX,PY), (PX+PW,PY+PH),
                  (40,160,40) if sc else (40,40,210), 1)

    W2=(225,225,225); GR=(60,215,80); RD=(50,55,215); OR=(40,150,255)
    YL=(30,205,255); CY=(200,200,0); PP=(200,80,200); TL=(165,195,50)
    GY=(150,150,150); CR=(80,130,255)
    sd = snap.drift_score if snap.drift_score == snap.drift_score else 0.0

    FS  = 0.38
    FS2 = 0.44

    def row(txt: str, col: tuple, y: int, sc2=None, th2: int = 1):
        cv2.putText(frame, txt, (PX+7, PY+y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc2 or FS, col, th2, cv2.LINE_AA)

    def sep(y: int):
        cv2.line(frame, (PX+4,PY+y), (PX+PW-4,PY+y), (55,55,55), 1)

    cv2.putText(frame, "[OK] Normal" if sc else "[!!] Crowded",
                (PX+7, PY+16), cv2.FONT_HERSHEY_SIMPLEX,
                FS2, GR if sc else RD, 2, cv2.LINE_AA)
    sep(22)

    acc = getattr(snap, "accurate_count", snap.final_count)
    row(f"Body   : {snap.body_count}",           W2,  36)
    row(f"Heads  : {snap.head_count}",           CR,  50)
    row(f"Extra+ : {len(snap.orphan_heads)}",    CY,  64)
    row(f"Density: {snap.density_est}",          GY,  78)
    sep(85)

    cv2.putText(frame, f"TOTAL  : {snap.final_count}",
                (PX+7, PY+100), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, YL, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (PX+3, PY+106), (PX+PW-3, PY+122), (25,25,25), -1)
    cv2.putText(frame, f"ACCURATE: {acc}",
                (PX+7, PY+119), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0,220,255), 2, cv2.LINE_AA)
    sep(125)

    if gate_mgr.gates:
        row(f"Gate In : {gate_mgr.total_entry}",  GR,  138)
        row(f"Gate Out: {gate_mgr.total_exit}",   OR,  151)
        _net = gate_mgr.total_entry - gate_mgr.total_exit
        row(f"Net Flow: {_net:+d}",  GR if _net >= 0 else RD, 164)
        sep(171); ny = 183
    else:
        row(f"Entry  : {snap.entry_count}",       GR,  138)
        row(f"Exit   : {snap.exit_count}",        OR,  151)
        _net2 = snap.entry_count - snap.exit_count
        row(f"Net    : {_net2:+d}",  GR if _net2>=0 else RD, 164)
        sep(171); ny = 183

    row(f"Thresh : {snap.crowd_thr}",             YL,  ny)
    sep(ny+7)

    row(f"5s     : {snap.forecasts.get(5,0)}",    TL,  ny+19)
    row(f"30s    : {snap.forecasts.get(30,0)}",   TL,  ny+31)
    row(f"60s    : {snap.forecasts.get(60,0)}",   TL,  ny+43)
    row(f"5min   : {snap.forecasts.get(300,0)}",  TL,  ny+55)
    sep(ny+62)

    row(f"Drift  : {sd:.2f}",                     PP,  ny+74)
    row(f"Anomaly: {snap.anomaly_score:.2f}",     OR,  ny+86)
    row(f"InfMs  : {snap.inference_ms:.1f}",      GY,  ny+98)
    row(f"Conf   : {snap.adaptive_conf:.2f}",     GY,  ny+110)
    row(f"Upd    : {snap.online_updates}",        GY,  ny+122)
    row(f"Retrain: {retrain_status}",             GY,  ny+134)
    enh_col = (0,220,255) if enhance_mode else GY
    row(f"Enh    : {'ON [I]' if enhance_mode else 'OFF [I]'}", enh_col, ny+146)
    sep(ny+153)

    bx = PX+7; by = PY+ny+162; blen = PW-16
    cv2.rectangle(frame, (bx,by), (bx+blen,by+5), (40,40,40), -1)
    _fill = int(min(sd,1.0)*blen)
    if _fill > 0:
        _dc = RD if sd>0.4 else OR if sd>0.2 else GR
        cv2.rectangle(frame, (bx,by), (bx+_fill,by+5), _dc, -1)
    cv2.putText(frame, "drift", (bx+blen+3,by+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, GY, 1, cv2.LINE_AA)

    for _i, _alert in enumerate((snap.anomalies or [])[:2]):
        cv2.putText(frame, f"!! {_alert}",
                    (PX+7, PY+PH+14+_i*14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, RD, 2, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════
def generate_pdf_report(peak_track: PeakTimeTracker,
                        cfg: Config,
                        session_stats: dict,
                        gate_mgr: GateManager):
    W, H = 794, 1123
    img  = np.ones((H,W,3), dtype=np.uint8) * 255

    def txt(im, t, x, y, sc=0.5, col=(30,30,30), th=1):
        cv2.putText(im, t, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, th, cv2.LINE_AA)

    def hrule(im, y, col=(180,180,180)):
        cv2.line(im, (40,y), (W-40,y), col, 1)

    cv2.rectangle(img, (0,0), (W,80), (20,20,20), -1)
    txt(img, "CROWD MASTER  |  Enterprise Surveillance  v5.0",
        30, 50, 0.80, (255,255,255), 2)
    txt(img, f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        40, 108, 0.48)
    txt(img, f"Video     : {Path(cfg.VIDEO_PATH).name}", 40, 132, 0.44)
    hrule(img, 148)

    cv2.rectangle(img, (40,162), (W-40,350), (240,240,250), -1)
    cv2.rectangle(img, (40,162), (W-40,350), (180,180,200), 1)
    txt(img, "SESSION DIAGNOSTICS", 55, 190, 0.55, (50,50,150), 2)
    diag = [
        ("Frames",         str(session_stats.get("frames",0))),
        ("Avg infer ms",   f"{session_stats.get('avg_infer_ms',0):.1f}"),
        ("Online updates", str(session_stats.get("updates",0))),
        ("Drift events",   str(session_stats.get("drifts",0))),
        ("CNN MAE",        f"{session_stats.get('mae_cnn',0):.2f}"),
        ("LSTM MAE 5s",    f"{session_stats.get('mae_lstm',0):.2f}"),
        ("Body model",     cfg.BODY_MODEL),
        ("Head model",     cfg.HEAD_MODEL),
        ("Tracker",        cfg.TRACKER),
        ("Sliced inf.",    "YES" if cfg.USE_SLICED else "NO"),
    ]
    for i, (label, value) in enumerate(diag):
        col = 0 if i < 5 else 1
        row = i if i < 5 else i - 5
        xb = 55 + col*380
        yp = 218 + row*26
        txt(img, f"{label}:", xb, yp, 0.43, (80,80,80))
        txt(img, value, xb+195, yp, 0.45, (20,20,120), 1)
    hrule(img, 358)

    cy_off = 370
    if gate_mgr.gates:
        txt(img, "GATE STATISTICS", 40, cy_off, 0.55, (50,50,150), 2)
        cy_off += 28
        for g in gate_mgr.gates:
            txt(img,
                f"Gate {g.gate_id}  Entry:{g.entry}  Exit:{g.exit}  "
                f"Net:{g.entry-g.exit:+d}",
                55, cy_off, 0.45)
            cy_off += 22
        txt(img,
            f"TOTAL  Entry:{gate_mgr.total_entry}  "
            f"Exit:{gate_mgr.total_exit}  "
            f"Net:{gate_mgr.total_entry-gate_mgr.total_exit:+d}",
            55, cy_off, 0.48, (20,20,120), 1)
        cy_off += 30
        hrule(img, cy_off)
        cy_off += 14

    if peak_track.sc:
        txt(img, "PEAK CROWD ANALYSIS", 40, cy_off+8, 0.55, (50,50,150), 2)
        cy_off += 36
        max_sec    = max(peak_track.sc.keys())
        all_c      = [v for vals in peak_track.sc.values() for v in vals]
        peak_val   = int(max(all_c))
        txt(img, f"Peak: {peak_val} people", 55, cy_off, 0.50, (200,0,0))
        cy_off += 24
        txt(img, f"Avg:  {np.mean(all_c):.1f}  Median:{np.median(all_c):.1f}  "
            f"Std:{np.std(all_c):.1f}", 55, cy_off, 0.45)
        cy_off += 28

        cx, cw, ch = 40, W-80, 130
        cv2.rectangle(img, (cx,cy_off), (cx+cw,cy_off+ch), (245,245,245), -1)
        cv2.rectangle(img, (cx,cy_off), (cx+cw,cy_off+ch), (180,180,180), 1)
        secs = sorted(peak_track.sc.keys())
        if len(secs) > 120:
            step = len(secs)//120
            secs = secs[::step][:120]
        if secs and peak_val > 0:
            bw = max(1, (cw-10)//len(secs))
            for bi, sec in enumerate(secs):
                avg = peak_track._avg(sec)
                bh  = int(avg/max(peak_val,1)*(ch-20))
                bx  = cx + 5 + bi*bw
                by2 = cy_off + ch - bh - 5
                rat = avg/max(peak_val,1)
                color = (int(50+200*rat), int(200-200*rat), 50)
                if bh > 0:
                    cv2.rectangle(img, (bx,by2),
                                  (bx+bw-1, cy_off+ch-5), color, -1)
        cy_off += ch + 10

    hrule(img, H-80)
    cv2.rectangle(img, (0,H-60), (W,H), (20,20,20), -1)
    txt(img, "Crowd Master v5.0  |  Enterprise AI Surveillance",
        40, H-25, 0.42, (200,200,200))

    png_path = cfg.PDF_REPORT.replace(".pdf", "_report.png")
    cv2.imwrite(png_path, img)
    logger.info(f"Report PNG: {png_path}")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Image as RLI
        # حساب الأبعاد بحيث تتناسب مع الصفحة بدون overflow
        page_w, page_h = A4
        margin = 40
        max_w = page_w - margin * 2
        max_h = page_h - margin * 2
        # حافظ على نسبة الأبعاد
        import PIL.Image as PILImg
        with PILImg.open(png_path) as pil_img:
            iw, ih = pil_img.size
        ratio = min(max_w / iw, max_h / ih)
        draw_w, draw_h = iw * ratio, ih * ratio
        doc = SimpleDocTemplate(cfg.PDF_REPORT, pagesize=A4,
                                leftMargin=margin, rightMargin=margin,
                                topMargin=margin, bottomMargin=margin)
        doc.build([RLI(png_path, width=draw_w, height=draw_h)])
        logger.info(f"PDF: {cfg.PDF_REPORT}")
    except ImportError:
        logger.info("reportlab not installed — PNG saved.")
    except Exception as e:
        logger.warning(f"PDF error: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN RUN
# ══════════════════════════════════════════════════════════════
WIN_NAME = "Crowd Master v5"


# ══════════════════════════════════════════════════════════════
#  CENTER TRACKER  — EMA smoothing per track ID
#  Converts bounding boxes to center points, applies EMA,
#  and predicts positions when detection is temporarily missing.
#  This eliminates "box trailing" because boxes are updated
#  BEFORE drawing, using the smoothed predicted center.
# ══════════════════════════════════════════════════════════════
class CenterTracker:
    """
    Maintains a per-ID dictionary of smoothed (cx, cy) centers.
    Uses EMA:  new = alpha * detected + (1-alpha) * previous
    alpha=0.70 → fast response (~3 frames to converge) + smooth.

    Also stores last-known head offset from body center so head
    can be predicted even when head detector misses.
    """

    def __init__(self, alpha: float = 0.70, max_age: int = 30):
        self.alpha   = alpha
        self.max_age = max_age          # frames before stale track removed
        self._cx:  Dict[int, float] = {}
        self._cy:  Dict[int, float] = {}
        self._age: Dict[int, int]   = {}  # frames since last detection
        # Head offset from body center: {tid: (dx, dy)}
        self._head_off: Dict[int, Tuple[float, float]] = {}

    def update(self, body_boxes: np.ndarray,
               body_ids: Optional[np.ndarray]) -> Dict[int, Tuple[int, int]]:
        """
        Feed new detections. Returns {tid: (cx, cy)} for ALL active
        tracks (including predicted positions for recently-missed IDs).
        """
        if body_ids is None or len(body_boxes) == 0:
            # Advance age; evict stale
            for tid in list(self._age.keys()):
                self._age[tid] += 1
                if self._age[tid] > self.max_age:
                    self._cx.pop(tid, None)
                    self._cy.pop(tid, None)
                    self._age.pop(tid, None)
                    self._head_off.pop(tid, None)
            return {tid: (int(self._cx[tid]), int(self._cy[tid]))
                    for tid in self._cx}

        seen = set()
        for box, tid in zip(body_boxes, body_ids):
            tid = int(tid)
            detected_cx = float((box[0] + box[2]) / 2)
            detected_cy = float((box[1] + box[3]) / 2)
            if tid in self._cx:
                # EMA blend
                self._cx[tid] = self.alpha * detected_cx + (1 - self.alpha) * self._cx[tid]
                self._cy[tid] = self.alpha * detected_cy + (1 - self.alpha) * self._cy[tid]
            else:
                self._cx[tid] = detected_cx
                self._cy[tid] = detected_cy
            self._age[tid] = 0
            seen.add(tid)

        # Age unseen tracks
        for tid in list(self._age.keys()):
            if tid not in seen:
                self._age[tid] += 1
                if self._age[tid] > self.max_age:
                    self._cx.pop(tid, None)
                    self._cy.pop(tid, None)
                    self._age.pop(tid, None)
                    self._head_off.pop(tid, None)

        return {tid: (int(self._cx[tid]), int(self._cy[tid]))
                for tid in self._cx}

    def update_head_offset(self, tid: int, head_cx: float,
                           head_cy: float):
        """Store head-to-body offset so head can be predicted later."""
        if tid in self._cx:
            dx = head_cx - self._cx[tid]
            dy = head_cy - self._cy[tid]
            if tid in self._head_off:
                alpha = 0.5
                ox, oy = self._head_off[tid]
                self._head_off[tid] = (alpha * dx + (1-alpha) * ox,
                                       alpha * dy + (1-alpha) * oy)
            else:
                self._head_off[tid] = (dx, dy)

    def predict_head(self, tid: int) -> Optional[Tuple[int, int]]:
        """Predict head position from last known body center + offset."""
        if tid not in self._cx or tid not in self._head_off:
            return None
        dx, dy = self._head_off[tid]
        return (int(self._cx[tid] + dx), int(self._cy[tid] + dy))

    def get_center(self, tid: int) -> Optional[Tuple[int, int]]:
        if tid in self._cx:
            return (int(self._cx[tid]), int(self._cy[tid]))
        return None


# ══════════════════════════════════════════════════════════════
#  TKINTER VIDEO LAUNCHER
#  Simple GUI that lets the user pick a video file before
#  the OpenCV window opens. No config editing required.
# ══════════════════════════════════════════════════════════════


def save_snapshot(frame: np.ndarray, cfg: Config, prefix: str = "snapshot") -> Optional[str]:
    try:
        shot_dir = Path(cfg._data_dir) / "snapshots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = shot_dir / f"{prefix}_{ts}.jpg"
        ok = cv2.imwrite(str(out_path), frame)
        if ok:
            logger.info(f"Snapshot saved: {out_path}")
            return str(out_path)
        logger.error(f"Snapshot save failed: {out_path}")
    except Exception as exc:
        logger.error(f"Snapshot save failed: {exc}")
    return None
def draw_controls_overlay(frame: np.ndarray, cfg: Config, player, gate_mgr: GateManager,
                          show_boxes: bool, show_heads: bool, show_zones: bool,
                          show_heatmap: bool, show_stats: bool, enhance_mode: bool,
                          det_worker) -> None:
    h, w = frame.shape[:2]
    panel_w = min(520, max(360, w - 40))
    panel_h = 315
    x1, y1 = 18, 48
    x2, y2 = x1 + panel_w, min(y1 + panel_h, h - 18)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (78, 204, 163), 2)

    def put(text_value, x, y, color=(235, 235, 235), scale=0.47, thick=1):
        cv2.putText(frame, text_value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    put("CONTROLS  (K hide/show)", x1 + 16, y1 + 28, (78, 204, 163), 0.58, 2)
    status = [
        f"Source: {source_label(cfg)}",
        f"Speed: {player.speed:.2f}x",
        f"Detection: {'ON' if getattr(det_worker, 'detection_enabled', True) else 'OFF'}",
        f"Gates: {len(gate_mgr.gates)}   In:{gate_mgr.total_entry} Out:{gate_mgr.total_exit}",
    ]
    yy = y1 + 58
    for line in status:
        put(line, x1 + 16, yy, (210, 220, 230), 0.45)
        yy += 21

    left = [
        "Q quit", "P pause/resume", "C snapshot", "+/- speed",
        "[/] zoom", "WASD / arrows pan", "R reset view", "K controls"
    ]
    right = [
        "B boxes", "H heads", "N stats", "Z zones", "M heatmap",
        "I enhance", "T TTA", "O detection on/off"
    ]
    yy = y1 + 155
    put("Playback", x1 + 16, yy, (255, 210, 120), 0.48, 2)
    put("AI / View", x1 + panel_w//2, yy, (255, 210, 120), 0.48, 2)
    yy += 24
    for i in range(max(len(left), len(right))):
        if i < len(left):
            put(left[i], x1 + 20, yy, (235, 235, 235), 0.43)
        if i < len(right):
            put(right[i], x1 + panel_w//2, yy, (235, 235, 235), 0.43)
        yy += 21

    flags = [
        ("Boxes", show_boxes), ("Heads", show_heads), ("Stats", show_stats),
        ("Zones", show_zones), ("Heat", show_heatmap), ("Enhance", enhance_mode),
    ]
    bx = x1 + 16
    by = y2 - 22
    for name, enabled in flags:
        col = (78, 204, 163) if enabled else (95, 95, 105)
        cv2.circle(frame, (bx, by - 4), 5, col, -1)
        put(name, bx + 10, by, (220, 220, 220), 0.38)
        bx += 78
class VideoLauncher:
    """
    Shows a clean Tkinter window with:
      - Video path entry + Browse button
      - Device indicator (GPU/CPU)
      - FP16 toggle
      - Start button

    Returns the selected video path via self.result after closing.
    Call VideoLauncher.run() — blocks until user clicks Start or closes.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.result = None   # video path chosen by user

    def run(self) -> Optional[str]:
        """Show GUI, return chosen video path or None if cancelled."""
        if not _TK_AVAILABLE:
            logger.warning("tkinter not available — skipping GUI launcher.")
            return cfg.VIDEO_PATH if os.path.exists(cfg.VIDEO_PATH) else None

        root = tk.Tk()
        root.title("Crowd Master v5  —  Select Source")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")

        # ── Styles ──────────────────────────────────────────
        BG     = "#1a1a2e"
        ACCENT = "#4ecca3"
        WHITE  = "#e0e0e0"
        DARK2  = "#16213e"
        BTN_BG = "#0f3460"
        FONT   = ("Segoe UI", 10)
        BOLD   = ("Segoe UI", 11, "bold")

        style = ttk.Style()
        style.theme_use("clam")

        # ── Header ──────────────────────────────────────────
        hdr = tk.Frame(root, bg=ACCENT, height=4)
        hdr.pack(fill="x")

        tk.Label(root, text="🎥  CROWD MASTER v5",
                 font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(16, 2))
        tk.Label(root, text="Enterprise AI Surveillance Platform",
                 font=("Segoe UI", 9), bg=BG, fg="#888").pack(pady=(0, 14))

        # ── Input source ─────────────────────────────────────
        mode_var = tk.StringVar(value=getattr(self.cfg, "INPUT_MODE", "file"))
        mode_frm = tk.Frame(root, bg=BG)
        mode_frm.pack(fill="x", padx=24, pady=4)
        tk.Label(mode_frm, text="Source:", font=FONT,
                 bg=BG, fg=WHITE, width=12, anchor="w").pack(side="left")
        for value, label in (("file", "Video File"), ("webcam", "Webcam"), ("rtsp", "IP / RTSP")):
            tk.Radiobutton(mode_frm, text=label, value=value, variable=mode_var,
                           font=FONT, bg=BG, fg=WHITE, selectcolor=DARK2,
                           activebackground=BG, activeforeground=ACCENT).pack(side="left", padx=(4, 12))

        # ── Video path ──────────────────────────────────────
        frm = tk.Frame(root, bg=BG)
        frm.pack(fill="x", padx=24, pady=4)
        source_label_var = tk.StringVar(value="Video File:")
        tk.Label(frm, textvariable=source_label_var, font=FONT,
                 bg=BG, fg=WHITE, width=12, anchor="w").pack(side="left")

        path_var = tk.StringVar(value=self.cfg.VIDEO_PATH)
        path_entry = tk.Entry(frm, textvariable=path_var,
                              font=FONT, bg=DARK2, fg=WHITE,
                              insertbackground=WHITE, relief="flat",
                              width=46)
        path_entry.pack(side="left", padx=(4, 6))
        cam_var = tk.StringVar(value=str(getattr(self.cfg, "CAMERA_INDEX", 0)))
        rtsp_var = tk.StringVar(value=str(getattr(self.cfg, "CAMERA_URL", "")))

        def browse():
            f = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files",
                     "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                    ("All files", "*.*"),
                ])
            if f:
                path_var.set(f)

        browse_btn = tk.Button(frm, text="Browse...", command=browse,
                               font=FONT, bg=BTN_BG, fg=ACCENT,
                               relief="flat", padx=10, cursor="hand2")
        browse_btn.pack(side="left")

        def refresh_source_fields(*_):
            mode = mode_var.get()
            if mode == "file":
                source_label_var.set("Video File:")
                path_var.set(self.cfg.VIDEO_PATH)
                path_entry.configure(textvariable=path_var, state="normal")
                browse_btn.configure(state="normal")
            elif mode == "webcam":
                source_label_var.set("Camera ID:")
                path_entry.configure(textvariable=cam_var, state="normal")
                browse_btn.configure(state="disabled")
            else:
                source_label_var.set("RTSP URL:")
                path_entry.configure(textvariable=rtsp_var, state="normal")
                browse_btn.configure(state="disabled")

        mode_var.trace_add("write", refresh_source_fields)
        refresh_source_fields()

        # ── Data dir ────────────────────────────────────────
        frm2 = tk.Frame(root, bg=BG)
        frm2.pack(fill="x", padx=24, pady=4)
        tk.Label(frm2, text="Output Dir:", font=FONT,
                 bg=BG, fg=WHITE, width=12, anchor="w").pack(side="left")
        dir_var = tk.StringVar(value=str(self.cfg._data_dir))
        tk.Entry(frm2, textvariable=dir_var,
                 font=FONT, bg=DARK2, fg="#aaa",
                 relief="flat", width=46,
                 state="readonly").pack(side="left", padx=(4, 6))
        def browse_dir():
            d = filedialog.askdirectory(title="Select Output Directory")
            if d: dir_var.set(d)
        tk.Button(frm2, text="Change…", command=browse_dir,
                  font=FONT, bg=BTN_BG, fg=ACCENT,
                  relief="flat", padx=10, cursor="hand2").pack(side="left")

        # ── Device info ─────────────────────────────────────
        sep = tk.Frame(root, bg="#333", height=1)
        sep.pack(fill="x", padx=24, pady=10)

        dev_frm = tk.Frame(root, bg=BG)
        dev_frm.pack(fill="x", padx=24, pady=4)
        dev_lbl = ("🟢 GPU (CUDA)" if self.cfg.DEVICE == "cuda"
                   else "🟡 CPU (no GPU)")
        dev_col = (ACCENT if self.cfg.DEVICE == "cuda" else "#f0c040")
        tk.Label(dev_frm, text=f"Device:  {dev_lbl}",
                 font=FONT, bg=BG, fg=dev_col).pack(side="left")

        # FP16 toggle (only shown for GPU)
        fp16_var = tk.BooleanVar(value=self.cfg.USE_FP16)
        if self.cfg.DEVICE == "cuda":
            tk.Checkbutton(dev_frm, text="FP16 (faster)",
                           variable=fp16_var, font=FONT,
                           bg=BG, fg=WHITE,
                           selectcolor=DARK2,
                           activebackground=BG,
                           activeforeground=ACCENT).pack(side="left", padx=20)

        # ── Key hint ────────────────────────────────────────
        sep2 = tk.Frame(root, bg="#333", height=1)
        sep2.pack(fill="x", padx=24, pady=10)
        hints = ("Q quit  P pause  G gate-draw  TAB select  "
                 "]/[ zoom  WASD pan  I enhance  O detect  +/- speed")
        tk.Label(root, text=hints, font=("Segoe UI", 8),
                 bg=BG, fg="#666", wraplength=520).pack(pady=(0, 6))

        # ── Start / Cancel buttons ───────────────────────────
        btn_frm = tk.Frame(root, bg=BG)
        btn_frm.pack(pady=(4, 18))

        started = {"v": False}

        def on_start():
            mode = mode_var.get()
            p = path_var.get().strip() if mode == "file" else ""
            if mode == "file":
                if not p:
                    messagebox.showerror("Error", "Please select a video file.")
                    return
                if not os.path.exists(p):
                    messagebox.showerror(
                        "File Not Found",
                        f"Video not found:\n{p}\n\nPlease choose a valid file.")
                    return
                self.cfg.VIDEO_PATH = p
            elif mode == "webcam":
                try:
                    self.cfg.CAMERA_INDEX = int(cam_var.get().strip() or "0")
                except ValueError:
                    messagebox.showerror("Error", "Camera ID must be a number, usually 0 or 1.")
                    return
            else:
                self.cfg.CAMERA_URL = rtsp_var.get().strip()
                if not self.cfg.CAMERA_URL:
                    messagebox.showerror("Error", "Please enter the IP camera / RTSP URL.")
                    return
            self.cfg.INPUT_MODE = mode
            self.cfg._data_dir  = Path(dir_var.get())
            self.cfg._data_dir.mkdir(parents=True, exist_ok=True)
            # Update all paths
            self.cfg.LOG_FILE    = str(self.cfg._data_dir / "crowd_log.csv")
            self.cfg.REPORT_FILE = str(self.cfg._data_dir / "crowd_report.txt")
            self.cfg.PDF_REPORT  = str(self.cfg._data_dir / "crowd_report.pdf")
            self.cfg.MODEL_SAVE  = str(self.cfg._data_dir / "best_crowd_model.pth")
            self.cfg.BUFFER_SAVE = str(self.cfg._data_dir / "replay_buffer.pkl")
            self.cfg.DRIFT_LOG   = str(self.cfg._data_dir / "drift_events.csv")
            self.cfg.ONNX_PATH   = str(self.cfg._data_dir / "crowd_model.onnx")
            self.cfg.GATES_FILE  = str(self.cfg._data_dir / "gates.json")
            self.cfg.USE_FP16    = fp16_var.get()
            self.result = source_label(self.cfg)
            started["v"] = True
            root.destroy()

        def on_cancel():
            root.destroy()

        tk.Button(btn_frm, text="  ▶  Start Analysis  ",
                  command=on_start,
                  font=("Segoe UI", 11, "bold"),
                  bg=ACCENT, fg="#000",
                  relief="flat", padx=18, pady=8,
                  cursor="hand2").pack(side="left", padx=8)

        tk.Button(btn_frm, text="  Cancel  ",
                  command=on_cancel,
                  font=FONT,
                  bg="#444", fg=WHITE,
                  relief="flat", padx=12, pady=8,
                  cursor="hand2").pack(side="left", padx=8)

        # Center window on screen
        root.update_idletasks()
        w = root.winfo_width(); h = root.winfo_height()
        sw= root.winfo_screenwidth(); sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        root.mainloop()
        return self.result

def run(cfg: Config):
    load_steps = [
        "Loading CNN model...",
        f"Loading Body Detector ({cfg.BODY_MODEL})...",
        f"Loading Head Detector ({cfg.HEAD_MODEL})...",
        "Warming up networks...",
        "Loading Replay Buffer...",
        "Starting Online Learner...",
        "Initializing Gate Manager...",
        "Initializing Analytics...",
        "Opening video stream...",
        "Starting Video Player thread...",
        "Starting Detection Worker...",
        "Starting Analytics thread...",
        "Starting REST API...",
        "System ready!",
    ]
    loader = LoadingScreen("Crowd Master v5  |  Enterprise Edition", load_steps)

    loader.step("Loading CNN model...")
    cnn_model = CrowdCNN().to(cfg.DEVICE)
    if os.path.exists(cfg.MODEL_SAVE):
        try:
            state = torch.load(cfg.MODEL_SAVE, map_location=cfg.DEVICE,
                               weights_only=True)
            cnn_model.load_state_dict(state)
            logger.info(f"CNN loaded: {cfg.MODEL_SAVE}")
        except Exception as e:
            logger.warning(f"CNN load failed ({e}) — starting fresh.")
    cnn_model.eval()

    loader.step(f"Loading Body Detector ({cfg.BODY_MODEL})...")
    body_yolo = YOLO(cfg.BODY_MODEL)
    body_yolo.to(cfg.DEVICE)
    if cfg.USE_FP16:
        body_yolo.model.half()   # FP16 — 2× faster on GPU
        logger.info("Body YOLO: FP16 enabled")
    try:
        body_yolo.model.fuse()   # fuse Conv+BN layers for faster inference
        logger.info("Body YOLO: layers fused")
    except Exception:
        pass

    # Second instance of the same weights for the slicing / TTA passes, so those
    # never touch the tracked model's predictor state. Costs ~22 MB of VRAM.
    aux_yolo = None
    if cfg.USE_SLICED or cfg.USE_TTA:
        try:
            aux_yolo = YOLO(cfg.BODY_MODEL)
            aux_yolo.to(cfg.DEVICE)
            if cfg.USE_FP16:
                aux_yolo.model.half()
            try:
                aux_yolo.model.fuse()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Auxiliary detector unavailable ({e}) — "
                           "slicing/TTA will share the tracked model.")

    loader.step(f"Loading Head Detector ({cfg.HEAD_MODEL})...")
    head_det = HeadDetector(cfg.HEAD_MODEL, cfg.DEVICE,
                            cfg.HEAD_CONF_BASE, cfg.HEAD_IOU,
                            cfg.HEAD_PERSIST_FRAMES)

    fusion      = DetectionFusion(cfg.FUSION_DIST)
    slicer      = SlicedInference(cfg.SLICE_ROWS, cfg.SLICE_COLS, cfg.SLICE_OVERLAP)
    density_est = DensityEstimator(cfg.CROWD_THRESHOLD)
    adapt_conf  = AdaptiveConfidence(cfg.BODY_CONF_BASE)

    loader.step("Loading Replay Buffer...")
    online_on = bool(getattr(cfg, "ENABLE_ONLINE_LEARNING", False))
    # Skip loading the (potentially ~0.5 GB) replay buffer when nothing will train.
    buffer  = ReplayBuffer(cfg.BUFFER_SIZE,
                           cfg.BUFFER_SAVE if online_on else "")

    loader.step("Starting Online Learner...")
    # One lock shared by the learner and the retrainer — they mutate the same
    # weights and used to hold two independent locks.
    model_lock = threading.Lock()
    if online_on:
        learner = OnlineLearner(cnn_model, buffer, cfg, model_lock)
        logger.info("Online learning: ENABLED (costs roughly 40-60%% of frame rate)")
    else:
        learner = NullOnlineLearner()
        logger.info("Online learning: disabled "
                    "(set CROWD_MASTER_ONLINE_LEARNING=1 to enable)")

    loader.step("Initializing Gate Manager...")
    gate_mgr = GateManager(cfg)

    loader.step("Initializing Analytics...")
    drift_det   = DriftDetector(cfg.DRIFT_WINDOW, cfg.DRIFT_THRESHOLD, cfg.DRIFT_LOG)
    forecaster  = CrowdForecaster(cfg.FORECAST_STEPS, cfg.DEVICE)
    anomaly_det = DeepAnomalyDetector(cfg.SPEED_THRESH, cfg.DENSITY_JUMP_THRESH)
    adapt_thr   = AdaptiveThreshold(cfg.CROWD_THRESHOLD, cfg.CROWD_THRESHOLD + 10)
    feedback    = FeedbackLoop()
    csv_log     = CSVLogger(cfg.LOG_FILE)
    peak_track  = PeakTimeTracker()
    retrainer   = AutoRetrainer(cnn_model, buffer, cfg, model_lock)
    if online_on:
        retrainer.start()   # pointless without a buffer being filled

    loader.step("Opening video stream...")
    video_source = resolve_video_source(cfg)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source_label(cfg)}")
        loader.done(); return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if is_live_source(cfg):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAMERA_BUFFER_SIZE)
    logger.info(f"Source: {source_label(cfg)}  {frame_w}x{frame_h}  FPS:{fps:.1f}  Device:{cfg.DEVICE}")

    # ── Warm up both networks before playback starts ───────────
    # torch.backends.cudnn.benchmark=True makes the first inference at a new
    # input shape run an exhaustive algorithm search, and the tracker/keypoint
    # paths only compile once they actually see a detection. Paying that on the
    # first live frame measured 10.4 s, during which the video plays with no
    # boxes at all. A real frame is used so the detection-bearing code paths are
    # exercised too — a blank frame only got it down to 4.1 s.
    loader.step("Warming up networks...")
    _warm = None
    try:
        _t0 = time.perf_counter()
        if not is_live_source(cfg):
            # Read the sample from a SEPARATE capture. Reading from `cap` and
            # seeking back is not reliable across container/codec combinations —
            # doing it truncated playback to roughly the first third of the clip.
            _probe = cv2.VideoCapture(video_source)
            _ok, _wf = _probe.read()
            _probe.release()
            if _ok and _wf is not None:
                _warm = cv2.resize(_wf, (cfg.INFER_W, cfg.INFER_H))
        if _warm is None:
            _warm = np.random.randint(0, 255, (cfg.INFER_H, cfg.INFER_W, 3),
                                      dtype=np.uint8)
        for _ in range(2):
            body_yolo.track(_warm, classes=[0], persist=False, verbose=False,
                            tracker=cfg.TRACKER, conf=cfg.BODY_CONF_BASE,
                            iou=cfg.BODY_IOU)
            head_det.detect(_warm, use_tta=False)
        head_det._tracks.clear()      # discard tracks seeded by the warm-up
        logger.info(f"Warm-up complete in {time.perf_counter()-_t0:.1f}s")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")

    loader.step("Starting Video Player thread...")
    player = VideoPlayerThread(cap, fps, cfg)
    player.start()

    audio_player: NullAudioPlayer = NullAudioPlayer()
    if not is_live_source(cfg):
        _aud_thread = AudioPlayerThread(cfg.VIDEO_PATH, fps=fps)
        _aud_thread.start()
        def _swap_audio():
            nonlocal audio_player
            audio_player = _aud_thread
        threading.Timer(0.3, _swap_audio).start()
    else:
        logger.info("Audio disabled for live camera sources.")

    result = DetectionResult()
    result.crowd_thr = cfg.CROWD_THRESHOLD
    result.forecasts = {5:0, 30:0, 60:0, 300:0}
    result.anomalies = []

    loader.step("Starting Detection Worker...")
    det_worker = DetectionWorker(
        player.detect_queue, result,
        body_yolo, head_det, fusion, slicer,
        density_est, adapt_conf, gate_mgr, learner,
        cfg, frame_h,
        warmup_frame=_warm,
        aux_yolo=aux_yolo,
    )
    det_worker.start()

    loader.step("Starting Analytics thread...")
    analytics = AnalyticsThread(
        result, drift_det, forecaster, anomaly_det,
        adapt_thr, feedback, csv_log, peak_track,
        gate_mgr, cfg, fps,
    )
    analytics.start()

    loader.step("Starting REST API...")
    start_api(result, gate_mgr, cfg)

    loader.step("System ready!")
    loader.done()

    heatmap    = CrowdHeatmap(frame_w, frame_h, cfg.HEATMAP_DECAY, cfg.HEATMAP_RADIUS)
    pz         = PanZoomView(cfg)
    show_boxes    = True
    show_stats    = True
    show_zones    = False
    show_heads    = True
    show_heatmap  = False
    show_controls = bool(getattr(cfg, "SHOW_CONTROLS", True))
    fullscreen    = False
    enhance_mode  = False
    quit_flag     = False
    zone_counts: Dict[str, int] = {z: 0 for z in cfg.ZONES}
    last_frame_id = 0
    last_frame    = None
    fps_hist      = deque(maxlen=30)
    last_draw_t   = time.perf_counter()

    display_enabled = (not getattr(cfg, "HEADLESS", False)) and highgui_available()
    max_frames      = int(getattr(cfg, "MAX_FRAMES", 0))
    if getattr(cfg, "HEADLESS", False):
        logger.info("Headless mode — no window; "
                    + (f"stopping after {max_frames} frames."
                       if max_frames else "stop with Ctrl+C."))
    if display_enabled:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    else:
        logger.warning(
            "OpenCV HighGUI unavailable — running in headless mode (no video window). "
            "To get the video window: pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python"
        )
        # Don't quit — keep processing and logging, just no display window

    while not quit_flag:
        try:
            fid, raw = player.display_queue.get_nowait()
            last_frame_id = fid
            last_frame    = raw
        except queue.Empty:
            raw = last_frame

        if raw is None:
            safe_wait_key(10)
            continue

        snap = result.snapshot()

        # ★ FIX: this reset was inside the `len(boxes) > 0` guard, so when the
        # scene emptied the zone overlay kept displaying the last non-zero counts.
        zone_counts = {z: 0 for z in cfg.ZONES}
        if snap.body_boxes is not None and len(snap.body_boxes) > 0:
            for box in snap.body_boxes:
                cx = (box[0]+box[2])/2/frame_w
                cy = (box[1]+box[3])/2/frame_h
                for name,(x1,y1,x2,y2) in cfg.ZONES.items():
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        zone_counts[name] += 1

        display = raw.copy()

        all_centers = (list(snap.head_centers or []) + list(snap.orphan_heads or []))
        heatmap.update(all_centers)
        if show_heatmap:
            display = heatmap.render(display, alpha=0.55)

        if show_boxes and snap.body_boxes is not None and len(snap.body_boxes) > 0:
            draw_boxes(display, snap.body_boxes,
                       snap.body_ids, snap.body_confs,
                       snap.crowd_thr, cfg)

        if show_heads:
            draw_heads(display,
                       snap.head_centers or [],
                       snap.orphan_heads or [], cfg)

        if show_zones:
            draw_zones(display, cfg.ZONES, zone_counts, frame_w, frame_h)

        # Gates live in video coordinates, so they are drawn BEFORE the pan/zoom
        # transform and travel with the footage.
        gate_mgr.draw_preview(display, scale=1.0)

        display = pz.apply(display, enhance=enhance_mode)

        # ★ FIX: the dashboard used to be drawn before pz.apply(), so zooming in
        # magnified and cropped the stats panel along with the video. The HUD is
        # screen furniture — it belongs on top of the transformed frame, at a
        # fixed size. The duplicate second gate_mgr.draw_preview() that used to
        # sit here (re-drawing gates at untransformed coordinates over the zoomed
        # image) is gone.
        draw_dashboard(display, snap, gate_mgr, show_stats,
                       retrainer.status, pz, player.speed,
                       player.paused, det_worker.use_tta,
                       cfg.USE_SLICED, enhance_mode)

        now_t = time.perf_counter()
        fps_hist.append(1.0 / (now_t - last_draw_t + 1e-9))
        last_draw_t = now_t
        result.update(display_fps=float(np.mean(fps_hist)))

        _gsel = (f"[G{gate_mgr.gates[gate_mgr.selected_idx].gate_id} sel | R=rev ESC=desel]  "
                 if 0 <= gate_mgr.selected_idx < len(gate_mgr.gates) else "")
        if display_enabled:
            cv2.setWindowTitle(
                WIN_NAME,
                f"Crowd Master v5  |  {frame_w}x{frame_h}  |  "
                f"People:{snap.final_count}  Accurate:{getattr(snap,'accurate_count',snap.final_count)}  "
                f"Gates In:{gate_mgr.total_entry} Out:{gate_mgr.total_exit}  |  "
                f"Spd:{player.speed:.1f}x  Zm:{pz.zoom:.1f}x  |  "
                f"{_gsel}"
                f"{'[PAUSED] ' if player.paused else ''}"
                f"Infer:{snap.inference_ms:.0f}ms"
            )
            if show_controls:
                draw_controls_overlay(display, cfg, player, gate_mgr, show_boxes, show_heads,
                                      show_zones, show_heatmap, show_stats, enhance_mode, det_worker)
            cv2.imshow(WIN_NAME, display)

        if max_frames and player.frames_served >= max_frames:
            logger.info(f"Reached CROWD_MASTER_MAX_FRAMES={max_frames} — stopping.")
            quit_flag = True
            continue

        if player.ended.is_set() and player.display_queue.empty():
            logger.info("End of source reached — stopping.")
            quit_flag = True
            continue

        raw_key = safe_wait_key(1)
        key = (raw_key & 0xFF) if raw_key >= 0 else 255

        if key == ord("q"):
            quit_flag = True

        elif display_enabled and cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            quit_flag = True

        elif key == ord("k"):
            show_controls = not show_controls

        elif key == ord("p"):
            player.paused = not player.paused
            try:
                if player.paused:
                    audio_player.pause()
                else:
                    cur_sec = player.frame_id / max(fps, 1)
                    audio_player.seek(cur_sec, resume=True)
            except Exception:
                pass

        elif key == ord("g"):
            disp_scale = pz.zoom if pz.zoom != 1.0 else 1.0
            gate_mgr.toggle_draw_mode(WIN_NAME, disp_scale)

        elif key == 9:  # TAB
            gate_mgr.select_next()

        elif key == 27:  # ESC
            gate_mgr.selected_idx = -1

        elif key == ord("r"):
            # R reverses the selected gate; with nothing selected it resets the
            # view. (These were two separate branches guarded by conditions that
            # both had to be re-evaluated — the second was unreachable whenever a
            # gate happened to be selected.)
            if gate_mgr.selected_idx >= 0 and len(gate_mgr.gates) > 0:
                gate_mgr.reverse_selected()
            else:
                pz.reset()
                player.speed = cfg.DEFAULT_SPEED
                if fullscreen:
                    fullscreen = False
                    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)

        elif key == ord("x"):
            if gate_mgr.selected_idx >= 0 and len(gate_mgr.gates) > 0:
                gate_mgr.delete_selected()
            else:
                gate_mgr.clear_all()

        elif key in (ord("+"), ord("=")):
            player.speed = min(cfg.MAX_SPEED, round(player.speed + cfg.SPEED_STEP, 2))

        elif key in (ord("-"), ord("_")):
            player.speed = max(cfg.MIN_SPEED, round(player.speed - cfg.SPEED_STEP, 2))

        elif key == ord("v"):
            try:
                v = getattr(audio_player, "_vol", 1.0)
                audio_player.set_volume(min(1.0, v + 0.1))
            except Exception:
                pass

        # ★ FIX: C was documented as volume-down but bound to snapshot, so there
        # was no way to lower the volume at all. C now lowers volume as
        # documented; snapshots moved to S (and WASD panning uses arrow keys).
        elif key == ord("c"):
            try:
                v = getattr(audio_player, "_vol", 1.0)
                audio_player.set_volume(max(0.0, v - 0.1))
            except Exception:
                pass

        elif key == ord("s"):
            save_snapshot(display, cfg)

        elif key == ord("]"):
            pz.zoom_in()

        elif key == ord("["):
            pz.zoom_out()

        elif key == ord("w") or key == 82 or raw_key in (2490368, 65362):
            pz.pan(0, -cfg.PAN_STEP)
        # S now saves a snapshot (see above), so pan-down is the down arrow.
        elif key == 84 or raw_key in (2621440, 65364):
            pz.pan(0,  cfg.PAN_STEP)
        elif key == ord("a") or key == 81 or raw_key in (2424832, 65361):
            pz.pan(-cfg.PAN_STEP, 0)
        elif key == ord("d") or key == 83 or raw_key in (2555904, 65363):
            pz.pan( cfg.PAN_STEP, 0)

        elif key == ord("."):
            player.seek_seconds(+5.0)
            try: audio_player.seek(player.frame_id / max(fps, 1) + 5.0)
            except Exception: pass

        elif key == ord(","):
            player.seek_seconds(-5.0)
            try: audio_player.seek(max(0.0, player.frame_id / max(fps, 1) - 5.0))
            except Exception: pass

        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

        elif key == ord("t"):
            det_worker.use_tta = not det_worker.use_tta

        elif key == ord("o"):
            # O = toggle detection ON/OFF (tracking continues)
            det_worker.detection_enabled = not det_worker.detection_enabled
            logger.info(
                f"Detection: {'ON' if det_worker.detection_enabled else 'OFF (tracking only)'}")

        elif key == 18:  # Ctrl+R
            retrainer.trigger()

        elif key == ord("i"):
            enhance_mode = not enhance_mode

        elif key == ord("n"):
            show_stats = not show_stats

        elif key == ord("b"):
            show_boxes = not show_boxes

        elif key == ord("h"):
            show_heads = not show_heads

        elif key == ord("z"):
            show_zones = not show_zones

        elif key == ord("m"):
            show_heatmap = not show_heatmap

        elif key == ord("e"):
            try:
                cnn_model.eval()
                dummy = torch.randn(1, 3, 224, 224).to(cfg.DEVICE)
                torch.onnx.export(cnn_model, dummy, cfg.ONNX_PATH,
                                  input_names=["image"],
                                  output_names=["count"],
                                  opset_version=12)
                logger.info(f"ONNX: {cfg.ONNX_PATH}")
            except Exception as ex:
                logger.error(f"ONNX failed: {ex}")

    # ── Cleanup ──────────────────────────────────────────────
    logger.info("Shutting down...")
    gate_mgr.save()
    player.stop()
    try:
        audio_player.stop()
    except Exception:
        pass
    det_worker.stop()
    analytics.stop()
    retrainer.stop()
    learner.stop()
    drift_det.close()
    csv_log.close()
    # ★ FIX: the model and buffer used to be written out unconditionally on every
    # exit, overwriting a good checkpoint with whatever half-trained state the
    # session happened to end in — even on a run that trained nothing at all.
    if online_on:
        buffer.save()
        torch.save(cnn_model.state_dict(), cfg.model_save_path())
        logger.info(f"CNN checkpoint saved: {cfg.model_save_path()}")
    cap.release()
    if highgui_available():
        cv2.destroyAllWindows()

    session_stats = {
        "frames":       last_frame_id,
        "updates":      learner.updates,
        "drifts":       drift_det.drift_events,
        "mae_cnn":      feedback.mae_cnn,
        "mae_lstm":     feedback.mae_lstm,
        "avg_infer_ms": det_worker.avg_infer_ms(),
        "display_fps":  analytics.avg_display_fps(),
    }

    peak_track.build_report(cfg.CROWD_THRESHOLD, cfg.REPORT_FILE,
                            session_stats, cfg, gate_mgr)
    generate_pdf_report(peak_track, cfg, session_stats, gate_mgr)

    print("\n" + "="*58)
    print("       CROWD MASTER v5  —  SESSION SUMMARY")
    print("="*58)
    for k, v in session_stats.items():
        val = v if isinstance(v, int) else f"{v:.2f}"
        print(f"  {k:<24}: {val}")
    print(f"  {'Gates':<24}: {len(gate_mgr.gates)} gate(s)")
    print(f"  {'Total Entry':<24}: {gate_mgr.total_entry}")
    print(f"  {'Total Exit':<24}: {gate_mgr.total_exit}")
    net = gate_mgr.total_entry - gate_mgr.total_exit
    print(f"  {'Net inside':<24}: {net:+d}")
    print(f"  {'Report':<24}: {cfg.REPORT_FILE}")
    print(f"  {'PDF':<24}: {cfg.PDF_REPORT}")
    print("="*58 + "\n")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = Config()

    # ── Tkinter video selector ─────────────────────────────────
    # Always show GUI launcher unless video is given via env var
    # and it already exists on disk.
    env_video = os.getenv("CROWD_MASTER_VIDEO", "")
    skip_gui  = bool(env_video and os.path.exists(env_video)) \
                or cfg.HEADLESS or is_live_source(cfg)

    if not skip_gui:
        launcher = VideoLauncher(cfg)
        chosen   = launcher.run()
        if not chosen:
            logger.info("No video selected — exiting.")
            sys.exit(0)
        logger.info(f"Source selected: {chosen}")

    # ── Startup validation ─────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CROWD MASTER v5.1  |  Enterprise AI Surveillance Platform")
    logger.info("=" * 70)
    logger.info(f"Device   : {cfg.DEVICE.upper()}")
    logger.info(f"FP16     : {'ON' if cfg.USE_FP16 else 'OFF'}")
    if cfg.DEVICE == "cuda":
        try:
            logger.info(f"GPU      : {torch.cuda.get_device_name(0)}")
        except Exception:
            logger.info("GPU      : CUDA available")
    logger.info(f"Source   : {source_label(cfg)}")
    logger.info(f"Data dir : {cfg._data_dir}")
    if not is_live_source(cfg) and not os.path.exists(cfg.VIDEO_PATH):
        logger.error("Video file not found: " + cfg.VIDEO_PATH)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("✅ All checks passed — Starting system...")
    logger.info("=" * 70)

    run(cfg)
