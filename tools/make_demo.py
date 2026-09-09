"""Render a demo clip: the system running, overlays and all.

Produces the video you would show someone rather than a screen recording —
deterministic, no window needed, no dropped frames, and it can render a
side-by-side against a different configuration for a before/after.

    venv\\Scripts\\python.exe tools\\make_demo.py clip.mp4 -o demo.mp4 --seconds 25

    # before/after: the shipped v1 config on the left, v2 on the right
    venv\\Scripts\\python.exe tools\\make_demo.py clip.mp4 -o compare.mp4 --compare
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load(data_dir: Path):
    os.environ["CROWD_MASTER_DATA_DIR"] = str(data_dir)
    spec = importlib.util.spec_from_file_location("cm", ROOT / "crowd_master_v2.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    return cm


def build_pipeline(cm, cfg, first):
    """A detection worker plus everything it needs, wired for offline rendering."""
    import queue
    body = cm.YOLO(cfg.BODY_MODEL)
    body.to(cfg.DEVICE)
    if cfg.USE_FP16:
        body.model.half()
    try:
        body.model.fuse()
    except Exception:
        pass
    head = cm.HeadDetector(cfg.HEAD_MODEL, cfg.DEVICE, cfg.HEAD_CONF_BASE,
                           cfg.HEAD_IOU, cfg.HEAD_PERSIST_FRAMES)
    gm = cm.GateManager(cfg)
    result = cm.DetectionResult()
    dq: queue.Queue = queue.Queue(maxsize=1)
    worker = cm.DetectionWorker(
        dq, result, body, head,
        cm.DetectionFusion(cfg.FUSION_DIST),
        cm.SlicedInference(cfg.SLICE_ROWS, cfg.SLICE_COLS, cfg.SLICE_OVERLAP),
        cm.DensityEstimator(cfg.CROWD_THRESHOLD),
        cm.AdaptiveConfidence(cfg.BODY_CONF_BASE),
        gm, cm.NullOnlineLearner(), cfg, first.shape[0],
        warmup_frame=first)
    worker.start()
    return worker, result, dq, gm


def label(cm, frame, text, sub=""):
    """Caption strip along the top, in the HUD's own typeface."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    h, w = frame.shape[:2]
    s = max(0.75, min(1.6, w / 1280.0))
    px = lambda v: max(1, int(round(v * s)))
    bar_h = px(52 if sub else 36)
    strip = Image.new("RGB", (w, bar_h), cm.HudRenderer.C_BG)
    d = ImageDraw.Draw(strip)
    d.text((px(16), px(9)), text, font=cm._HUD._font("b", px(19)),
           fill=cm.HudRenderer.C_VALUE)
    if sub:
        d.text((px(16), px(31)), sub, font=cm._HUD._font("r", px(13)),
               fill=cm.HudRenderer.C_LABEL)
    arr = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)
    return np.vstack([arr, frame])


def render(cm, cfg, video, seconds, caption, sub, on_frame=None):
    """Run the pipeline over the clip and yield finished display frames."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    worker, result, dq, gm = build_pipeline(cm, cfg, first)
    time.sleep(0.3)

    heat = cm.CrowdHeatmap(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           cfg.HEATMAP_DECAY, cfg.HEATMAP_RADIUS)
    pz = cm.PanZoomView(cfg)
    want = int(seconds * fps)
    out = []
    n = 0
    while n < want:
        ok, raw = cap.read()
        if not ok:
            break
        n += 1
        dq.put((n, raw))
        # Let the worker finish this frame so the overlay matches the image.
        deadline = time.time() + 1.5
        while result.snapshot().frame_id < n and time.time() < deadline:
            time.sleep(0.002)
        snap = result.snapshot()

        disp = raw.copy()
        vs = 1.0
        if cfg.HUD_MIN_WIDTH and disp.shape[1] < cfg.HUD_MIN_WIDTH:
            vs = cfg.HUD_MIN_WIDTH / disp.shape[1]
            disp = cv2.resize(disp, (cfg.HUD_MIN_WIDTH,
                                     int(round(disp.shape[0] * vs))),
                              interpolation=cv2.INTER_LINEAR)
        heat.update(list(snap.head_centers or []))
        if snap.body_boxes is not None and len(snap.body_boxes):
            cm.draw_boxes(disp, np.asarray(snap.body_boxes) * vs,
                          snap.body_ids, snap.body_confs, snap.crowd_thr, cfg)
        cm.draw_dashboard(disp, snap, gm, True, "idle", pz, 1.0,
                          False, False, cfg.USE_SLICED, False)
        if on_frame:
            on_frame(disp, snap, vs)
        out.append(label(cm, disp, caption, sub))
        if n % 50 == 0:
            print(f"    {n}/{want}")
    worker.stop()
    cap.release()
    return out, fps


def write(path, frames, fps):
    import cv2
    if not frames:
        raise SystemExit("nothing rendered")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"\n  wrote {path}  {w}x{h}  {len(frames)} frames  {len(frames)/fps:.1f}s")


# Each capability gets its own stretch of the clip, captioned as it appears, so
# a viewer is told what they are looking at rather than left to infer it.
SHOWCASE = [
    ("Detection and tracking", "every person boxed and given a stable ID"),
    ("Gate counting",          "a line you draw; crossings tallied by direction"),
    ("Per-zone occupancy",     "peak and average per area, written to CSV"),
    ("Density heatmap",        "where the crowd accumulates over time"),
]


def render_showcase(cm, cfg, video, seconds):
    """One pass over the clip; overlays and caption change on a schedule."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    worker, result, dq, gm = build_pipeline(cm, cfg, first)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # A vertical line across the middle, so the flow counters carry real
    # numbers instead of sitting at zero through the whole clip.
    gm.gates = [cm.Gate(0, fw // 2, 0, fw // 2, fh, (0, 220, 255))]
    time.sleep(0.3)

    heat = cm.CrowdHeatmap(fw, fh, cfg.HEATMAP_DECAY, cfg.HEATMAP_RADIUS)
    pz = cm.PanZoomView(cfg)
    want = int(seconds * fps)
    seg_len = max(1, want // len(SHOWCASE))
    out, n = [], 0

    while n < want:
        ok, raw = cap.read()
        if not ok:
            break
        n += 1
        dq.put((n, raw))
        deadline = time.time() + 1.5
        while result.snapshot().frame_id < n and time.time() < deadline:
            time.sleep(0.002)
        snap = result.snapshot()
        seg = min(len(SHOWCASE) - 1, (n - 1) // seg_len)
        title, sub = SHOWCASE[seg]

        disp = raw.copy()
        vs = 1.0
        if cfg.HUD_MIN_WIDTH and disp.shape[1] < cfg.HUD_MIN_WIDTH:
            vs = cfg.HUD_MIN_WIDTH / disp.shape[1]
            disp = cv2.resize(disp, (cfg.HUD_MIN_WIDTH,
                                     int(round(disp.shape[0] * vs))),
                              interpolation=cv2.INTER_LINEAR)

        heat.update(list(snap.head_centers or []))
        if seg == 3:
            disp = heat.render(disp, alpha=0.55)
        if seg >= 1:
            gm.draw_preview(disp, scale=vs)
        if snap.body_boxes is not None and len(snap.body_boxes):
            cm.draw_boxes(disp, np.asarray(snap.body_boxes) * vs,
                          snap.body_ids, snap.body_confs, snap.crowd_thr, cfg)
        if seg == 2:
            cm.draw_zones(disp, cfg.ZONES, snap.zone_counts or {},
                          disp.shape[1], disp.shape[0])
        cm.draw_dashboard(disp, snap, gm, seg >= 1, "idle", pz, 1.0,
                          False, False, cfg.USE_SLICED, False)
        out.append(label(cm, disp, title, sub))
        if n % 50 == 0:
            print(f"    {n}/{want}")

    worker.stop()
    cap.release()
    return out, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="demo.mp4")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--compare", action="store_true",
                    help="side-by-side against the shipped v1 configuration")
    ap.add_argument("--showcase", action="store_true",
                    help="cycle through the features, captioned")
    a = ap.parse_args()

    import cv2
    import numpy as np

    tmp = ROOT / "DATA" / "_demo"
    cm = load(tmp)

    cfg = cm.Config()
    cfg.API_ENABLED = False

    if a.showcase:
        print("  rendering showcase ...")
        frames, fps = render_showcase(cm, cfg, a.video, a.seconds)
        write(a.out, frames, fps)
        return 0

    print("  rendering v2 ...")
    right, fps = render(cm, cfg, a.video, a.seconds,
                        "v2", "yolov8s @ 960x540 - fixed NMS - 43ms/frame")

    if a.compare:
        # The configuration the published v1 shipped with.
        cfg1 = cm.Config()
        cfg1.API_ENABLED = False
        cfg1.BODY_MODEL, cfg1.HEAD_MODEL = "yolov8n.pt", "yolov8n-pose.pt"
        cfg1.INFER_W, cfg1.INFER_H = 640, 360
        print("  rendering v1 configuration ...")
        left, _ = render(cm, cfg1, a.video, a.seconds,
                         "v1", "yolov8n @ 640x360")
        n = min(len(left), len(right))
        frames = [np.hstack([left[i], right[i]]) for i in range(n)]
    else:
        frames = right

    write(a.out, frames, fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
