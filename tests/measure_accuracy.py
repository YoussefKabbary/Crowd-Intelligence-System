"""Measure counting accuracy against hand-counted frames.

Runs the real detection pipeline over a clip and compares its output, at the
frames you labelled, with your counts. Every other benchmark in this project
measures speed; this is the only one that says whether the numbers are right.

    venv\\Scripts\\python.exe tools\\extract_frames.py clip.mp4 --count 200
    # ... count the frames, fill in labels.csv ...
    venv\\Scripts\\python.exe tests\\measure_accuracy.py clip.mp4 clip_labels/labels.csv

Reports MAE, RMSE and bias for each count the system produces, so you can see
which of them to trust:

    bodies    raw YOLO detections
    fused     bodies + heads that matched no body
    smooth    the 5-frame rolling figure shown as TOTAL
    accurate  the weighted blend shown as ACCURATE

Bias matters as much as MAE: a system that is consistently 2 low is easy to
correct, one that scatters +-4 is not.

The whole clip is played through in order, not just the labelled frames, so
tracking behaves exactly as it does in normal use.

Options let you measure a change without editing the code:
    --infer-width 640      inference resolution
    --model yolov8n.pt     detector weights
    --no-heads             skip the pose model (is it earning its 17-34 ms?)
"""
import argparse
import csv
import importlib.util
import os
import queue
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load_module(data_dir: Path):
    os.environ["CROWD_MASTER_DATA_DIR"] = str(data_dir)
    spec = importlib.util.spec_from_file_location("cm", ROOT / "crowd_master_v2.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    return cm


def read_labels(path: str):
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("count") or "").strip()
            if not raw:
                continue
            try:
                labels[int(row["frame"])] = int(float(raw))
            except (ValueError, KeyError):
                continue
    return labels


def stats(pairs):
    """(MAE, RMSE, bias, worst_error) for a list of (truth, predicted)."""
    if not pairs:
        return 0.0, 0.0, 0.0, 0
    errs = [p - t for t, p in pairs]
    n = len(errs)
    mae = sum(abs(e) for e in errs) / n
    rmse = (sum(e * e for e in errs) / n) ** 0.5
    bias = sum(errs) / n
    worst = max(errs, key=abs)
    return mae, rmse, bias, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("labels")
    ap.add_argument("--model", default=None, help="override body model weights")
    ap.add_argument("--infer-width", type=int, default=None)
    ap.add_argument("--no-heads", action="store_true",
                    help="run without the pose model")
    ap.add_argument("--csv", default=None, help="write per-frame results here")
    a = ap.parse_args()

    truth = read_labels(a.labels)
    if not truth:
        print(f"No filled-in counts in {a.labels}.")
        print("Fill the 'count' column first — see tools/extract_frames.py.")
        return 2

    import cv2
    import numpy as np

    data_dir = Path(a.labels).resolve().parent / "_accuracy_run"
    cm = load_module(data_dir)
    cfg = cm.Config()
    cfg.API_ENABLED = False
    if a.model:
        cfg.BODY_MODEL = a.model
    if a.infer_width:
        cfg.INFER_W = a.infer_width
        cfg.INFER_H = int(round(a.infer_width * 9 / 16 / 2) * 2)

    print(f"  video       : {a.video}")
    print(f"  labelled    : {len(truth)} frames")
    print(f"  model       : {cfg.BODY_MODEL} @ {cfg.INFER_W}x{cfg.INFER_H}"
          f"  heads={'off' if a.no_heads else cfg.HEAD_MODEL}")
    print()

    body = cm.YOLO(cfg.BODY_MODEL)
    body.to(cfg.DEVICE)
    if cfg.USE_FP16:
        body.model.half()
    try:
        body.model.fuse()
    except Exception:
        pass

    class _NoHeads:
        def detect(self, *_a, **_kw):
            return [], 0
        _tracks: list = []

    head = (_NoHeads() if a.no_heads else
            cm.HeadDetector(cfg.HEAD_MODEL, cfg.DEVICE, cfg.HEAD_CONF_BASE,
                            cfg.HEAD_IOU, cfg.HEAD_PERSIST_FRAMES))

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"cannot open {a.video}")
        return 2
    ok, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    result = cm.DetectionResult()
    dq: queue.Queue = queue.Queue(maxsize=1)
    worker = cm.DetectionWorker(
        dq, result, body, head,
        cm.DetectionFusion(cfg.FUSION_DIST),
        cm.SlicedInference(cfg.SLICE_ROWS, cfg.SLICE_COLS, cfg.SLICE_OVERLAP),
        cm.DensityEstimator(cfg.CROWD_THRESHOLD),
        cm.AdaptiveConfidence(cfg.BODY_CONF_BASE),
        cm.GateManager(cfg), cm.NullOnlineLearner(), cfg,
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        warmup_frame=first if ok else None)
    worker.start()
    time.sleep(0.2)

    rows = []
    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        # Blocking put: every frame is analysed, so the tracker sees a
        # continuous sequence exactly as it would in a live run.
        dq.put((idx, frame))
        if (idx - 1) in truth:
            # The worker reports the frame it has just finished; give it a
            # moment so the snapshot corresponds to this frame, not the last.
            deadline = time.time() + 2.0
            while result.snapshot().frame_id < idx and time.time() < deadline:
                time.sleep(0.002)
            s = result.snapshot()
            rows.append({
                "frame": idx - 1,
                "truth": truth[idx - 1],
                "bodies": s.body_count,
                "fused": s.raw_fused,
                "smooth": s.final_count,
                "accurate": s.accurate_count,
            })
        if idx % 200 == 0:
            print(f"    {idx} frames, {len(rows)}/{len(truth)} labelled seen")
    worker.stop()
    cap.release()
    elapsed = time.time() - t0

    if not rows:
        print("None of the labelled frame numbers were reached in this video.")
        return 2

    print()
    print(f"  compared {len(rows)} frames in {elapsed:.0f}s")
    print()
    print(f"  {'signal':<10}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'worst':>8}")
    print("  " + "-" * 42)
    best = None
    for name in ("bodies", "fused", "smooth", "accurate"):
        pairs = [(r["truth"], r[name]) for r in rows]
        mae, rmse, bias, worst = stats(pairs)
        print(f"  {name:<10}{mae:>8.2f}{rmse:>8.2f}{bias:>+8.2f}{worst:>+8d}")
        if best is None or mae < best[1]:
            best = (name, mae)

    truths = [r["truth"] for r in rows]
    mean_truth = sum(truths) / len(truths)
    print()
    print(f"  ground truth: mean {mean_truth:.1f}, "
          f"range {min(truths)}-{max(truths)}")
    print(f"  best signal : {best[0]}  (MAE {best[1]:.2f}, "
          f"{best[1] / max(mean_truth, 1e-9) * 100:.0f}% of the mean count)")
    print()
    print("  A positive bias means over-counting, negative means under-counting.")
    print("  Re-run after a change and compare MAE on the same labels.")

    out = a.csv or str(Path(a.labels).with_name("accuracy_results.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  per-frame   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
