"""Gate crossing regression test.

Builds a clip with a known number of people walking left->right across a known
vertical line, runs the real detection pipeline over it, and asserts the gate
reports exactly that many entries and no exits.

This is the test that would have caught the v5.2 regression where enabling
sliced inference returned body_ids=None and silently zeroed every gate count.

    ./venv/Scripts/python.exe tests/test_gate_counting.py
"""
import os, sys, time, queue, warnings, tempfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="crowdmaster_gate_"))
os.environ["CROWD_MASTER_DATA_DIR"] = str(TMP)

import cv2
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("cm", ROOT / "crowd_master_v2.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

sys.path.insert(0, str(Path(__file__).parent))
from make_gate_clip import build           # noqa: E402


def run_case(n_people: int, force_slicing: bool) -> tuple:
    clip = TMP / f"gate_{n_people}_{int(force_slicing)}.mp4"
    build(str(clip), n_people)

    cfg = cm.Config()
    cfg.API_ENABLED = False
    if force_slicing:
        cfg.USE_SLICED = True
        cfg.DENSE_FALLBACK_THR = 1        # make the slicing path always engage

    body = cm.YOLO(cfg.BODY_MODEL); body.to(cfg.DEVICE)
    if cfg.USE_FP16:
        body.model.half()
    try:
        body.model.fuse()
    except Exception:
        pass
    head = cm.HeadDetector(cfg.HEAD_MODEL, cfg.DEVICE,
                           cfg.HEAD_CONF_BASE, cfg.HEAD_IOU,
                           cfg.HEAD_PERSIST_FRAMES)

    aux = None
    if cfg.USE_SLICED or cfg.USE_TTA:
        aux = cm.YOLO(cfg.BODY_MODEL); aux.to(cfg.DEVICE)
        if cfg.USE_FP16:
            aux.model.half()
        try:
            aux.model.fuse()
        except Exception:
            pass

    gm = cm.GateManager(cfg)
    gm.gates = [cm.Gate(0, 640, 0, 640, 720, (0, 220, 255))]   # vertical mid-line
    result = cm.DetectionResult()
    dq = queue.Queue(maxsize=1)

    cap = cv2.VideoCapture(str(clip))
    ok, first = cap.read()
    worker = cm.DetectionWorker(
        dq, result, body, head,
        cm.DetectionFusion(cfg.FUSION_DIST),
        cm.SlicedInference(cfg.SLICE_ROWS, cfg.SLICE_COLS, cfg.SLICE_OVERLAP),
        cm.DensityEstimator(cfg.CROWD_THRESHOLD),
        cm.AdaptiveConfidence(cfg.BODY_CONF_BASE),
        gm, cm.NullOnlineLearner(), cfg, 720,
        warmup_frame=first, aux_yolo=aux)
    worker.start()
    time.sleep(0.2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = 0
    saw_track_ids = False
    max_tracked = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        dq.put((n, fr))            # blocking: every frame is analysed
        # Sample tracking DURING the run. Checking after the clip ends is
        # meaningless: by then everyone has walked out of frame and body_ids is
        # legitimately None because there is nothing to track.
        snap = result.snapshot()
        if snap.body_ids is not None and len(snap.body_ids) > 0:
            saw_track_ids = True
            max_tracked = max(max_tracked, len(snap.body_ids))
    time.sleep(1.5)
    worker.stop()
    cap.release()
    g = gm.gates[0]
    return g.entry, g.exit, saw_track_ids, max_tracked


def main() -> int:
    failures = []
    for slicing in (False, True):
        for expected in (1, 3):
            entry, exit_, saw_ids, max_tracked = run_case(expected, slicing)
            tag = "sliced" if slicing else "plain "
            ok_ids = saw_ids
            ok_cnt = entry == expected and exit_ == 0
            print(f"  [{tag}] expect entry={expected} exit=0 -> "
                  f"got entry={entry} exit={exit_} | tracked_ids={max_tracked} "
                  f"{'OK' if (ok_ids and ok_cnt) else 'FAIL'}")
            if not ok_ids:
                failures.append(f"{tag}/{expected}: tracking lost (no track IDs seen)")
            if not ok_cnt:
                failures.append(f"{tag}/{expected}: entry={entry} exit={exit_}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("all gate cases passed")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
