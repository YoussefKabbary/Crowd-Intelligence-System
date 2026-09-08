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


_MODELS = {}


def reset_tracker(model) -> None:
    """Clear ByteTrack state so a reused model starts each case clean.

    The models are loaded once and shared across cases (see get_models); their
    tracker state is not shareable, so it is reset here rather than paying to
    reload three networks per case.
    """
    pred = getattr(model, "predictor", None)
    trackers = getattr(pred, "trackers", None) if pred is not None else None
    if trackers:
        for t in trackers:
            if hasattr(t, "reset"):
                t.reset()
            else:                       # older ultralytics: drop and rebuild
                model.predictor = None
                return
    else:
        model.predictor = None


def get_models(cfg):
    """Load body / head / aux once for the whole suite.

    Reloading them per case needed roughly four times the memory and time, and
    on a machine with under a gigabyte free it failed outright with a CPU
    allocator error partway through the run.
    """
    if "body" not in _MODELS:
        body = cm.YOLO(cfg.BODY_MODEL); body.to(cfg.DEVICE)
        if cfg.USE_FP16:
            body.model.half()
        try:
            body.model.fuse()
        except Exception:
            pass
        aux = cm.YOLO(cfg.BODY_MODEL); aux.to(cfg.DEVICE)
        if cfg.USE_FP16:
            aux.model.half()
        try:
            aux.model.fuse()
        except Exception:
            pass
        _MODELS["body"] = body
        _MODELS["aux"] = aux
        _MODELS["head"] = cm.HeadDetector(
            cfg.HEAD_MODEL, cfg.DEVICE, cfg.HEAD_CONF_BASE,
            cfg.HEAD_IOU, cfg.HEAD_PERSIST_FRAMES)
    return _MODELS["body"], _MODELS["head"], _MODELS["aux"]


def run_case(n_people: int, force_slicing: bool,
             sequential: bool = True) -> tuple:
    clip = TMP / f"gate_{n_people}_{int(force_slicing)}_{int(sequential)}.mp4"
    build(str(clip), n_people, sequential=sequential)

    cfg = cm.Config()
    cfg.API_ENABLED = False
    if force_slicing:
        cfg.USE_SLICED = True
        cfg.DENSE_FALLBACK_THR = 1        # make the slicing path always engage

    body, head, aux = get_models(cfg)
    reset_tracker(body)
    head._tracks.clear()

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

    # ── Asserted: walkers cross one at a time ────────────────
    # Identity is unambiguous here, so the count is a property of the gate logic
    # and an exact match is a fair thing to require.
    print("  Sequential crossings (asserted)")
    for slicing in (False, True):
        for expected in (1, 3):
            entry, exit_, saw_ids, max_tracked = run_case(
                expected, slicing, sequential=True)
            tag = "sliced" if slicing else "plain "
            ok = saw_ids and entry == expected and exit_ == 0
            print(f"    [{tag}] expect entry={expected} exit=0 -> "
                  f"got entry={entry} exit={exit_} | tracked_ids={max_tracked} "
                  f"{'OK' if ok else 'FAIL'}")
            if not saw_ids:
                failures.append(f"{tag}/{expected}: tracking lost (no track IDs seen)")
            elif not ok:
                failures.append(f"{tag}/{expected}: entry={entry} exit={exit_}")

    # ── Reported, not asserted: simultaneous crossings ───────
    # Three walkers of the same size at the same speed in the same direction is
    # close to the worst case for a motion-only tracker. It swaps their IDs, and
    # gate counting is identity-based, so the total is not reliable: observed
    # runs credit one ID with two crossings and miss another walker entirely,
    # landing on the right total only by cancellation. Asserting on it would
    # make the suite pass or fail on unrelated changes — turning cuDNN
    # autotuning off was enough to flip it — so it is measured and printed
    # instead. Treat a number far from the expectation as a signal to look at
    # tracker choice or re-identification, not as a broken gate.
    print()
    print("  Simultaneous crossings (reported — see comment in main())")
    for slicing in (False, True):
        entry, exit_, saw_ids, max_tracked = run_case(3, slicing, sequential=False)
        tag = "sliced" if slicing else "plain "
        print(f"    [{tag}] expect entry=3 exit=0 -> "
              f"got entry={entry} exit={exit_} | tracked_ids={max_tracked}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("all asserted gate cases passed")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
