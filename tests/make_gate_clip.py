"""Build a deterministic clip: N people walk left->right across the frame.

A real person crop is translated across a known vertical line over a synthetic
background, so the expected gate result is exactly N entries and 0 exits.

The crop and the background are both validated: the background must yield zero
person detections and the crop exactly one, otherwise the "expected" count would
be a fiction. (An earlier version of this generator used a blurred photo as the
backdrop; it contained three detectable people standing still near the gate
line, which made the test fail against correct code.)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT   = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "venv/Lib/site-packages/ultralytics/assets"


def _background(w: int, h: int) -> np.ndarray:
    """Neutral gradient floor/wall — nothing a person detector can latch onto."""
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        v = 58 + int(46 * y / h)
        bg[y, :] = (v, v - 4, v - 9)
    cv2.rectangle(bg, (0, int(h * 0.78)), (w, h), (74, 70, 66), -1)
    for x in range(0, w, 160):                      # faint floor tiling
        cv2.line(bg, (x, int(h * 0.78)), (x, h), (82, 78, 74), 1)
    return bg


def _person_crop(model=None) -> np.ndarray:
    """Highest-confidence single person from bus.jpg, cropped tight."""
    src = cv2.imread(str(ASSETS / "bus.jpg"))
    if src is None:
        raise SystemExit(f"sample image not found under {ASSETS}")
    from ultralytics import YOLO
    model = model or YOLO(str(ROOT / "yolov8s.pt"))
    res = model(src, classes=[0], conf=0.5, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        raise SystemExit("no person found in the sample image")
    confs = boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[int(confs.argmax())].astype(int)
    pad = 4
    crop = src[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    return cv2.resize(crop, (140, 380))


def build(out_path: str, n_people: int = 3, w: int = 1280, h: int = 720,
          fps: int = 25, frames: int = 170, verify: bool = True) -> None:
    from ultralytics import YOLO
    model  = YOLO(str(ROOT / "yolov8s.pt"))
    bg     = _background(w, h)
    person = _person_crop(model)

    if verify:
        n_bg = len(model(bg, classes=[0], conf=0.25, verbose=False)[0].boxes)
        n_pp = len(model(person, classes=[0], conf=0.25, verbose=False)[0].boxes)
        if n_bg != 0:
            raise SystemExit(f"background yields {n_bg} phantom detections")
        if n_pp != 1:
            raise SystemExit(f"person crop yields {n_pp} detections, expected 1")

    # Give each walker its own size, depth and colour. Pixel-identical clones at
    # the same height are an adversarial case for any tracker — with nothing to
    # distinguish them, ByteTrack swaps their IDs as they pass, which is a
    # property of the test footage rather than a defect in the counter. Real
    # people differ; the walkers here do too.
    variants = []
    for i in range(n_people):
        scale = 0.82 + 0.18 * i
        vp = cv2.resize(person, (int(140 * scale), int(380 * scale)))
        tint = [(1.0, 0.85, 0.85), (0.85, 1.0, 0.88), (0.88, 0.88, 1.0)][i % 3]
        vp = np.clip(vp.astype(np.float32) * np.array(tint), 0, 255).astype(np.uint8)
        variants.append(vp)

    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    baseline = int(h * 0.78) + 60
    stride = 34                       # frames between successive walkers
    span   = frames - stride * (n_people - 1) - 10
    for f in range(frames):
        canvas = bg.copy()
        for i in range(n_people):
            t = (f - i * stride) / float(span)
            if not (0.0 <= t <= 1.0):
                continue
            vp = variants[i]
            ph, pw = vp.shape[:2]
            y = baseline - ph - i * 26          # staggered depth
            x = int(-pw + t * (w + 2 * pw))
            x0, x1 = max(0, x), min(w, x + pw)
            if x1 <= x0 or y < 0 or y + ph > h:
                continue
            canvas[y:y + ph, x0:x1] = vp[:, x0 - x:x1 - x]
        vw.write(canvas)
    vw.release()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gate_clip.mp4"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    build(out, n)
    print(f"wrote {out} ({n} people crossing left->right)")
