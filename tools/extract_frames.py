"""Pull evenly-spaced frames out of a clip so they can be counted by hand.

This is step one of measuring detection accuracy. Nothing in the project
currently knows whether its counts are right — every benchmark so far measures
speed. Comparing against frames a person has counted is the only way to tell
whether a change helped.

    venv\\Scripts\\python.exe tools\\extract_frames.py myclip.mp4 --count 200

Writes to <video>_labels/:
    frame_000123.jpg ...     the frames to look at
    labels.csv               frame,count  — fill in the count column

Then:
    venv\\Scripts\\python.exe tests\\measure_accuracy.py myclip.mp4 \\
        <video>_labels/labels.csv

Counting 200 frames takes about an hour. Do it once on footage that looks like
what the system will actually watch, and every future change becomes measurable.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2


def extract(video: str, n: int, out_dir: Path, jpeg_quality: int = 92) -> int:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit("video reports no frame count; cannot sample evenly")
    n = min(n, total)
    # Evenly spaced rather than random: an even spread over the clip covers the
    # quiet and busy stretches in proportion, and the indices are reproducible.
    step = total / float(n)
    wanted = sorted({int(i * step) for i in range(n)})

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    idx = 0
    want_i = 0
    while want_i < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx == wanted[want_i]:
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            written.append(idx)
            want_i += 1
        idx += 1
    cap.release()

    labels = out_dir / "labels.csv"
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "count"])
        for i in written:
            w.writerow([i, ""])

    print(f"  video      : {video}")
    print(f"  frames     : {len(written)} of {total}")
    print(f"  images     : {out_dir}")
    print(f"  labels     : {labels}")
    print()
    print("  Open each image, count the people, and put the number in the")
    print("  'count' column. Leave a row blank to skip that frame.")
    return len(written)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--count", type=int, default=200,
                    help="how many frames to sample (default 200)")
    ap.add_argument("--out", default=None,
                    help="output directory (default <video>_labels)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.video).with_suffix("").parent / (
        Path(a.video).stem + "_labels")
    extract(a.video, a.count, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
