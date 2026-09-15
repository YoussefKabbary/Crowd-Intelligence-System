"""Run a Crowd Master version unmodified and record exactly what its main window shows.

cv2.imshow is intercepted: frames for the main window are written to a video at
wall-clock pace instead of being shown, so nothing pops up on the desktop.
cv2.waitKey returns scripted key presses at set times, driving the app's own
keyboard handlers (zoom, enhance, speed, zones, heatmap, screenshot...).
"""
import importlib.util
import os
import sys
import time

SRC, VIDEO, OUT, DATA, DURATION = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
SCRIPT = [tuple(x.split("@")) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
SCRIPT = sorted((float(t), k) for k, t in SCRIPT)   # "key@seconds"

os.environ["CROWD_MASTER_VIDEO"] = VIDEO
os.environ["CROWD_MASTER_DATA_DIR"] = DATA
os.environ["CROWD_MASTER_API"] = "0"
os.makedirs(DATA, exist_ok=True)

import cv2  # noqa: E402

FPS = 24.0
state = {"writer": None, "t0": None, "written": 0, "last": None, "size": None, "keys": list(SCRIPT), "quit": False}
real_imshow, real_waitKey = cv2.imshow, cv2.waitKey


import queue, threading
_q = queue.Queue(maxsize=48)


def _writer():
    while True:
        item = _q.get()
        if item is None:
            break
        img, count = item
        if (img.shape[1], img.shape[0]) != state["size"]:
            img = cv2.resize(img, state["size"])
        for _ in range(count):
            state["writer"].write(img)


def rec_imshow(win, img):
    # Only copy the frame here; encoding happens on another thread so the
    # app's own display loop is not slowed down by the recording.
    if not str(win).lower().startswith("crowd master"):
        return
    now = time.perf_counter()
    if state["writer"] is None:
        h, w = img.shape[:2]
        state["size"] = (w, h)
        state["writer"] = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        state["t0"] = now
        state["thread"] = threading.Thread(target=_writer, daemon=True)
        state["thread"].start()
        print(f"[rec] first frame {w}x{h}", flush=True)
    due = int((now - state["t0"]) * FPS) + 1
    count = due - state["written"]
    if count > 0:
        state["written"] = due
        try:
            _q.put((img.copy(), count))
        except queue.Full:
            pass
    if now - state["t0"] >= DURATION:
        state["quit"] = True


def rec_waitKey(ms=0):
    if state["quit"]:
        return ord("q")
    if state["t0"] is not None and state["keys"]:
        el = time.perf_counter() - state["t0"]
        if el >= state["keys"][0][0]:
            t, k = state["keys"].pop(0)
            print(f"[rec] {el:5.1f}s key {k}", flush=True)
            return {"up": 2490368, "left": 2424832}.get(k, ord(k[0]))
    time.sleep(max(ms, 1) / 1000.0)
    return -1


def noop(*a, **k):
    return None


cv2.imshow = rec_imshow
cv2.waitKey = rec_waitKey
cv2.waitKeyEx = rec_waitKey
for name in ("namedWindow", "resizeWindow", "moveWindow", "setWindowProperty",
             "setMouseCallback", "setWindowTitle", "destroyWindow", "destroyAllWindows"):
    setattr(cv2, name, noop)
cv2.getWindowProperty = lambda *a, **k: 1.0

spec = importlib.util.spec_from_file_location("cm_rec", SRC)
cm = importlib.util.module_from_spec(spec)
sys.modules["cm_rec"] = cm
spec.loader.exec_module(cm)

cfg = cm.Config()
cfg.VIDEO_PATH = VIDEO
if hasattr(cfg, "API_ENABLED"):
    cfg.API_ENABLED = False
if hasattr(cm, "start_api"):
    cm.start_api = lambda *a, **k: None
try:
    cm.run(cfg)
except SystemExit:
    pass
finally:
    if state["writer"] is not None:
        _q.put(None)
        state["thread"].join()
        state["writer"].release()
    print(f"[rec] wrote {OUT}  {state['written']} frames  {state['written']/FPS:.1f}s", flush=True)
    os._exit(0)
