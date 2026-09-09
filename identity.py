# -*- coding: utf-8 -*-
"""Identity matching against a watchlist, gated on what the pixels can support.

The design problem this solves: face recognition, appearance re-identification
and gait all need different amounts of image detail, and a camera either
supplies that detail or it does not. Measured on a 640x360 CCTV clip in this
project, faces were 3.2 px between the eyes where reliable recognition needs
about 90 — a 28x shortfall no model can close. On a 4K camera at a doorway the
same scene would comfortably clear it.

So capability is measured per person, per frame, and the strongest method the
pixels actually support is the one that runs. Nothing is configured by hand and
nothing runs on detail it does not have.

    IOD  = inter-ocular distance in pixels, from pose keypoints
    H    = person bounding-box height in pixels

    face   IOD >= FACE_MIN_IOD (90)      identification
    reid   H   >= REID_MIN_H   (96)      "same person as before", not "who"
    gait   H   >= GAIT_MIN_H   (128) and a full unbroken walking cycle

A match is never an assertion. The engine produces candidates with scores for a
human to confirm or reject, records every decision, and takes no action itself.
That is a deliberate constraint: a false positive here means a person is flagged
as wanted, which is a different class of consequence from miscounting a crowd.
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
#  CAPABILITY GATE
# ══════════════════════════════════════════════════════════════
class Capability:
    """Which identity methods the available pixels can support."""

    FACE_MIN_IOD = 90.0     # px between the eyes for reliable identification
    FACE_WARN_IOD = 60.0    # below this, face results are not trustworthy
    REID_MIN_H = 96.0       # person height for a usable appearance embedding
    GAIT_MIN_H = 128.0      # person height for a usable silhouette
    GAIT_MIN_FRAMES = 25    # one walking cycle at ~25 fps

    __slots__ = ("iod", "height", "track_frames", "face", "reid", "gait")

    def __init__(self, iod: float, height: float, track_frames: int):
        self.iod = iod
        self.height = height
        self.track_frames = track_frames
        self.face = iod >= self.FACE_MIN_IOD
        self.reid = height >= self.REID_MIN_H
        self.gait = (height >= self.GAIT_MIN_H
                     and track_frames >= self.GAIT_MIN_FRAMES)

    @property
    def best(self) -> Optional[str]:
        """Strongest method the pixels support, or None."""
        if self.face:
            return "face"
        if self.reid:
            return "reid"
        if self.gait:
            return "gait"
        return None

    def reason(self) -> str:
        """Why identification is not possible, in terms a person can act on."""
        if self.best:
            return ""
        if self.iod and self.iod < self.FACE_MIN_IOD:
            need = self.FACE_MIN_IOD / max(self.iod, 0.1)
            return (f"faces are {self.iod:.1f}px between the eyes, "
                    f"need {self.FACE_MIN_IOD:.0f} ({need:.0f}x more detail); "
                    f"person height {self.height:.0f}px, "
                    f"need {self.REID_MIN_H:.0f} for appearance matching")
        return (f"person height {self.height:.0f}px, "
                f"need {self.REID_MIN_H:.0f} for appearance matching")


def measure_capability(box: np.ndarray,
                       keypoints: Optional[np.ndarray],
                       kp_conf: Optional[np.ndarray],
                       track_frames: int = 0) -> Capability:
    """Measure what this particular detection supports.

    keypoints are COCO pose order; 1 and 2 are the left and right eye.
    """
    height = float(box[3] - box[1])
    iod = 0.0
    if keypoints is not None and len(keypoints) > 2:
        ok = kp_conf is None or (kp_conf[1] > 0.3 and kp_conf[2] > 0.3)
        le, re = keypoints[1], keypoints[2]
        seen = not ((le[0] < 1 and le[1] < 1) or (re[0] < 1 and re[1] < 1))
        if ok and seen:
            iod = float(np.hypot(le[0] - re[0], le[1] - re[1]))
    return Capability(iod, height, track_frames)


# ══════════════════════════════════════════════════════════════
#  BACKENDS
# ══════════════════════════════════════════════════════════════
class Backend:
    """A source of comparable embeddings. Absent dependencies disable it."""

    name = "base"
    available = False

    def embed(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [0, 1]."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0) * 0.5 + 0.5)


class FaceBackend(Backend):
    """Face embeddings via InsightFace when installed.

    Deliberately not bundled: face identification is the highest-consequence
    part of this system, and requiring a separate, explicit install keeps it
    from being switched on without a decision.
    """

    name = "face"

    def __init__(self, device: str = "cuda"):
        self.app = None
        try:
            from insightface.app import FaceAnalysis
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if device == "cuda" else ["CPUExecutionProvider"])
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
            self.available = True
        except Exception:
            self.available = False

    def embed(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                          int(box[2]), int(box[3]))
        # Search the upper third of the body box; that is where the head is.
        crop = frame[y1:y1 + max(1, (y2 - y1) // 3), x1:x2]
        if crop.size == 0:
            return None
        faces = self.app.get(crop)
        if not faces:
            return None
        best = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        emb = getattr(best, "normed_embedding", None)
        return None if emb is None else np.asarray(emb, dtype=np.float32)

    @staticmethod
    def install_hint() -> str:
        return ("pip install insightface onnxruntime-gpu   "
                "(face identification is off until this is installed)")


class ReIDBackend(Backend):
    """Whole-body appearance embeddings from a purpose-trained ReID network.

    Answers "is this the same person I saw before", not "who is this". Robust to
    distance in a way face recognition is not, and useless across a change of
    clothes.
    """

    name = "reid"

    def __init__(self, model: str = "yolo26s-reid.onnx", device: str = "cuda"):
        self.model = None
        try:
            from ultralytics.trackers.utils.reid import ReID
            self.model = ReID(model, device=device)
            self.available = True
        except Exception:
            self.available = False

    def embed(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            xywh = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                              box[2] - box[0], box[3] - box[1]]], dtype=np.float32)
            feats = self.model(frame, xywh)
            if feats is None or len(feats) == 0 or feats[0] is None:
                return None
            return np.asarray(feats[0], dtype=np.float32).ravel()
        except Exception:
            return None


class GaitBackend(Backend):
    """Gait signature from silhouette dynamics. Experimental — off by default.

    Measured on this project's footage: only 44% of tracks survive an unbroken
    walking cycle and the median person is 94 px tall, so silhouettes are coarse.
    Loose clothing is gait recognition's documented worst case, which matters
    directly for the prayer-hall deployment this project targets.

    What is implemented here is a cheap descriptor over silhouette width
    oscillation — enough to corroborate an identity already established by
    another method, not to establish one. It is not a substitute for a trained
    gait network, and it should not be read as one.
    """

    name = "gait"

    def __init__(self):
        self.available = True
        self._seq: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

    def observe(self, track_id: int, box: np.ndarray) -> None:
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        if h > 1:
            self._seq[track_id].append(w / h)      # aspect oscillates with stride

    def embed(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        return None                                 # use signature() instead

    def signature(self, track_id: int) -> Optional[np.ndarray]:
        seq = self._seq.get(track_id)
        if seq is None or len(seq) < Capability.GAIT_MIN_FRAMES:
            return None
        a = np.asarray(seq, dtype=np.float32)
        a = a - a.mean()
        if np.allclose(a, 0):
            return None
        spec = np.abs(np.fft.rfft(a, n=64))[:16]    # stride frequency content
        n = np.linalg.norm(spec)
        return (spec / n).astype(np.float32) if n > 1e-9 else None

    def forget(self, track_id: int) -> None:
        self._seq.pop(track_id, None)


# ══════════════════════════════════════════════════════════════
#  WATCHLIST
# ══════════════════════════════════════════════════════════════
@dataclass
class WatchlistEntry:
    person_id: str
    label: str
    note: str = ""
    added: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)

    def add_embedding(self, method: str, vec: np.ndarray) -> None:
        self.embeddings.setdefault(method, []).append([float(v) for v in vec])

    def vectors(self, method: str) -> List[np.ndarray]:
        return [np.asarray(v, dtype=np.float32)
                for v in self.embeddings.get(method, [])]


@dataclass
class Candidate:
    """A possible match, for a person to confirm or reject. Never an assertion."""
    track_id: int
    person_id: str
    label: str
    method: str
    score: float
    capability: str
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    frame_id: int = 0
    confirmed: Optional[bool] = None


# ══════════════════════════════════════════════════════════════
#  ENGINE
# ══════════════════════════════════════════════════════════════
class IdentityEngine:
    """Watchlist matching, gated on measured capability, human-confirmed.

    Thresholds are starting points, not calibrated values. They should be set
    from a measured false-accept rate on the specific camera before anyone acts
    on the output; until that is done, treat every candidate as a suggestion.
    """

    THRESHOLDS = {"face": 0.62, "reid": 0.80, "gait": 0.90}

    def __init__(self, data_dir: str, device: str = "cuda",
                 enable_face: bool = True, enable_reid: bool = True,
                 enable_gait: bool = False, logger=None):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = logger
        self.watchlist: Dict[str, WatchlistEntry] = {}
        self.candidates: List[Candidate] = []
        self._recent: Dict[Tuple[int, str], float] = {}
        self._lock = threading.Lock()
        self._audit = self.dir / "identity_audit.csv"
        self._wl_path = self.dir / "watchlist.json"

        self.backends: Dict[str, Backend] = {}
        if enable_reid:
            b = ReIDBackend(device=device)
            if b.available:
                self.backends["reid"] = b
        if enable_face:
            b = FaceBackend(device=device)
            if b.available:
                self.backends["face"] = b
            elif self.log:
                self.log.info("Identity: face matching unavailable — %s",
                              FaceBackend.install_hint())
        if enable_gait:
            self.backends["gait"] = GaitBackend()

        self._load()
        if self.log:
            self.log.info("Identity: backends %s | watchlist %d",
                          sorted(self.backends) or "none", len(self.watchlist))

    # ── watchlist ────────────────────────────────────────────
    def enrol(self, person_id: str, label: str, images: List[np.ndarray],
              note: str = "") -> Dict[str, int]:
        """Register a person from reference images. Returns embeddings per method."""
        entry = self.watchlist.get(person_id) or WatchlistEntry(person_id, label, note)
        counts: Dict[str, int] = {}
        for img in images:
            box = np.array([0, 0, img.shape[1], img.shape[0]], dtype=np.float32)
            for name, backend in self.backends.items():
                if name == "gait":
                    continue                       # gait needs video, not stills
                vec = backend.embed(img, box)
                if vec is not None:
                    entry.add_embedding(name, vec)
                    counts[name] = counts.get(name, 0) + 1
        self.watchlist[person_id] = entry
        self._save()
        self._write_audit("enrol", person_id, label, "", 0.0,
                          f"{len(images)} images -> {counts}")
        return counts

    def remove(self, person_id: str) -> bool:
        e = self.watchlist.pop(person_id, None)
        if e:
            self._save()
            self._write_audit("remove", person_id, e.label, "", 0.0, "")
        return e is not None

    # ── matching ─────────────────────────────────────────────
    def check(self, frame: np.ndarray, box: np.ndarray, track_id: int,
              cap: Capability, frame_id: int = 0) -> Optional[Candidate]:
        """Test one detection against the watchlist.

        Returns a candidate for human review, or None. Returns None whenever the
        pixels do not support any method — silence is the correct output for a
        camera that cannot see well enough, rather than a low-confidence guess.
        """
        if not self.watchlist or cap.best is None:
            return None
        method = cap.best
        backend = self.backends.get(method)
        if backend is None:
            for alt in ("reid", "face"):            # fall back within capability
                if getattr(cap, alt, False) and alt in self.backends:
                    method, backend = alt, self.backends[alt]
                    break
        if backend is None:
            return None

        vec = (backend.signature(track_id) if method == "gait"
               else backend.embed(frame, box))
        if vec is None:
            return None

        best_id, best_label, best_score = None, "", 0.0
        for entry in self.watchlist.values():
            for ref in entry.vectors(method):
                s = Backend.similarity(vec, ref)
                if s > best_score:
                    best_id, best_label, best_score = entry.person_id, entry.label, s

        if best_id is None or best_score < self.THRESHOLDS.get(method, 0.9):
            return None

        key = (track_id, best_id)
        now = time.monotonic()
        with self._lock:
            if now - self._recent.get(key, -1e9) < 30.0:
                return None                         # one candidate per track per 30 s
            self._recent[key] = now
            cand = Candidate(track_id, best_id, best_label, method,
                             round(best_score, 4), cap.reason() or method, frame_id=frame_id)
            self.candidates.append(cand)
        self._write_audit("candidate", best_id, best_label, method, best_score,
                          f"track={track_id} frame={frame_id} "
                          f"iod={cap.iod:.1f} h={cap.height:.0f}")
        if self.log:
            self.log.warning(
                "Identity CANDIDATE (unconfirmed): %s via %s score=%.3f track=%d",
                best_label, method, best_score, track_id)
        return cand

    def confirm(self, index: int, is_match: bool, operator: str = "") -> bool:
        """Record a human decision. Nothing downstream should act before this."""
        with self._lock:
            if not 0 <= index < len(self.candidates):
                return False
            c = self.candidates[index]
            c.confirmed = is_match
        self._write_audit("confirmed" if is_match else "rejected",
                          c.person_id, c.label, c.method, c.score,
                          f"track={c.track_id} operator={operator or 'unspecified'}")
        return True

    def pending(self) -> List[Candidate]:
        with self._lock:
            return [c for c in self.candidates if c.confirmed is None]

    # ── persistence ──────────────────────────────────────────
    def _load(self) -> None:
        if not self._wl_path.exists():
            return
        try:
            for d in json.loads(self._wl_path.read_text(encoding="utf-8")):
                self.watchlist[d["person_id"]] = WatchlistEntry(**d)
        except Exception as e:
            if self.log:
                self.log.warning(f"Identity: watchlist load failed: {e}")

    def _save(self) -> None:
        try:
            self._wl_path.write_text(
                json.dumps([asdict(e) for e in self.watchlist.values()], indent=2),
                encoding="utf-8")
        except Exception as e:
            if self.log:
                self.log.warning(f"Identity: watchlist save failed: {e}")

    def _write_audit(self, action: str, person_id: str, label: str,
                     method: str, score: float, detail: str) -> None:
        """Append-only record of every enrolment, candidate and decision."""
        try:
            new = not self._audit.exists()
            with open(self._audit, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "action", "person_id", "label",
                                "method", "score", "detail"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), action,
                            person_id, label, method, f"{score:.4f}", detail])
        except Exception:
            pass
