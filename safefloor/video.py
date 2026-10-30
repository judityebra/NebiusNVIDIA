"""Sample frames from a video clip for the vision model."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

MAX_CLIP_SECONDS = 60.0
DEFAULT_FPS = 25.0  # used when the container reports no frame rate


def sample_frames(
    path: str | Path, fps: float = 1.0, max_frames: int = 30, max_side: int = 768
) -> list[tuple[float, bytes]]:
    """Return ``[(t_seconds, jpeg_bytes)]`` sampled at ``fps`` from the first 60 s of the clip.

    Frames are resized so the long side is at most ``max_side``. Raises ``ValueError`` when the
    file cannot be opened or holds no readable frame.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("cannot open video")
    try:
        native = cap.get(cv2.CAP_PROP_FPS)
        if not native or math.isnan(native) or native <= 0:
            native = DEFAULT_FPS
        step = 1.0 / max(fps, 1e-6)
        next_t = 0.0
        out: list[tuple[float, bytes]] = []
        index = 0
        while len(out) < max_frames:
            t = index / native
            if t >= MAX_CLIP_SECONDS or not cap.grab():
                break
            if t + 1e-9 >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    out.append((round(t, 3), _jpeg(frame, max_side)))
                    next_t += step
            index += 1
    finally:
        cap.release()
    if not out:
        raise ValueError("no readable frames in video")
    return out


def _jpeg(frame, max_side: int) -> bytes:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode frame")
    return buf.tobytes()
