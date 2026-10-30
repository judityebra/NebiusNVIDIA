import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from safefloor import app as app_module
from safefloor.app import create_app
from safefloor.config import DATA_DIR, Settings
from safefloor.events import build_events, fmt_t
from safefloor.video import sample_frames


def make_video(path, seconds=6, fps=10, size=(640, 360)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), (i * 4) % 255, dtype=np.uint8)
        cv2.putText(frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        w.write(frame)
    w.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.avi")


def test_sampling_count_times_and_resize(video):
    frames = sample_frames(video, fps=1.0, max_side=320)
    assert [t for t, _ in frames] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    img = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 320 and img.shape[1] > img.shape[0]
    assert len(sample_frames(video, fps=2.0)) == 12
    assert len(sample_frames(video, fps=2.0, max_frames=4)) == 4


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError):
        sample_frames(bad)
    with pytest.raises(ValueError):
        sample_frames(tmp_path / "missing.avi")


def frame(t, *rules, summary=None, conf=0.6):
    return {
        "t": float(t),
        "summary": summary or f"frame {t}",
        "objects": [f"o{t}"],
        "violations": [{"rule_id": r, "confidence": conf} for r in rules],
        "jpeg": f"jpeg{t}".encode(),
    }


def test_gap_of_one_frame_is_bridged():
    ev = build_events([frame(0, "a"), frame(1), frame(2, "a")])
    assert len(ev) == 1 and (ev[0]["start_t"], ev[0]["end_t"], ev[0]["frame_count"]) == (0.0, 2.0, 3)


def test_gap_of_two_frames_splits():
    ev = build_events([frame(0, "a"), frame(1), frame(2), frame(3, "a")])
    assert [(e["start_t"], e["end_t"]) for e in ev] == [(0.0, 0.0), (3.0, 3.0)]


def test_overlapping_rules_chain_into_one_event_with_union():
    frames = [frame(0, "a", conf=0.5), frame(1, "a", "b", conf=0.7), frame(2, "b", conf=0.9)]
    (ev,) = build_events(frames)
    assert ev["rules"] == {"a": 0.7, "b": 0.9}
    assert ev["jpeg"] == b"jpeg1"  # peak = highest summed confidence (0.7 + 0.7)


def test_unrelated_rules_do_not_merge():
    ev = build_events([frame(0, "a"), frame(1, "b")])
    assert [list(e["rules"]) for e in ev] == [["a"], ["b"]]


def test_peak_frame_summary_and_range():
    ev = build_events([frame(7, "a", conf=0.4, summary="early"), frame(8, "a", conf=0.9, summary="worst"), frame(12, "a", conf=0.5)])
    assert len(ev) == 1  # consecutive sampled frames, whatever their timestamps
    assert ev[0]["summary"] == "00:07 - 00:12: worst" and ev[0]["jpeg"] == b"jpeg8"
    assert fmt_t(75) == "01:15"


def test_clean_clip_gives_one_negative_with_middle_frame():
    ev = build_events([frame(0), frame(1), frame(2), frame(3), frame(4)])
    assert len(ev) == 1 and ev[0]["violations"] == [] and ev[0]["jpeg"] == b"jpeg2"
    assert "No violations in clip" in ev[0]["summary"] and ev[0]["frame_count"] == 5
    assert build_events([]) == []


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        None, "http://unused/v1/", "m", "v", tmp_path / "app.db", DATA_DIR / "zones.json", True,
        frames_dir=tmp_path / "frames",
    )
    return TestClient(create_app(settings))


def post_clip(client, path, **data):
    with open(path, "rb") as fh:
        return client.post("/api/clips", data={"camera": "cam-01", **data}, files={"video": ("clip.avi", fh, "video/x-msvideo")})


def test_clip_endpoint_in_mock_mode_gives_one_event(client, video):
    r = post_clip(client, video, ts="2026-10-30T10:00:00Z")
    assert r.status_code == 200
    (obs,) = r.json()
    assert obs["ts"] == "2026-10-30T10:00:00Z" and obs["end_ts"] == "2026-10-30T10:00:05Z"
    assert obs["frame_count"] == 6 and obs["clip"] == "clip.avi" and obs["source"] == "mock"
    assert [v["rule_id"] for v in obs["violations"]] == ["ppe-vest"] and obs["summary"].startswith("00:00 - 00:05:")
    assert client.get(f"/api/observations/{obs['id']}/frame").content[:2] == b"\xff\xd8"


def test_clip_fps_is_clamped(client, video):
    (obs,) = post_clip(client, video, fps="50").json()
    assert obs["frame_count"] == 12  # clamped to 2 fps over 6 s


def test_oversize_and_corrupt_uploads(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_CLIP_BYTES", 1000)
    assert post_clip(client, make_video(tmp_path / "big.avi")).status_code == 413
    monkeypatch.undo()
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    assert post_clip(client, junk).status_code == 400
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert post_clip(client, empty).status_code == 400
    r = client.post("/api/clips", data={"camera": "cam-99"}, files={"video": ("c.avi", b"x", "video/x-msvideo")})
    assert r.status_code == 404


def test_all_frames_failing_is_502_and_nothing_is_stored(tmp_path, video, monkeypatch):
    class Boom:
        def vision(self, *a, **k):
            raise TimeoutError("slow")

    settings = Settings(None, "http://unused/v1/", "m", "v", tmp_path / "x.db", DATA_DIR / "zones.json", True, frames_dir=tmp_path / "f")
    monkeypatch.setattr(app_module, "make_llm", lambda s: Boom())
    client = TestClient(create_app(settings))
    before = len(client.get("/api/observations").json())
    assert post_clip(client, video).status_code == 502
    assert len(client.get("/api/observations").json()) == before


def test_partial_frame_failures_are_skipped(tmp_path, video, monkeypatch):
    from safefloor.llm import MockLLM

    class Flaky(MockLLM):
        n = 0

        def vision(self, prompt, image_b64, mime="image/jpeg"):
            Flaky.n += 1
            if Flaky.n % 2 == 0:
                raise TimeoutError("slow")
            return super().vision(prompt, image_b64, mime)

    settings = Settings(None, "http://unused/v1/", "m", "v", tmp_path / "y.db", DATA_DIR / "zones.json", True, frames_dir=tmp_path / "f")
    monkeypatch.setattr(app_module, "make_llm", lambda s: Flaky())
    client = TestClient(create_app(settings))
    r = post_clip(client, video)
    assert r.status_code == 200 and int(r.headers["X-Frames-Skipped"]) > 0 and r.json()


def test_vision_mode_other_than_per_frame_is_rejected(monkeypatch):
    monkeypatch.setenv("SAFEFLOOR_VISION_MODE", "multi_image")
    with pytest.raises(ValueError):
        Settings.from_env()
