import importlib.util
import json
from pathlib import Path

import pytest

from safefloor.config import DATA_DIR
from safefloor.evaluation import compute_metrics, load_labels, make_meta, render_markdown, run_eval
from safefloor.llm import MockLLM
from safefloor.perception import build_prompt
from safefloor.store import load_zones

ZONES = load_zones(DATA_DIR / "zones.json")
ROOT = Path(__file__).resolve().parent.parent


def row(zone, truth, pred, image="x.jpg"):
    return {"image": image, "zone_id": zone, "truth": truth, "pred": pred, "summary": "s", "error": None, "note": ""}


def test_metric_arithmetic_by_hand():
    rows = [
        row("dock-a", ["ppe-vest"], ["ppe-vest"], "r1"),
        row("dock-a", ["ppe-vest", "forklift-gap"], ["forklift-gap"], "r2"),
        row("dock-a", [], ["ppe-vest"], "r3"),
        row("dock-a", [], [], "r4"),
        row("aisle-7", [], [], "r5"),
    ]
    m = compute_metrics(rows, ZONES)
    ppe, fork = m["rules"]["ppe-vest"], m["rules"]["forklift-gap"]
    assert (ppe["tp"], ppe["fp"], ppe["fn"], ppe["tn"]) == (1, 1, 1, 1)
    assert (ppe["precision"], ppe["recall"], ppe["f1"]) == (0.5, 0.5, 0.5)
    assert (fork["tp"], fork["fp"], fork["fn"], fork["tn"]) == (1, 0, 0, 3)
    assert (fork["precision"], fork["recall"], fork["f1"]) == (1.0, 1.0, 1.0)
    o = m["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (2, 1, 1)
    assert o["precision"] == pytest.approx(2 / 3) and o["recall"] == pytest.approx(2 / 3) and o["f1"] == pytest.approx(2 / 3)
    assert m["clean_frames"] == 3 and m["clean_frames_flagged"] == 1
    assert m["clean_false_alarm_rate"] == pytest.approx(1 / 3)
    assert [r["image"] for r in m["misses"]] == ["r2"] and [r["image"] for r in m["false_positives"]] == ["r3"]


def test_zero_denominators_are_none_not_zero():
    m = compute_metrics([row("aisle-7", [], [])], ZONES)
    assert m["rules"]["clear-aisle"]["precision"] is None and m["rules"]["clear-aisle"]["recall"] is None
    assert m["overall"]["f1"] is None and m["clean_false_alarm_rate"] == 0.0
    assert compute_metrics([], ZONES)["clean_false_alarm_rate"] is None
    assert "n/a" in render_markdown(m, make_meta("v", "zone-rules", False))


def test_both_prompt_modes_build_the_expected_prompt():
    zone = ZONES["dock-a"]
    full = build_prompt("cam-01", zone)
    bare = build_prompt("cam-01", zone, "rule-ids-only")
    assert "High-visibility vest required" in full and "- ppe-vest (high): " in full
    assert "- ppe-vest (high)\n" in bare and "High-visibility vest" not in bare and "forklift-gap (critical)" in bare
    assert full == build_prompt("cam-01", zone, "zone-rules")


@pytest.fixture
def labelled(tmp_path):
    (tmp_path / "frames").mkdir()
    labels = []
    for i, (cam, truth) in enumerate([("cam-01", ["ppe-vest"]), ("cam-01", []), ("cam-02", ["clear-aisle"])]):
        (tmp_path / "frames" / f"{i}.jpg").write_bytes(b"\xff\xd8fake")
        labels.append({"image": f"frames/{i}.jpg", "camera": cam, "violations": truth})
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return path


def test_run_eval_with_mock_model_in_both_modes(labelled):
    labels = load_labels(labelled)
    for mode in ("zone-rules", "rule-ids-only"):
        rows = run_eval(MockLLM(), ZONES, labels, labelled.parent, mode)
        # the mock flags the first rule of each zone on every frame
        assert [r["pred"] for r in rows] == [["ppe-vest"], ["ppe-vest"], ["ppe-helmet"]]
    m = compute_metrics(rows, ZONES)
    assert m["overall"]["tp"] == 1 and m["clean_frames_flagged"] == 1
    with pytest.raises(ValueError):
        run_eval(MockLLM(), ZONES, labels, labelled.parent, "nope")


def test_bad_labels_are_rejected(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="do not exist"):
        run_eval(MockLLM(), ZONES, [{"image": "a.jpg", "camera": "cam-01", "violations": ["no-smoking"]}], tmp_path)
    with pytest.raises(ValueError, match="unknown camera"):
        run_eval(MockLLM(), ZONES, [{"image": "a.jpg", "camera": "cam-99", "violations": []}], tmp_path)
    bad = tmp_path / "l.json"
    bad.write_text('[{"image": "a.jpg"}]')
    with pytest.raises(ValueError, match="missing"):
        load_labels(bad)


def test_failed_model_call_counts_as_no_prediction(tmp_path):
    class Boom:
        def vision(self, *a, **k):
            raise TimeoutError("slow")

    (tmp_path / "a.jpg").write_bytes(b"x")
    rows = run_eval(Boom(), ZONES, [{"image": "a.jpg", "camera": "cam-01", "violations": ["ppe-vest"]}], tmp_path)
    assert rows[0]["pred"] == [] and "TimeoutError" in rows[0]["error"]
    assert compute_metrics(rows, ZONES)["errors"] == 1


def load_script():
    spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mock_guard_refuses_then_labels_the_dry_run(labelled, tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("SAFEFLOOR_MOCK", raising=False)
    script, out = load_script(), tmp_path / "out" / "EVAL.md"
    assert script.main(["--labels", str(labelled), "--out", str(out)]) == 2
    assert not out.exists()
    assert script.main(["--labels", str(labelled), "--out", str(out), "--allow-mock"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("> **MOCK RUN, not a measurement.**") and "zone-rules" in text
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["meta"]["mock"] is True


def test_example_labels_file_is_valid():
    labels = load_labels(DATA_DIR / "eval" / "labels.example.json")
    by_camera = {z.camera: z for z in ZONES.values()}
    for item in labels:
        assert set(item["violations"]) <= {r["id"] for r in by_camera[item["camera"]].rules}
