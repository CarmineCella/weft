#!/usr/bin/env python3
"""Verify recorded evidence and distributable listening assets without datasets."""
from array import array
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import statistics
import sys
import wave

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "book/experiments/v020"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    corpus = json.loads((EVIDENCE / "corpus.json").read_text())
    assert len(corpus["notes"]) == corpus["files"] == 1532
    paths = [n["path"] for n in corpus["notes"]]
    assert len(set(paths)) == len(paths)
    for split, count in (("train", 1433), ("validation", 75), ("too_short", 24)):
        expected = [n["path"] for n in corpus["notes"] if n["split"] == split]
        assert (EVIDENCE / (split + ".txt")).read_text().splitlines() == expected
        assert len(expected) == count
    held = {n["path"] for n in corpus["notes"] if n["split"] == "validation"}
    assert all(row[1] in held for row in corpus["listening_notes"])
    for part in ("mnist", "vae", "synthesis"):
        data = json.loads((EVIDENCE / (part + ".json")).read_text())
        assert data["status"] == "passed" and data["environment"]["source_tag"] == "v0.2.0"
        for cmd in data["commands"]:
            assert cmd["status"] == "passed" and cmd["returncode"] == 0
            assert digest(ROOT / cmd["log"]) == cmd["log_sha256"]
    results = json.loads((EVIDENCE / "results.json").read_text())
    assert {row["seed"] for row in results["mnist"]} == {1, 2, 3}
    for row in results["mnist"]:
        log = (EVIDENCE / "logs" / f'mnist_seed_{row["seed"]}.log').read_text()
        found = re.findall(r"final test accuracy: [0-9.]+\s+\((\d+) / (\d+) correct\)", log)
        assert len(found) == 1 and tuple(map(int, found[0])) == (row["test_correct"], row["test_total"])
        assert row["test_accuracy"] == row["test_correct"] / row["test_total"]
    accuracies = [row["test_accuracy"] for row in results["mnist"]]
    assert math.isclose(results["mnist_test_mean"], statistics.mean(accuracies), abs_tol=1e-12)
    assert math.isclose(results["mnist_test_sample_sd"], statistics.stdev(accuracies), abs_tol=1e-12)
    assert results["vae"]["epochs"] == 20 and results["synthesis"]["epochs"] == 40
    assets = results["listening_assets"]
    assert len(assets) == 13
    groups = {}
    for item in assets:
        path = ROOT / item["path"]
        assert path.is_file() and digest(path) == item["sha256"]
        with wave.open(str(path)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 44100)
            assert wav.getnframes() == item["samples"]
            values = array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            values.byteswap()
        rms = math.sqrt(math.fsum(float(v) ** 2 for v in values) / len(values)) / 32768
        peak = max(abs(v) for v in values) / 32768
        assert abs(20 * math.log10(rms) - item["rms_dbfs"]) < 1e-6
        assert abs(20 * math.log10(peak) - item["peak_dbfs"]) < 1e-6
        assert item["peak_dbfs"] < -0.99 and item["gain"] > 0
        if item["group"] != "gesture":
            groups.setdefault(item["group"], []).append(item["rms_dbfs"])
    assert all(max(levels) - min(levels) < 0.02 for levels in groups.values())
    for group in results["audio_measurements"].values():
        for measurements in group.values():
            assert all(math.isfinite(x) for x in measurements.values())
            assert measurements["gain_aligned_magnitude_residual"] >= 0
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("src", "href") and "://" not in value and not value.startswith("#"):
                    assert (ROOT / "notes/audio" / value).is_file(), value
    Links().feed((ROOT / "notes/audio/index.html").read_text())
    print("Recorded commands, data splits, numerical summaries, and 13 listening assets verified.")


if __name__ == "__main__":
    check()
