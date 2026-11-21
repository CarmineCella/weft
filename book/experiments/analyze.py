#!/usr/bin/env python3
"""Turn the fixed book runs into measured figures and listening copies."""
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import statistics
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/book_v020/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/book_v020/cache"))
sys.path.insert(0, str(ROOT / "book"))
import make_figures as figures
import matplotlib.pyplot as plt

EVIDENCE = ROOT / "book/experiments/v020"
WORK = ROOT / "outputs/book_v020"
AUDIO = ROOT / "notes/audio/v020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    sr, data = wavfile.read(path)
    if data.ndim != 1 or data.dtype != np.int16 or sr != 44100:
        raise ValueError(f"expected mono PCM16 / 44100: {path}")
    return data.astype(np.float64) / 32768.0


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(data).all() or np.max(np.abs(data)) > 1.000001:
        raise ValueError(f"invalid or clipping audio: {path}")
    wavfile.write(path, 44100, np.rint(data * 32767).astype(np.int16))


def rows(name, columns):
    text = (EVIDENCE / "logs" / name).read_text()
    pattern = re.compile(r"^\s*(\d+)\s+" + r"\s+".join([r"([0-9.]+)"] * (columns - 1)) + r"\s*[s]?\s*$", re.M)
    result = np.array([[float(v) for v in match] for match in pattern.findall(text)])
    if result.ndim != 2 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError(f"missing or invalid epoch rows in {name}")
    return text, result


def spectra(x, n=4096, hop=1024):
    count = 1 + (len(x) - n) // hop
    if count < 1:
        raise ValueError("audio too short for spectral comparison")
    idx = hop * np.arange(count)[:, None] + np.arange(n)[None, :]
    return np.fft.rfft(x[idx] * np.hanning(n), axis=1)


def compare(target, estimate):
    n = min(len(target), len(estimate))
    a, b = np.abs(spectra(target[:n])), np.abs(spectra(estimate[:n]))
    den = np.sum(a * a)
    gain = np.sum(a * b) / max(np.sum(b * b), 1e-30)
    return {"common_samples": n, "target_samples": len(target), "output_samples": len(estimate),
            "magnitude_residual": float(np.sum((a - b) ** 2) / den),
            "gain_aligned_magnitude_residual": float(np.sum((a - gain * b) ** 2) / den),
            "magnitude_alignment_gain": float(gain),
            "raw_output_rms_dbfs": db(rms(estimate)),
            "raw_output_peak_dbfs": db(np.max(np.abs(estimate))),
            "raw_output_rail_fraction": float(np.mean(np.abs(estimate) >= 32767 / 32768))}


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def db(amplitude):
    return float(20 * np.log10(max(float(amplitude), 1e-15)))


def roundtrip(x):
    n, hop = 4096, 2048
    w = np.hanning(n)
    count = 1 + (len(x) - n) // hop
    out, norm = np.zeros_like(x), np.zeros_like(x)
    for f in range(count):
        start = f * hop
        frame = np.fft.irfft(np.fft.rfft(x[start:start+n] * w), n=n)
        out[start:start+n] += frame * w
        norm[start:start+n] += w * w
    valid = norm > 1e-8
    out[valid] /= norm[valid]
    return out, {"frames": count, "covered_samples": int(np.sum(valid)),
                 "last_frame_end": (count-1)*hop+n,
                 "covered_sample_max_error": float(np.max(np.abs(out[valid]-x[valid])))}


def listening_group(label, items):
    # One common RMS target; a peak constraint may lower it for the entire group.
    target = min([10 ** (-24 / 20)] + [10 ** (-1 / 20) * rms(x) / np.max(np.abs(x)) for _, x in items])
    assets = []
    for name, x in items:
        gain = target / rms(x)
        path = AUDIO / (label + "_" + name + ".wav")
        save(path, gain * x)
        actual = read(path)
        assets.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "group": label,
                       "variant": name, "samples": len(actual), "sample_rate": 44100,
                       "gain": gain, "rms_dbfs": db(rms(actual)), "peak_dbfs": db(np.max(np.abs(actual)))})
    if max(a["rms_dbfs"] for a in assets) - min(a["rms_dbfs"] for a in assets) > 0.02:
        raise ValueError("listening copies failed RMS matching")
    return assets


def make_plots(mnist, vae, synth, target, variants):
    # Use the standard chapter-12 entry point so its regeneration stays consistent.
    figures.ch12()
    fig, ax = plt.subplots(figsize=(figures.W, 2.3))
    ax.plot(vae[:, 0], vae[:, 1], color=figures.BLUE, label="reconstruction")
    ax.plot(vae[:, 0], vae[:, 2], color=figures.ORANGE, ls="--", label="KL")
    ax.set(xlabel="epoch", ylabel="loss per frame")
    ax.legend()
    fig.tight_layout()
    figures.save(fig, "19_audio_training_v020")

    fig, ax = plt.subplots(figsize=(figures.W, 2.3))
    ax.plot(synth[:, 0], synth[:, 1], color=figures.BLUE, label="training crops")
    ax.plot(synth[:, 0], synth[:, 2], color=figures.ORANGE, ls="--", label="validation crops")
    ax.set(xlabel="epoch", ylabel="spectral loss")
    ax.legend()
    fig.tight_layout()
    figures.save(fig, "24_network_training_v020")

    signals = [("recording", target), ("VAE + input phase", variants["vae"]),
               ("directly fitted controls", variants["fit"]), ("network controls", variants["network"])]
    fig, axes = plt.subplots(4, 1, figsize=(figures.W, 5.3), sharex=True, sharey=True)
    reference = np.max(np.abs(spectra(target, 2048, 512)))
    for ax, (label, signal) in zip(axes, signals):
        # A single waveform-RMS gain for display, matching the listening intention.
        signal = signal * rms(target) / rms(signal)
        mag = np.abs(spectra(signal, 2048, 512)).T
        image = 20 * np.log10(np.maximum(mag / reference, 1e-5))
        ax.imshow(image, origin="lower", aspect="auto", cmap="Greys", vmin=-70, vmax=0,
                  extent=(0, (mag.shape[1]-1)*512/44100, 0, 22050), interpolation="nearest")
        ax.set_ylim(0, 6000)
        ax.set_ylabel("Hz")
        ax.set_title(label, loc="left", fontsize=8, pad=3)
        ax.grid(False)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout(h_pad=0.65)
    figures.save(fig, "00_cello_comparison_v020")


def make_player(assets):
    names = {"original": "Recording", "stft": "STFT round trip", "vae": "Dense VAE, input phase",
             "fit": "Directly fitted synthesiser", "network": "Network-controlled synthesiser",
             "phrase": "Prescribed eight-note flute phrase", "morph": "Prescribed flute-to-cello morph"}
    groups = {"cello": "1. Cello A4, mezzoforte", "trumpet": "2. Trumpet A4, pianissimo",
              "flute": "3. Flute D5, pianissimo", "gesture": "4. Controls without a target recording"}
    body = []
    for group, title in groups.items():
        cards = []
        for item in assets:
            if item["group"] != group:
                continue
            source = "v020/" + Path(item["path"]).name
            cards.append(f'<article><h3>{escape(names[item["variant"]])}</h3>'
                         f'<audio controls preload="none" src="{escape(source)}"></audio>'
                         f'<p><a href="{escape(source)}" download>Download WAV</a> · '
                         f'{item["samples"]/44100:.2f} s</p></article>')
        body.append(f'<section><h2>{escape(title)}</h2><div class="clips">' + "".join(cards) + '</div></section>')
    html = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Designing Musical Networks — listening examples</title>
<style>
body{margin:2rem auto;padding:0 1.2rem;max-width:68rem;background:#faf9f5;color:#202520;font:17px/1.55 system-ui,sans-serif}
h1{font:2.25rem/1.15 Georgia,serif;max-width:24ch}h2{font:1.5rem Georgia,serif;margin-top:2.4rem}
h3{font-size:1rem;margin:0 0 .8rem}a{color:#245a75}section{border-top:1px solid #d9ddd4}
.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,19rem),1fr));gap:1rem}
article{padding:1rem;background:white;border:1px solid #d9ddd4;border-radius:.35rem}audio{width:100%}
article p{font-size:.85rem;margin-bottom:0}.note{max-width:75ch}footer{border-top:1px solid #d9ddd4;margin-top:3rem;padding-top:1rem;font-size:.88rem}
</style>
<main><h1>Listen before building</h1>
<p class="note">Companion examples for <em>Designing Musical Networks</em>, rerun with weft v0.2.0.
Start with the recording, then compare the attack, sustained partials, breath/bow noise, and release.
The cello includes a transform-only reference and a direct per-note fit.</p>
<p class="note">Copies in each recording group have matched whole-clip RMS and a common peak constraint.
This does not equalize perceived loudness. The VAE application already normalizes and fades its output;
its original processing has not been undone. Training and raw-output measurements are documented in the
<a href="../../book/experiments/README.md">experiment protocol</a>.</p>
''' + "\n".join(body) + '''
<footer><p>All three recordings were reserved for validation before training. The direct fit uses its own target.
The phrase and morph follow explicit controls; the network does not compose their score or trajectory.</p>
<p>Source recordings: TinySOL, derived from Studio On Line, recorded at Ircam. Dataset creators:
Carmine-Emanuele Cella, Daniele Ghisi, Vincent Lostanlen, Fabien Lévy, Joshua Fineberg, and Yan Maresz.
<a href="https://www.orch-idea.org/datasets/">Orchidea distribution</a> ·
<a href="https://zenodo.org/records/3685367">TinySOL credits and terms</a> ·
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
Copies are RMS-scaled; the other files are transformations or synthesized outputs. See
<a href="README.md">the file guide</a> for source paths and processing.</p></footer></main></html>
'''
    (ROOT / "notes/audio/index.html").write_text(html)


def main():
    for part in ("mnist", "vae", "synthesis"):
        record = json.loads((EVIDENCE / (part + ".json")).read_text())
        if record.get("status") != "passed":
            raise RuntimeError(f"{part} is incomplete")
    inventory = json.loads((EVIDENCE / "corpus.json").read_text())
    mnist = []
    for seed in (1, 2, 3):
        text, table = rows(f"mnist_seed_{seed}.log", 6)
        assert len(table) == 10
        best = table[np.argmin(table[:, 3])]
        assert np.sum(table[:, 3] == best[3]) == 1, "selected epoch is ambiguous at logged precision"
        match = re.search(r"final test accuracy: ([0-9.]+)\s+\((\d+) / (\d+) correct\)", text)
        correct, total = int(match[2]), int(match[3])
        mnist.append({"seed": seed, "selected_epoch": int(best[0]),
                      "validation_loss": float(best[3]), "validation_accuracy": float(best[4]),
                      "test_correct": correct, "test_total": total, "test_accuracy": correct / total,
                      "training_seconds": float(re.search(r"total training time: ([0-9.]+)s", text)[1])})
    vae_text, vae = rows("vae_train.log", 5)
    _, synth = rows("synth_train.log", 4)
    assert len(vae) == 20 and len(synth) == 40
    results = {"protocol": "book-v020-1", "mnist": mnist,
               "mnist_test_mean": statistics.mean(r["test_accuracy"] for r in mnist),
               "mnist_test_sample_sd": statistics.stdev(r["test_accuracy"] for r in mnist),
               "vae": {"epochs": 20, "training_frames": int(re.search(r"frames:\s+(\d+)", vae_text)[1]),
                       "final_reconstruction": float(vae[-1, 1]), "final_kl": float(vae[-1, 2]),
                       "training_seconds": float(re.search(r"total training time: ([0-9.]+)s", vae_text)[1])},
               "synthesis": {"epochs": 40, "final_training_loss": float(synth[-1, 1]),
                             "final_validation_loss": float(synth[-1, 2]), "training_seconds": float(synth[-1, 3])},
               "audio_measurements": {}, "listening_assets": []}
    for name, rel, _, _ in inventory["listening_notes"]:
        original = read(ROOT / "data/TinySOL_2020" / rel)
        variants = {"vae": read(WORK / "vae" / (name + ".wav")),
                    "network": read(WORK / "synthesis" / (name + ".wav"))}
        if name == "cello":
            transformed, details = roundtrip(original)
            path = WORK / "reference/cello_stft.wav"
            save(path, transformed)
            variants = {"stft": read(path), "vae": variants["vae"],
                        "fit": read(WORK / "synthesis/cello_fit_resynth.wav"), "network": variants["network"]}
            results["stft_reference"] = details
            make_plots(mnist, vae, synth, original, variants)
        results["audio_measurements"][name] = {key: compare(original, value) for key, value in variants.items()}
        results["listening_assets"] += listening_group(name, [("original", original), *variants.items()])
    for gesture in ("phrase", "morph"):
        x = read(WORK / "synthesis" / (gesture + ".wav"))
        results["listening_assets"] += listening_group("gesture", [(gesture, x)])
    assert all(np.isfinite(list(r.values())).all() for d in results["audio_measurements"].values() for r in d.values())
    results["analysis"] = {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                           "script_sha256": sha(Path(__file__))}
    (EVIDENCE / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    make_player(results["listening_assets"])
    print(json.dumps({key: results[key] for key in ("mnist", "mnist_test_mean", "mnist_test_sample_sd", "vae", "synthesis", "audio_measurements")}, indent=2))


if __name__ == "__main__":
    main()
