#!/usr/bin/env python3
"""Run real example pipelines on small synthetic fixtures, without downloads."""
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/examples"


def run(name, *args, cwd, succeeds=True):
    result = subprocess.run([str(BIN / name), *map(str, args)], cwd=cwd,
                            text=True, capture_output=True, timeout=120)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"{name} exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def write_idx(directory, prefix, count):
    images = bytearray()
    labels = bytearray()
    for n in range(count):
        label = n % 10
        labels.append(label)
        images.extend(220 if (i + label * 17) % 37 < 6 else 10 for i in range(784))
    (directory / f"{prefix}-images-idx3-ubyte").write_bytes(
        struct.pack(">4I", 2051, count, 28, 28) + images)
    (directory / f"{prefix}-labels-idx1-ubyte").write_bytes(
        struct.pack(">2I", 2049, count) + labels)


def tone(path, rate, frequency):
    values = [int(6000 * math.sin(2 * math.pi * frequency * n / rate)) for n in range(8192)]
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(struct.pack(f"<{len(values)}h", *values))


def main():
    with tempfile.TemporaryDirectory(prefix="weft-smoke-") as temporary:
        work = Path(temporary)
        data = work / "mnist"
        data.mkdir()
        write_idx(data, "train", 40)
        write_idx(data, "t10k", 10)
        a, b = work / "first.wnet", work / "second.wnet"
        first = run("mnist_classifier", data, 42, a, cwd=work)
        run("mnist_classifier", data, 42, b, cwd=work)
        assert a.read_bytes() == b.read_bytes(), "same-seed training did not reproduce weights"
        assert first.stdout.count("final test accuracy:") == 1
        assert "validation loss" in first.stdout and "val loss" in first.stdout
        print("MNIST: reproducible training, validation checkpoint, one final test evaluation")
        iris = run("iris_classifier", cwd=work)
        assert "validation" in iris.stdout and iris.stdout.count("final test accuracy:") == 1
        sounds = work / "sounds"
        sounds.mkdir()
        for n in range(20):
            tone(sounds / f"{'low' if n % 2 else 'high'}-{n}.wav",
                 44100, 220 if n % 2 else 880)
        features = work / "mfcc.feat"
        run("extract_features", sounds, features, "mfcc", "averaged", cwd=work)
        classified = run("sound_classifier", features, cwd=work)
        assert "val acc" in classified.stdout and classified.stdout.count("final test accuracy:") == 1
        print("Audio classifier: extraction, file-aware validation, final confusion matrix")
        spectra = work / "spectra.feat"
        run("extract_features", sounds, spectra, "logmag", "per_frame", cwd=work)
        model = work / "audio"
        run("train_audio_vae", spectra, model, 1, 1, cwd=work)
        source = next(sounds.glob("*.wav"))
        output = work / "remapped.wav"
        run("vae_remap", model, source, output, cwd=work)
        assert output.is_file()
        wrong_rate = work / "wrong-rate.wav"
        tone(wrong_rate, 48000, 440)
        rejected = run("vae_remap", model, wrong_rate, work / "invalid.wav", cwd=work, succeeds=False)
        assert "44100" in rejected.stderr
        print("Audio VAE: metadata-checked training/loading/remapping and sample-rate rejection")
    print("All example smoke checks passed.")


if __name__ == "__main__":
    main()
