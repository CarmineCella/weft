#!/usr/bin/env python3
"""Run the fixed v0.2.0 book protocol; see README.md beside this file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "book/experiments/v020"
WORK = ROOT / "outputs/book_v020"
CORPUS = ROOT / "data/TinySOL_2020"
MNIST_SEEDS = (1, 2, 3)
VAE_EPOCHS = 20
SYNTH_EPOCHS = 40
SYNTH_BATCH = 14
LISTENING = (
    ("cello", "Strings/Vc/ordinario/Vc-ord-A4-mf-1c.wav", "A4", "Vc"),
    ("trumpet", "Brass/TpC/ordinario/TpC-ord-A4-pp.wav", "A4", "TpC"),
    ("flute", "Winds/Fl/ordinario/Fl-ord-D5-pp.wav", "D5", "Fl"),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def utc():
    return datetime.now(timezone.utc).isoformat()


def environment(compiler, backend):
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            cpu = result.stdout.strip()
    return {
        "source_tag": "v0.2.0", "source_revision": git("rev-parse", "v0.2.0^{}"),
        "checkout_revision": git("rev-parse", "HEAD"), "platform": platform.platform(),
        "cpu": cpu, "logical_cpus": os.cpu_count(), "compiler": version,
        "backend": backend, "python": platform.python_version(),
        "thread_environment": {key: "1" for key in
                               ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "cpp_sources_sha256": {relative(p): digest(p) for folder in ("src", "examples")
                               for p in sorted((ROOT / folder).iterdir()) if p.suffix in (".h", ".cpp")},
    }


class Run:
    def __init__(self, name, args):
        self.args = args
        self.path = EVIDENCE / (name + ".json")
        self.data = {"protocol": "book-v020-1", "started_utc": utc(),
                     "environment": environment(args.compiler, args.backend), "commands": []}
        self.env = os.environ.copy()
        self.env.update(self.data["environment"]["thread_environment"])
        self.name = name

    def command(self, key, argv):
        # Arguments in the record are relative to the repository root.
        argv = [str(arg) for arg in argv]
        argv = [str(Path(arg).relative_to(ROOT)) if arg.startswith(str(ROOT) + "/") else arg
                for arg in argv]
        log = EVIDENCE / "logs" / (key + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        item = {"argv": argv, "shell": shlex.join(argv), "cwd": ".",
                "log": relative(log), "started_utc": utc(), "status": "running"}
        self.data["commands"].append(item)
        write_json(self.path, self.data)
        print("RUN", shlex.join(argv), flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(argv, cwd=ROOT, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
        item.update(wall_seconds=round(time.monotonic() - start, 3),
                    returncode=result.returncode, status="passed" if result.returncode == 0 else "failed",
                    log_sha256=digest(log))
        write_json(self.path, self.data)
        if result.returncode:
            raise RuntimeError(f"{key} failed; see {log}")
        return log

    def build(self, *names):
        (WORK / "bin").mkdir(parents=True, exist_ok=True)
        flags = ["-std=c++17", "-O3", "-pthread", "-Wall", "-Wextra", "-Isrc"]
        libs = []
        if self.args.backend == "blas":
            flags.append("-DWEFT_USE_BLAS")
            libs = ["-framework", "Accelerate"] if sys.platform == "darwin" else ["-lopenblas"]
        for name in names:
            self.command("build_" + name,
                         [self.args.compiler, *flags, ROOT / "examples" / (name + ".cpp"),
                          "-o", WORK / "bin" / name, *libs])

    def finish(self, files):
        self.data.update(completed_utc=utc(), status="passed",
                         artifacts_sha256={relative(p): digest(p) for p in files})
        write_json(self.path, self.data)
        print("DONE", self.name, flush=True)


def prepare():
    records = []
    for path in sorted(CORPUS.rglob("*.wav")):
        with wave.open(str(path)) as wav:
            item = {"path": str(path.relative_to(CORPUS)), "sha256": digest(path),
                    "sample_rate": wav.getframerate(), "samples": wav.getnframes(),
                    "channels": wav.getnchannels(), "sample_width": wav.getsampwidth()}
        if (item["sample_rate"], item["channels"], item["sample_width"]) != (44100, 1, 2):
            raise RuntimeError(f"unexpected audio format: {path}")
        records.append(item)
    if not records:
        raise RuntimeError("TinySOL_2020 is missing; run data/download_TinySOL_2020.sh")
    accepted = [r for r in records if r["samples"] // 512 + 1 >= 130]
    validation = {r["path"] for i, r in enumerate(accepted) if i % 20 == 10}
    train = {r["path"] for r in accepted} - validation
    for r in records:
        r["split"] = "train" if r["path"] in train else "validation" if r["path"] in validation else "too_short"
    for _, path, _, _ in LISTENING:
        if path not in validation:
            raise RuntimeError(f"predeclared listening note is not held out: {path}")
    inventory = {"source": "https://www.orch-idea.org/data/db/TinySOL_2020.zip",
                 "split": "sort full paths; discard notes under 130 control frames; index % 20 == 10 is validation",
                 "notes": records, "files": len(records), "train_files": len(train),
                 "validation_files": len(validation), "excluded_short_files": len(records)-len(accepted),
                 "listening_notes": [list(row) for row in LISTENING]}
    write_json(EVIDENCE / "corpus.json", inventory)
    for split in ("train", "validation", "too_short"):
        (EVIDENCE / (split + ".txt")).write_text("".join(r["path"] + "\n" for r in records if r["split"] == split))
    for rel in train:
        target = WORK / "vae_corpus" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(CORPUS / rel)
    mnist = ROOT / "data/mnist"
    write_json(EVIDENCE / "mnist_data.json", {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(mnist.iterdir()) if p.is_file() and not p.name.startswith(".")})
    print("Prepared", len(train), "training and", len(validation), "validation notes.")


def mnist(args):
    run = Run("mnist", args)
    run.build("mnist_classifier")
    (WORK / "mnist").mkdir(parents=True, exist_ok=True)
    files = []
    for seed in MNIST_SEEDS:
        model = WORK / "mnist" / ("seed_" + str(seed) + ".wnet")
        run.command("mnist_seed_" + str(seed), [WORK / "bin/mnist_classifier", ROOT / "data/mnist", seed, model])
        files.append(model)
    run.finish(files)


def vae(args):
    run = Run("vae", args)
    run.build("extract_features", "train_audio_vae", "vae_remap")
    (WORK / "vae").mkdir(parents=True, exist_ok=True)
    cache = WORK / "vae/train.feat"
    prefix = WORK / "vae/model"
    run.command("vae_extract", [WORK / "bin/extract_features", WORK / "vae_corpus", cache, "logmag", "per_frame"])
    run.command("vae_train", [WORK / "bin/train_audio_vae", cache, prefix, VAE_EPOCHS, 1])
    files = [cache, prefix.with_suffix(".enc"), prefix.with_suffix(".dec")]
    for name, rel, _, _ in LISTENING:
        out = WORK / "vae" / (name + ".wav")
        run.command("vae_" + name, [WORK / "bin/vae_remap", prefix, CORPUS / rel, out])
        files.append(out)
    run.finish(files)


def synthesis(args):
    run = Run("synthesis", args)
    run.build("synth_fit", "synth_train")
    (WORK / "synthesis").mkdir(parents=True, exist_ok=True)
    prefix = WORK / "synthesis/model"
    run.command("synth_train", [WORK / "bin/synth_train", "train", CORPUS, prefix, SYNTH_EPOCHS, SYNTH_BATCH])
    actual = set(Path(str(prefix) + "_held_out.txt").read_text().splitlines())
    expected = {Path(p).stem for p in (EVIDENCE / "validation.txt").read_text().splitlines()}
    if actual != expected:
        raise RuntimeError("synth_train split differs from the predeclared file manifest")
    files = [Path(str(prefix) + ext) for ext in (".net", ".emb", ".txt", "_held_out.txt")]
    for name, rel, pitch, inst in LISTENING:
        out = WORK / "synthesis" / (name + ".wav")
        run.command("synth_" + name, [WORK / "bin/synth_train", "resynth", prefix, CORPUS / rel, pitch, inst, out])
        files.append(out)
    cello = LISTENING[0]
    fit = WORK / "synthesis/cello_fit"
    run.command("synth_direct_fit", [WORK / "bin/synth_fit", "fit", CORPUS / cello[1], cello[2], fit, 400])
    for label, args_tail in (
        ("phrase", ["phrase", prefix, "Fl", WORK / "synthesis/phrase.wav"]),
        ("morph", ["note", prefix, "Fl", "Vc", "A4", 6, "flat", WORK / "synthesis/morph.wav"]),
    ):
        run.command("synth_" + label, [WORK / "bin/synth_train", *args_tail])
        files.append(WORK / "synthesis" / (label + ".wav"))
    files.extend([Path(str(fit) + "_resynth.wav"), fit.with_suffix(".ctl")])
    run.finish(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("prepare", "mnist", "vae", "synthesis", "all"))
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--backend", choices=("portable", "blas"), default="blas")
    args = parser.parse_args()
    if git("diff", "v0.2.0", "--", "src", "examples", "build/Makefile"):
        raise RuntimeError("this protocol requires the v0.2.0 C++ sources and build settings")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.part in ("prepare", "all"):
        prepare()
    elif not (EVIDENCE / "corpus.json").exists():
        raise RuntimeError("run the prepare step first")
    for part in ("mnist", "vae", "synthesis"):
        if args.part in (part, "all"):
            globals()[part](args)


if __name__ == "__main__":
    main()
