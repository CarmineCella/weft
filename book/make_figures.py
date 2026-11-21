#!/usr/bin/env python3
"""
make_figures.py  --  draw the figures of the book.

Run from the repository root:

    python3 book/make_figures.py            # all figures
    python3 book/make_figures.py 28 29      # only those chapters

The figures are written to notes/figures/ as PNG (shown by the Markdown
chapters) and, for line drawings, also as PDF (used by the LaTeX book).
They are committed to the repository, so the book builds without running
this script.

Most figures are drawn from the files the example programs write: the
training logs, atom lists, control files, MIDI files and WAV files in
outputs/, and the data sets in data/.  A figure whose input is missing
is skipped with a message.

Needs numpy, scipy and matplotlib.
"""
import glob
import os
import re
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notes", "figures")
O = os.path.join(ROOT, "outputs")
D = os.path.join(ROOT, "data")

# One small palette, used in a fixed order.  Lines that share a plot also
# differ in dash pattern, so the figures can be read in black and white.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "TeX Gyre Pagella", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

W = 5.6          # figure width in inches


def save(fig, name, pdf=True):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    if pdf:
        fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=300)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128.0) / 128.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x.astype(np.float64)


def stft_mag(x, n=2048, hop=512):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1)).T      # (bins, frames)


def show_spec(ax, x, sr, fmax=6000, n=2048, hop=512, ref=None, floor=-70):
    S = stft_mag(x, n, hop)
    if ref is None:
        ref = S.max()
    db = 20 * np.log10(np.maximum(S, 1e-12) / ref)
    kmax = int(fmax * n / sr)
    ax.imshow(db[:kmax], origin="lower", aspect="auto", cmap="Greys",
              vmin=floor, vmax=0, interpolation="nearest", rasterized=True,
              extent=[0, len(x) / sr, 0, kmax * sr / n / 1000.0])
    ax.grid(False)
    return ref


def read_table(path, ncols, start_after=None):
    """Rows of a log that begin with a number and have at least ncols numbers."""
    rows, on = [], start_after is None
    for line in open(path):
        if not on:
            on = start_after in line
            continue
        nums = re.findall(r"-?\d+\.\d+|-?\d+", line.replace("%", " "))
        if len(nums) >= ncols and re.match(r"\s*(epoch\s+)?\d", line):
            rows.append([float(v) for v in nums[:ncols]])
    return np.array(rows)


def read_atoms(path):
    d = open(path, "rb").read()
    assert d[:4] == b"WATM", path
    sr, n, ns = struct.unpack("<3I", d[4:16])
    lens = struct.unpack("<%dI" % ns, d[16:16 + 4 * ns])
    p = 16 + 4 * ns
    na, = struct.unpack("<I", d[p:p + 4])
    rec = np.dtype([("scale", "<u4"), ("start", "<i4"), ("bin", "<u4"),
                    ("amp", "<f4"), ("phase", "<f4")])
    a = np.frombuffer(d, dtype=rec, count=na, offset=p + 4)
    return sr, n, np.array(lens), a


def read_ctl(path):
    d = open(path, "rb").read()
    assert d[:4] == b"WCTL", path
    sr, n, F, K, B = struct.unpack("<5I", d[4:24])
    v = np.frombuffer(d, dtype="<f4", offset=24)
    f0 = v[:F]
    A = v[F:F + K * F].reshape(K, F)
    M = v[F + K * F:F + K * F + B * F].reshape(B, F)
    return sr, n, f0, A, M


def read_midi(path):
    """Notes of a MIDI file: per track a list of (start, end, pitch) in
    quarter notes."""
    d = open(path, "rb").read()
    assert d[:4] == b"MThd"
    _, ntr, div = struct.unpack(">3H", d[8:14])
    p, tracks = 14, []
    for _ in range(ntr):
        assert d[p:p + 4] == b"MTrk"
        ln, = struct.unpack(">I", d[p + 4:p + 8])
        q, end = p + 8, p + 8 + ln
        t, status, on, notes = 0, 0, {}, []
        while q < end:
            dt = 0
            while True:
                b = d[q]; q += 1
                dt = (dt << 7) | (b & 0x7F)
                if b < 0x80:
                    break
            t += dt
            if d[q] & 0x80:
                status = d[q]; q += 1
            if status == 0xFF:
                q += 1
                ln2 = 0
                while True:
                    b = d[q]; q += 1
                    ln2 = (ln2 << 7) | (b & 0x7F)
                    if b < 0x80:
                        break
                q += ln2
            elif status in (0xF0, 0xF7):
                ln2 = 0
                while True:
                    b = d[q]; q += 1
                    ln2 = (ln2 << 7) | (b & 0x7F)
                    if b < 0x80:
                        break
                q += ln2
            else:
                kind = status & 0xF0
                if kind in (0xC0, 0xD0):
                    q += 1
                else:
                    a, b = d[q], d[q + 1]; q += 2
                    if kind == 0x90 and b > 0:
                        if a in on:
                            notes.append((on[a] / div, t / div, a))
                        on[a] = t
                    elif kind == 0x80 or (kind == 0x90 and b == 0):
                        if a in on:
                            notes.append((on.pop(a) / div, t / div, a))
        if notes:
            tracks.append(sorted(notes))
        p = end
    return tracks


def box(ax, x, y, w, h, text, fill="#f3f2ee", edge=MUTED, size=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                shrinkA=0, shrinkB=0, **kw))


def blank(ax):
    ax.set_axis_off()
    ax.grid(False)


# --------------------------------------------------------------------------
# chapters 4-12
# --------------------------------------------------------------------------

def ch04():
    # The neuron of the chapter: x = (2, -1), w = (0.5, 1), b = 0.1, t = 1.
    w = np.linspace(0.1, 1.5, 300)
    L = lambda w1: 0.5 * (2 * w1 - 1.0 + 0.1 - 1.0) ** 2
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.6))
    ax.plot(w, L(w), color=BLUE)
    w0, w1 = 0.5, 0.68
    slope = -1.8
    ww = np.array([0.33, 0.67])
    ax.plot(ww, L(w0) + slope * (ww - w0), color=MUTED, lw=0.9, ls="--")
    ax.plot([w0, w1], [L(w0), L(w1)], "o", color=ORANGE, ms=5, zorder=3)
    ax.annotate("", xy=(w1, L(w1) + 0.012), xytext=(w0 + 0.01, L(w0)),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(w0 + 0.05, L(w0) + 0.03, "before: $w_1=0.5$, $L=0.405$", fontsize=8)
    ax.text(w1 + 0.04, L(w1) + 0.03, "after: $w_1=0.68$, $L=0.1458$", fontsize=8)
    ax.text(0.28, 0.72, "slope $-1.8$", fontsize=8, color=MUTED)
    ax.set_xlabel("weight $w_1$")
    ax.set_ylabel("loss $L$")
    ax.set_ylim(-0.02, 0.9)
    save(fig, "04_one_step")


def ch05():
    z = np.linspace(-5, 5, 400)
    sig = 1 / (1 + np.exp(-z))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3))
    a.plot(z, np.maximum(z, 0), color=BLUE, label="ReLU")
    a.plot(z, sig, color=ORANGE, ls="--", label="sigmoid")
    a.plot(z, np.tanh(z), color=AQUA, ls=":", label="tanh")
    a.set_ylim(-1.3, 3)
    a.set_title("activation")
    a.set_xlabel("$z$")
    a.legend(loc="upper left")
    b.plot(z, (z > 0).astype(float), color=BLUE)
    b.plot(z, sig * (1 - sig), color=ORANGE, ls="--")
    b.plot(z, 1 - np.tanh(z) ** 2, color=AQUA, ls=":")
    b.set_ylim(-0.08, 1.15)
    b.set_title("derivative")
    b.set_xlabel("$z$")
    save(fig, "05_activations")


def ch07():
    p = np.linspace(0.005, 1, 400)
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.5))
    ax.plot(p, -np.log(p), color=BLUE, label="cross-entropy, $-\\log p$")
    ax.plot(p, (1 - p) ** 2, color=ORANGE, ls="--", label="squared error, $(1-p)^2$")
    ax.set_ylim(0, 5)
    ax.set_xlabel("probability $p$ given to the correct class")
    ax.set_ylabel("loss")
    ax.legend()
    save(fig, "07_cross_entropy")


def curves(ax_l, ax_a, ep, trl, tra, tel, tea, names=("training", "test")):
    ax_l.plot(ep, trl, color=BLUE, label=names[0])
    ax_l.plot(ep, tel, color=ORANGE, ls="--", label=names[1])
    ax_l.set_xlabel("epoch"); ax_l.set_title("loss")
    ax_l.legend()
    ax_a.plot(ep, tra, color=BLUE)
    ax_a.plot(ep, tea, color=ORANGE, ls="--")
    ax_a.set_xlabel("epoch"); ax_a.set_title("accuracy")


def ch09():
    t = read_table(os.path.join(O, "09_iris/log.txt"), 5, "train_loss")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3))
    curves(a, b, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4])
    b.set_ylim(0, 1.03)
    save(fig, "09_iris_training")


def ch10():
    # The test problem of tests/test_optimizer.cpp: f = x^2/2 + 50 y^2,
    # 100 steps from (5, 5).  SGD with lr 0.01, Adam with lr 0.1.
    f = lambda p: 0.5 * p[0] ** 2 + 50 * p[1] ** 2
    g = lambda p: np.array([p[0], 100 * p[1]])
    p = np.array([5.0, 5.0]); sgd = [p.copy()]
    for _ in range(100):
        p = p - 0.01 * g(p); sgd.append(p.copy())
    p = np.array([5.0, 5.0]); adam = [p.copy()]
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, 101):
        gr = g(p)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        p = p - 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        adam.append(p.copy())
    sgd, adam = np.array(sgd), np.array(adam)
    print("    final loss: SGD %.2f, Adam %.2f" % (f(sgd[-1]), f(adam[-1])))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.5),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
    X, Y = np.meshgrid(np.linspace(-1, 5.6, 200), np.linspace(-2.2, 5.6, 200))
    a.contour(X, Y, 0.5 * X ** 2 + 50 * Y ** 2, levels=[1, 10, 50, 150, 400, 900],
              colors=GRID, linewidths=0.7)
    a.plot(sgd[:, 0], sgd[:, 1], color=BLUE, marker="o", ms=2, lw=1, label="SGD")
    a.plot(adam[:, 0], adam[:, 1], color=ORANGE, marker="s", ms=2, lw=1, ls="--",
           label="Adam")
    a.plot([0], [0], "+", color=INK, ms=8)
    a.set_xlabel("$x$"); a.set_ylabel("$y$"); a.grid(False)
    a.legend(loc="upper left")
    b.semilogy([f(q) for q in sgd], color=BLUE)
    b.semilogy([f(q) for q in adam], color=ORANGE, ls="--")
    b.set_xlabel("step"); b.set_ylabel("loss")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "10_sgd_adam")


def mnist_images(n):
    path = os.path.join(D, "mnist/train-images-idx3-ubyte")
    d = np.fromfile(path, dtype=np.uint8, offset=16, count=n * 784)
    return d.reshape(n, 28, 28)


def ch12():
    img = mnist_images(30)
    fig, axs = plt.subplots(2, 15, figsize=(W, 0.85))
    for ax, im in zip(axs.ravel(), img):
        ax.imshow(im, cmap="Greys", interpolation="nearest")
        blank(ax)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save(fig, "12_mnist_digits", pdf=False)

    t = read_table(os.path.join(O, "12_mnist/log.txt"), 5, "train loss")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3))
    curves(a, b, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4])
    b.set_ylim(0.86, 1.0)
    save(fig, "12_mnist_training")


# --------------------------------------------------------------------------
# chapters 13-15
# --------------------------------------------------------------------------

def ch13():
    sr, n = 8000, 1024
    t = np.arange(n) / sr
    x = (1.0 * np.sin(2 * np.pi * 250 * t) + 0.5 * np.sin(2 * np.pi * 750 * t)
         + 0.25 * np.sin(2 * np.pi * 1500 * t))
    X = np.abs(np.fft.rfft(x)) / (n / 2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.2))
    a.plot(t[:160] * 1000, x[:160], color=BLUE, lw=1.2)
    a.set_xlabel("time (ms)"); a.set_title("signal")
    b.plot(np.fft.rfftfreq(n, 1 / sr), X, color=BLUE, lw=1.2)
    b.set_xlim(0, 2000)
    b.set_xlabel("frequency (Hz)"); b.set_title("magnitude of the FFT")
    save(fig, "13_signal_spectrum")


def ch14():
    # leakage: a sine between two bins, with and without the Hann window
    n, sr = 4096, 44100
    k = np.arange(n)
    x = np.sin(2 * np.pi * 440.0 * k / sr)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    fr = np.fft.rfftfreq(n, 1 / sr)
    R = np.abs(np.fft.rfft(x)); H = np.abs(np.fft.rfft(x * hann))
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.4))
    ax.plot(fr, 20 * np.log10(R / R.max() + 1e-12), color=ORANGE, ls="--",
            lw=1.2, label="no window")
    ax.plot(fr, 20 * np.log10(H / H.max() + 1e-12), color=BLUE, lw=1.2,
            label="Hann window")
    ax.set_xlim(0, 2000); ax.set_ylim(-120, 5)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("level (dB)")
    ax.legend()
    save(fig, "14_window")

    # the mel filterbank of AudioFeatures.h: 40 triangles, 2049 bins
    n_mels, n_bins = 40, 2049
    mel = lambda hz: 2595 * np.log10(1 + hz / 700.0)
    inv = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = inv(np.linspace(mel(0), mel(sr / 2), n_mels + 2)) * 2 * (n_bins - 1) / sr
    kk = np.arange(n_bins, dtype=float)
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        l, c, r = pts[m], pts[m + 1], pts[m + 2]
        up = (kk >= l) & (kk <= c)
        dn = (kk > c) & (kk <= r)
        fb[m, up] = (kk[up] - l) / (c - l)
        fb[m, dn] = (r - kk[dn]) / (r - c)
    hz = kk * sr / (2 * (n_bins - 1))
    fig, ax = plt.subplots(figsize=(W, 1.9))
    for m in range(n_mels):
        ax.plot(hz / 1000, fb[m], color=BLUE if m % 2 == 0 else MUTED, lw=0.8)
    ax.set_xlim(0, 22.05); ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("weight")
    save(fig, "14_mel_filterbank")

    # one note: waveform, log-magnitude frames, log mel energies
    path = os.path.join(D, "TinySOL_2020/Winds/Fl/ordinario/Fl-ord-A4-mf.wav")
    sr, x = read_wav(path)
    hop = 2048
    frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * hann, axis=1))        # (frames, 2049)
    logmel = np.log(mag ** 2 @ fb.T + 1e-12)
    dur = len(x) / sr
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.6), sharex=True)
    tt = np.arange(len(x)) / sr
    axs[0].plot(tt, x, color=BLUE, lw=0.3)
    axs[0].set_ylabel("amplitude"); axs[0].set_title("waveform", loc="left")
    db = 20 * np.log10(np.maximum(mag, 1e-9) / mag.max()).T
    kmax = int(8000 * n / sr)
    axs[1].imshow(db[:kmax], origin="lower", aspect="auto", cmap="Greys",
                  vmin=-80, vmax=0, extent=[0, dur, 0, 8], interpolation="nearest")
    axs[1].grid(False); axs[1].set_ylabel("frequency (kHz)")
    axs[1].set_title("log-magnitude spectrum of each frame (shown up to 8 kHz)",
                     loc="left")
    axs[2].imshow(logmel.T, origin="lower", aspect="auto", cmap="Greys",
                  extent=[0, dur, 0.5, 40.5], interpolation="nearest",
                  vmin=logmel.max() - 18, vmax=logmel.max())
    axs[2].grid(False); axs[2].set_ylabel("mel band")
    axs[2].set_title("log energy in the 40 mel bands", loc="left")
    axs[2].set_xlabel("time (s)")
    fig.subplots_adjust(hspace=0.35)
    save(fig, "14_features", pdf=False)


def ch15():
    path = os.path.join(O, "15_sound_classifier/log_classifier_mfcc_averaged.txt")
    lines = open(path).read().split("confusion matrix")[1].splitlines()
    names = lines[1].split("|")[1].split()
    M = np.array([[int(v) for v in l.split("|")[1].split()]
                  for l in lines[3:3 + len(names)]])
    fig, ax = plt.subplots(figsize=(W * 0.7, W * 0.62))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
    for i in range(len(names)):
        for j in range(len(names)):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.55 * M.max() else INK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "15_confusion")


# --------------------------------------------------------------------------
# images written by the example programs (chapters 16-18, 22)
# --------------------------------------------------------------------------

def copies():
    import shutil
    for src, dst in [
        ("16_autoencoder/ae_reconstructions.png", "16_reconstructions.png"),
        ("16_autoencoder/ae_interpolation.png", "16_interpolation.png"),
        ("17_vae/vae_latent_grid.png", "17_latent_grid.png"),
        ("17_vae/vae_samples.png", "17_samples.png"),
        ("17_vae/vae_interpolation.png", "17_interpolation.png"),
        ("18_weight_visualization/mnist_weights.png", "18_weights.png"),
        ("22_convnets/cifar_filters.png", "22_filters.png"),
    ]:
        s = os.path.join(O, src)
        if os.path.exists(s):
            os.makedirs(OUT, exist_ok=True)
            shutil.copyfile(s, os.path.join(OUT, dst))
            print("  copied", dst)
        else:
            print("  missing", src)


def spec_rows(name, items, fmax=6000, t0=0.0, t1=None, height=1.45, own=False):
    fig, axs = plt.subplots(len(items), 1, figsize=(W, height * len(items)),
                            sharex=True)
    ref = None
    for ax, (title, path) in zip(np.atleast_1d(axs), items):
        sr, x = read_wav(path)
        a, b = int(t0 * sr), (len(x) if t1 is None else int(t1 * sr))
        r = show_spec(ax, x[a:b], sr, fmax=fmax, ref=ref)
        ref = None if own else (ref or r)     # own: each panel has its own 0 dB
        ax.set_title(title, loc="left"); ax.set_ylabel("kHz")
    np.atleast_1d(axs)[-1].set_xlabel("time (s)")
    fig.subplots_adjust(hspace=0.38)
    save(fig, name, pdf=False)


def ch19():
    spec_rows("19_remap", [
        ("input: cage.wav", os.path.join(O, "19_audio_vae/input_cage.wav")),
        ("output of the dense VAE", os.path.join(O, "19_audio_vae/remap_cage.wav")),
    ], t1=8.0, own=True)


def ch22():
    t = read_table(os.path.join(O, "22_convnets/log_train.txt"), 5, "Starting")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3))
    a.plot(t[:, 0] + 1, t[:, 1], color=BLUE)
    a.set_xlabel("epoch"); a.set_title("training loss")
    b.plot(t[:, 0] + 1, t[:, 2], color=BLUE, label="training (dropout on)")
    b.plot(t[:, 0] + 1, t[:, 3], color=ORANGE, ls="--", label="test")
    b.set_xlabel("epoch"); b.set_title("accuracy (%)"); b.legend()
    save(fig, "22_cifar_training")


def ch23():
    spec_rows("23_remap", [
        ("input: cage.wav", os.path.join(O, "23_convvae/input_cage.wav")),
        ("output of the convolutional VAE", os.path.join(O, "23_convvae/remap_cage.wav")),
    ], t1=8.0, own=True)


# --------------------------------------------------------------------------
# chapter 24
# --------------------------------------------------------------------------

def ch24():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    blank(ax); ax.set_xlim(0, 10); ax.set_ylim(-0.5, 3.4)
    s = 7.5
    box(ax, 0.95, 2.75, 1.7, 0.55, "pitch $f_0$", size=s)
    box(ax, 0.95, 1.95, 1.7, 0.7, "harmonic\namplitudes", size=s)
    box(ax, 0.95, 0.7, 1.7, 0.7, "noise band\ngains", size=s)
    box(ax, 4.0, 2.35, 2.5, 0.85, "HarmonicSynth\nsum of sinusoids", fill="#e4eefb", size=s)
    box(ax, 4.0, 0.7, 2.5, 0.85, "FilteredNoise\nnoise through filters", fill="#e4eefb", size=s)
    box(ax, 6.35, 1.52, 0.45, 0.45, "+", size=s)
    box(ax, 8.65, 1.52, 2.3, 0.85, "SpectralLoss\nagainst the recording", fill="#fdeee6", size=s)
    arrow(ax, 1.83, 2.75, 2.72, 2.5); arrow(ax, 1.83, 1.95, 2.72, 2.2)
    arrow(ax, 1.83, 0.7, 2.72, 0.7)
    arrow(ax, 5.28, 2.35, 6.2, 1.72); arrow(ax, 5.28, 0.7, 6.2, 1.32)
    arrow(ax, 6.6, 1.52, 7.47, 1.52)
    ax.text(7.03, 1.63, "audio", ha="center", fontsize=7, color=MUTED)
    ax.plot([8.65, 8.65, 0.95], [1.07, -0.15, -0.15], color=ORANGE, lw=0.9, ls="--")
    ax.annotate("", xy=(0.95, 0.32), xytext=(0.95, -0.15),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9))
    ax.text(4.8, -0.4, "gradient of the loss with respect to the controls",
            ha="center", fontsize=7, color=MUTED)
    save(fig, "24_synthesiser")

    C = os.path.join(O, "24_differentiable_synthesis/controls")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, (nm, f) in zip(axs, [("flute, A4", "flute_A4.ctl"),
                                 ("clarinet, A4", "clarinet_A4.ctl")]):
        sr, n, f0, A, M = read_ctl(os.path.join(C, f))
        F = A.shape[1]
        mid = np.abs(A[:, F // 4: 3 * F // 4])
        a = np.median(mid, axis=1)
        db = 20 * np.log10(np.maximum(a, 1e-6) / a.max())
        ax.bar(np.arange(1, 21), db[:20] + 80, bottom=-80, color=BLUE, width=0.7)
        ax.set_ylim(-80, 3); ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(nm); ax.set_xlabel("harmonic")
        ax.grid(axis="x")
    axs[0].set_ylabel("level (dB)")
    save(fig, "24_harmonics")

    R = os.path.join(O, "24_differentiable_synthesis/1_resynthesis")
    spec_rows("24_resynthesis", [
        ("recording: flute, A4", os.path.join(R, "flute_A4_a_original.wav")),
        ("fitted synthesiser: harmonic part", os.path.join(C, "refit_flute_A4_harmonic.wav")),
        ("fitted synthesiser: noise part", os.path.join(C, "refit_flute_A4_noise.wav")),
    ], fmax=8000, t1=4.0, height=1.3)


# --------------------------------------------------------------------------
# chapters 25-27
# --------------------------------------------------------------------------

def ch25():
    rng = np.random.default_rng(3)
    T = 8
    S = rng.normal(size=(T, T)) * 1.2
    sm = lambda Z: np.exp(Z - Z.max(1, keepdims=True)) / \
        np.exp(Z - Z.max(1, keepdims=True)).sum(1, keepdims=True)
    Sm = S.copy(); Sm[np.triu_indices(T, 1)] = -np.inf
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, 2.5))
    for ax, A, ttl in zip(axs, [sm(S), sm(Sm)], ["without the mask", "with the causal mask"]):
        ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(ttl); ax.grid(False)
        ax.set_xticks(range(T)); ax.set_xticklabels(range(1, T + 1))
        ax.set_yticks(range(T)); ax.set_yticklabels(range(1, T + 1))
        ax.set_xlabel("position attended to")
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("position being computed")
    save(fig, "25_causal_mask")


def ch26():
    d, T = 64, 100
    pos = np.arange(T)[:, None]; i = np.arange(d // 2)[None, :]
    ang = pos / 10000 ** (2 * i / d)
    PE = np.zeros((T, d)); PE[:, 0::2] = np.sin(ang); PE[:, 1::2] = np.cos(ang)
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.4))
    ax.grid(False)
    im = ax.imshow(PE.T, cmap="RdBu", vmin=-1, vmax=1, aspect="auto", origin="lower",
                   interpolation="nearest")
    ax.set_xlabel("position"); ax.set_ylabel("feature"); ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02, ticks=[-1, 0, 1])
    save(fig, "26_positional_encoding", pdf=False)


def ch27():
    L = os.path.join(O, "27_transformer_chorales/training_log.txt")
    t = read_table(L, 3, "held-out")
    txt = open(L).read()
    uni = float(re.search(r"unigram\s*:\s*([\d.]+)", txt).group(1))
    mk = float(re.search(r"Markov\s*:\s*([\d.]+)", txt).group(1))
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.5))
    ax.plot(t[:, 0], t[:, 1], color=BLUE, label="transformer, training")
    ax.plot(t[:, 0], t[:, 2], color=ORANGE, ls="--", label="transformer, held out")
    ax.axhline(uni, color=MUTED, lw=0.9, ls=":")
    ax.axhline(mk, color=MUTED, lw=0.9, ls=":")
    ax.text(t[-1, 0], uni + 0.02, "unigram, held out", ha="right", fontsize=8, color=MUTED)
    ax.text(t[-1, 0], mk + 0.02, "Markov chain, held out", ha="right", fontsize=8,
            color=MUTED)
    ax.set_ylim(0.3, 1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (nats per token)")
    ax.legend(loc="center right")
    save(fig, "27_training")

    P = os.path.join(O, "27_transformer_chorales")
    items = [("Bach, BWV 269", "bach_originals/bwv269.mid"),
             ("transformer", "transformer/transformer_1.mid"),
             ("Markov chain", "markov_baseline/markov_1.mid")]
    cols = [BLUE, ORANGE, AQUA, "#6250d6"]
    fig, axs = plt.subplots(3, 1, figsize=(W, 5.4), sharex=True, sharey=True)
    for ax, (ttl, f) in zip(axs, items):
        tr = read_midi(os.path.join(P, f))
        for v, notes in enumerate(tr[:4]):
            rect = [Rectangle((s, p - 0.4), e - s - 0.06, 0.8)
                    for s, e, p in notes if s < 32]
            ax.add_collection(PatchCollection(rect, fc=cols[v], ec="none"))
        ax.set_axisbelow(True)
        ax.set_xlim(0, 32); ax.set_ylim(34, 84)
        ax.set_xticks(range(0, 33, 4)); ax.set_yticks([36, 48, 60, 72, 84])
        ax.set_yticklabels(["C2", "C3", "C4", "C5", "C6"])
        ax.set_title(ttl, loc="left")
    for v, nm in enumerate(["soprano", "alto", "tenor", "bass"]):
        axs[0].plot([], [], color=cols[v], lw=4, label=nm)
    axs[0].legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  handlelength=1.2, columnspacing=1.2)
    axs[-1].set_xlabel("time (quarter notes)")
    fig.subplots_adjust(hspace=0.3)
    save(fig, "27_piano_rolls")


def ch26_block():
    # The pre-LN block of TransformerEncoderBlock.h.
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.9))
    blank(ax); ax.set_xlim(0, 6); ax.set_ylim(0, 8.2)
    x = 2.6
    box(ax, x, 0.45, 2.6, 0.55, "input $X$")
    box(ax, x, 1.75, 2.6, 0.6, "LayerNorm")
    box(ax, x, 2.85, 2.6, 0.7, "multi-head attention", fill="#e4eefb")
    box(ax, x, 3.9, 0.5, 0.5, "+")
    box(ax, x, 5.2, 2.6, 0.6, "LayerNorm")
    box(ax, x, 6.3, 2.6, 0.7, "Dense, ReLU, Dense", fill="#e4eefb")
    box(ax, x, 7.35, 0.5, 0.5, "+")
    for y0, y1 in [(0.75, 1.42), (2.07, 2.47), (3.22, 3.62), (4.17, 4.87),
                   (5.52, 5.92), (6.67, 7.07)]:
        arrow(ax, x, y0, x, y1)
    arrow(ax, x, 7.62, x, 8.15)
    ax.text(x + 0.2, 8.0, "output", fontsize=8, color=MUTED, va="center")
    for y0, y1 in [(1.08, 3.9), (4.52, 7.35)]:
        ax.plot([x, 4.7, 4.7], [y0, y0, y1], color=MUTED, lw=0.9)
        arrow(ax, 4.7, y1, x + 0.27, y1)
        ax.text(4.85, (y0 + y1) / 2, "residual", fontsize=8, color=MUTED,
                rotation=90, va="center")
    save(fig, "26_block")


# --------------------------------------------------------------------------
# chapters 28-29
# --------------------------------------------------------------------------

def atom_boxes(ax, sr, lens, a, t0, t1, fmax, floor=-40):
    N = lens[a["scale"]].astype(float)
    tc = (a["start"] + N / 2) / sr
    f = a["bin"] * sr / N
    amp = np.abs(a["amp"])
    db = 20 * np.log10(np.maximum(amp, 1e-9) / amp.max())
    keep = (tc > t0 - 0.2) & (tc < t1 + 0.2) & (f < fmax * 1.1)
    order = np.argsort(db[keep])
    rect, col = [], []
    for i in np.flatnonzero(keep)[order]:
        dt, df = N[i] / sr / 2, sr / N[i]        # half the window, one bin
        rect.append(Rectangle((tc[i] - dt / 2, (f[i] - df / 2) / 1000), dt, df / 1000))
        col.append(np.clip(1 - db[i] / floor, 0.15, 1))
    pc = PatchCollection(rect, cmap="Greys", lw=0.7)
    pc.set_array(np.array(col)); pc.set_clim(0, 1)
    pc.set_edgecolor(plt.cm.Greys(np.array(col)))
    ax.add_collection(pc)
    ax.set_xlim(t0, t1); ax.set_ylim(0, fmax / 1000); ax.grid(False)


def ch28():
    sr = 44100
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.6), sharey=True)
    for ax, N in zip(axs, [128, 512, 2048, 8192]):
        k = np.arange(N)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * k / N)
        ax.plot((k - N / 2) / sr * 1000, w * np.cos(2 * np.pi * 1000 * (k - N / 2) / sr),
                color=BLUE, lw=0.5)
        ax.set_xlim(-95, 95); ax.set_title("%d samples" % N)
        ax.set_xlabel("ms"); ax.set_yticks([])
    save(fig, "28_atoms")

    fig, ax = plt.subplots(figsize=(W * 0.8, 2.5))
    sty = [(BLUE, "-", "o"), (ORANGE, "--", "s"), (AQUA, ":", "^"), (MUTED, "-.", "D")]
    for (c, ls, mk), nm in zip(sty, ["cage", "radiohead", "Frank_Gamb_ori", "cello_A4"]):
        t = read_table(os.path.join(O, "28_matching_pursuit/logs/%s.txt" % nm), 4, "atoms/s")
        t = t[:6]
        ax.semilogx(t[:, 0], t[:, 2], color=c, ls=ls, marker=mk, ms=3.5,
                    label=nm + ".wav")
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["250", "500", "1000", "2000", "4000", "8000"])
    ax.minorticks_off()
    ax.set_xlabel("number of atoms"); ax.set_ylabel("signal-to-residual ratio (dB)")
    ax.legend()
    save(fig, "28_how_many")

    A = os.path.join(O, "28_matching_pursuit/atoms/cage.atoms")
    sra, n, lens, a = read_atoms(A)
    sr, x = read_wav(os.path.join(D, "cage.wav"))
    t0, t1, fmax = 0.0, 6.0, 4000
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True)
    show_spec(axs[0], x[:int(t1 * sr)], sr, fmax=fmax)
    axs[0].set_title("spectrogram of cage.wav", loc="left"); axs[0].set_ylabel("kHz")
    atom_boxes(axs[1], sra, lens, a[:2000], t0, t1, fmax)
    axs[1].set_title("the 2,000 strongest atoms", loc="left")
    axs[1].set_ylabel("kHz"); axs[1].set_xlabel("time (s)")
    fig.subplots_adjust(hspace=0.3)
    save(fig, "28_atom_list", pdf=False)


def ch29():
    L = os.path.join(O, "_work/atom3_train.log")
    t = read_table(L, 3, "held-out")
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.5))
    ax.plot(t[:, 0], t[:, 1], color=BLUE, label="training")
    ax.plot(t[:, 0], t[:, 2], color=ORANGE, ls="--", label="held out")
    best = int(np.argmin(t[:, 2]))
    ax.plot(t[best, 0], t[best, 2], "o", color=ORANGE, ms=5)
    ax.annotate("lowest held-out loss:\nthese weights are kept",
                xy=(t[best, 0], t[best, 2]), xytext=(t[best, 0], 3.2),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    for v, nm in [(2.71, "unigram"), (2.83, "Markov chain")]:
        ax.axhline(v, color=MUTED, lw=0.9, ls=":")
        ax.text(40, v + 0.03, nm + ", held out", ha="right", fontsize=8, color=MUTED)
    ax.set_ylim(1.7, 3.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (nats per token)")
    ax.legend(loc="lower left")
    save(fig, "29_training")

    P = os.path.join(O, "29_atom_transformer")
    items = [("recording: cage.wav, 2,000 atoms",
              os.path.join(O, "28_matching_pursuit/atoms/cage.atoms"), 2000),
             ("transformer, temperature 1.0",
              os.path.join(P, "2_generated_from_scratch/new_sound_1_temp1.0.atoms"), None),
             ("Markov chain",
              os.path.join(P, "4_markov_baseline/markov_1.atoms"), None)]
    fig, axs = plt.subplots(3, 1, figsize=(W, 5.6), sharex=True)
    for ax, (ttl, f, top) in zip(axs, items):
        sr, n, lens, a = read_atoms(f)
        atom_boxes(ax, sr, lens, a[:top] if top else a, 0.0, 6.0, 4000)
        ax.set_title(ttl, loc="left"); ax.set_ylabel("kHz")
    axs[-1].set_xlabel("time (s)")
    fig.subplots_adjust(hspace=0.3)
    save(fig, "29_atom_lists", pdf=False)


CHAPTERS = [("04", ch04), ("05", ch05), ("07", ch07), ("09", ch09), ("10", ch10),
            ("12", ch12), ("13", ch13), ("14", ch14), ("15", ch15),
            ("16", copies), ("19", ch19), ("22", ch22), ("23", ch23), ("24", ch24),
            ("25", ch25), ("26", ch26), ("26", ch26_block), ("27", ch27),
            ("28", ch28), ("29", ch29)]

if __name__ == "__main__":
    want = sys.argv[1:]
    for num, fn in CHAPTERS:
        if want and num not in want:
            continue
        print("chapter", num)
        try:
            fn()
        except (FileNotFoundError, OSError) as e:
            print("  skipped:", e)
