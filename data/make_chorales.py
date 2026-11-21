#!/usr/bin/env python3
"""
make_chorales.py -- write the Bach four-part chorales as MIDI files into
./chorales/, for examples/transformer_chorales.cpp.

The chorales come from the corpus that ships with the music21 toolkit, so
nothing is downloaded:

    pip install music21
    python3 data/make_chorales.py

Each output file is a Format 1 MIDI file with exactly four tracks, one
voice per track, in the order soprano, alto, tenor, bass -- the layout
transformer_chorales expects.  Chorales that do not have exactly four
parts (a few have an extra instrumental line) are skipped.

By default every chorale is transposed so that its key is C major or
A minor.  A small model trained on a few hundred pieces learns voice
leading much more easily when it does not also have to learn it twelve
times over, once per key.  Pass --no-transpose to keep the original keys.

This is the only place in the project that uses Python: the MIDI files it
writes are plain data, and everything that reads them is C++.
"""

import argparse
import os
import struct
import sys

try:
    from music21 import corpus
except ImportError:
    sys.exit("music21 is not installed.  Run:  pip install music21")

TICKS = 480                      # ticks per quarter note
VOICE_ORDER = ["soprano", "alto", "tenor", "bass"]


def vlq(n):
    """MIDI variable-length quantity."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def track_bytes(notes, tempo_us=None):
    """notes: list of (start_tick, duration_tick, pitch)."""
    events = []                                   # (tick, order, bytes)
    for start, dur, pitch in notes:
        events.append((start, 1, bytes([0x90, pitch, 80])))
        events.append((start + dur, 0, bytes([0x80, pitch, 64])))
    events.sort(key=lambda e: (e[0], e[1]))       # note-offs before note-ons
    body = b""
    if tempo_us is not None:
        body += vlq(0) + b"\xff\x51\x03" + struct.pack(">I", tempo_us)[1:]
    now = 0
    for tick, _, data in events:
        body += vlq(tick - now) + data
        now = tick
    body += vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + body


def write_midi(path, voices):
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(voices), TICKS))
        for i, notes in enumerate(voices):
            f.write(track_bytes(notes, tempo_us=666667 if i == 0 else None))


def semitones_to_c_or_a(score):
    """Shift (in semitones, -6..+5) that moves the key to C major / A minor."""
    key = score.analyze("key")
    target = 0 if key.mode == "major" else 9      # C or A
    shift = (target - key.tonic.pitchClass) % 12
    return shift - 12 if shift > 5 else shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transpose", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "chorales"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = corpus.getComposer("bach")
    written = skipped = 0
    for p in paths:
        name = os.path.splitext(os.path.basename(str(p)))[0]
        try:
            score = corpus.parse(p)
        except Exception:
            skipped += 1
            continue
        parts = list(score.parts)
        if len(parts) != 4:
            skipped += 1
            continue
        # Put the parts in S, A, T, B order by name when the names say so;
        # otherwise trust the score order (which is top to bottom).
        by_name = {}
        for part in parts:
            label = (part.partName or part.id or "").lower()
            for v in VOICE_ORDER:
                if label.startswith(v) and v not in by_name:
                    by_name[v] = part
        ordered = [by_name[v] for v in VOICE_ORDER] if len(by_name) == 4 else parts

        shift = 0 if args.no_transpose else semitones_to_c_or_a(score)
        voices, ok = [], True
        for part in ordered:
            notes = []
            for n in part.stripTies().flatten().notes:
                pitch = (max(n.pitches).midi if n.isChord else n.pitch.midi) + shift
                start = int(round(float(n.offset) * TICKS))
                dur = int(round(float(n.quarterLength) * TICKS))
                if dur > 0 and 0 <= pitch < 128:
                    notes.append((start, dur, pitch))
            if not notes:
                ok = False
            voices.append(notes)
        if not ok:
            skipped += 1
            continue
        write_midi(os.path.join(args.out, name + ".mid"), voices)
        written += 1

    print(f"wrote {written} chorales to {args.out}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
