#!/usr/bin/env python3
"""Package the PDF and its relative listening links as one offline archive."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "book/book.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "book/build/designing-musical-networks.zip")
    args = parser.parse_args()
    if not args.pdf.is_file() or not args.pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("missing or invalid compiled PDF")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for folder in (ROOT / "notes", ROOT / "book/experiments"):
        files += [p for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")]
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.pdf, "book/book.pdf")
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("READ-ME.txt", "Designing Musical Networks\n\n"
                         "Extract the complete archive. Open book/book.pdf to read, or\n"
                         "notes/audio/index.html to listen. Keep the folder structure so\n"
                         "the PDF's relative links find its companion audio files.\n\n"
                         "Source and reproduction: https://github.com/CarmineCella/weft\n"
                         "The experiment protocol is in book/experiments/README.md.\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
