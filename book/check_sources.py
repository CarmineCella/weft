#!/usr/bin/env python3
"""Validate the canonical Markdown chapters and, optionally, a built PDF."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    errors, identifiers = [], defaultdict(list)
    references = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", (ROOT / "book/bibliography/refs.bib").read_text()))
    for path in sorted((ROOT / "notes").glob("*.md")):
        source = path.read_text()
        if sum(line.strip() == "$$" for line in source.splitlines()) % 2:
            errors.append(f"{path.name}: unbalanced display-math delimiters")
        prose = re.sub(r"```.*?```|`[^`\n]*`", "", source, flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", prose):
            if "://" not in target and not target.startswith("#"):
                if not (path.parent / target.split("#")[0]).exists():
                    errors.append(f"{path.name}: missing link {target}")
        for key in re.findall(r"@([a-zA-Z0-9_:-]+)", prose):
            if key not in references:
                errors.append(f"{path.name}: missing bibliography key {key}")
        prepared = subprocess.check_output(
            ["python3", str(ROOT / "book/preprocess_note.py"), path.stem],
            input=source, text=True)
        ast = json.loads(subprocess.check_output(
            ["pandoc", "--from", "markdown+tex_math_dollars+raw_tex", "--to", "json"],
            input=prepared, text=True))
        for block in ast["blocks"]:
            if block["t"] == "Header":
                identifiers[block["c"][1][0]].append(path.name)
    for identifier, files in identifiers.items():
        if len(files) != 1:
            errors.append(f"duplicate heading {identifier}: {', '.join(files)}")
    if args.pdf:
        pdf = args.pdf.resolve()
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            errors.append(f"missing or invalid PDF: {pdf}")
        else:
            text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
            if "??" in text:
                errors.append("PDF contains unresolved-reference markers")
            log = pdf.with_suffix(".log")
            if log.exists() and re.search(r"^!|LaTeX Error|Undefined control sequence", log.read_text(), re.M):
                errors.append("LaTeX log contains errors")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Book sources and requested PDF checks passed.")


if __name__ == "__main__":
    main()
