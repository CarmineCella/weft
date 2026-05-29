#!/usr/bin/env python3
"""
preprocess_note.py  --  prepare a markdown note for pandoc + LaTeX.

The notes/*.md sources were written for markdown viewers and use
Unicode math characters freely (Greek letters, operators, sub- and
superscripts).  This script normalises them before pandoc sees them:

  1. Outside code blocks:
       - Greek letters and binary operators are wrapped in $...$ so that
         pandoc's tex_math_dollars extension treats them as math.
       - Sub- and superscripts use \textsubscript/\textsuperscript so
         they don't trigger LaTeX's "double superscript" / "missing
         math mode" errors.  These pass through pandoc as raw_tex.

  2. Inside code blocks: ASCII approximations.  Code is rendered with
     listings (not math), so Unicode would either fail to render or
     pull in a font we don't want.

  3. Heading prefixes like "# 01 -- Title" or "## 1. Section" are
     stripped so the book's auto-numbering doesn't double up.

Reads markdown on stdin, writes markdown on stdout.
"""

import re
import sys

# --- in-prose: Greek + binary operators wrapped in $...$ -------------------
INLINE_MATH = {
    "\u00b7": r"$\cdot$",
    "\u00d7": r"$\times$",
    "\u00b1": r"$\pm$",
    "\u2248": r"$\approx$",
    "\u2260": r"$\ne$",
    "\u2192": r"$\to$",
    "\u2190": r"$\leftarrow$",
    "\u2202": r"$\partial$",
    "\u2207": r"$\nabla$",
    "\u2208": r"$\in$",
    "\u2218": r"$\circ$",
    "\u221a": r"$\sqrt{\ }$",
    "\u2299": r"$\odot$",
    "\u03a3": r"$\Sigma$",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b4": r"$\delta$",
    "\u03b5": r"$\varepsilon$",
    "\u03b7": r"$\eta$",
    "\u03bc": r"$\mu$",
    "\u03c0": r"$\pi$",
    "\u03c3": r"$\sigma$",
    "\u211d": r"$\mathbb{R}$",
    "\u2212": "-",
    "\u2026": r"\ldots ",
    "\u2211": r"$\Sigma$",
    "\u22ee": r"$\vdots$",
}

# --- in-prose: sub/super scripts use text-mode commands --------------------
# These avoid the "double superscript / missing $" pitfalls of putting a
# bare ^ or _ into pandoc math mode.
INLINE_TEXT = {
    "\u2070": r"\textsuperscript{0}",
    "\u00b9": r"\textsuperscript{1}",
    "\u00b2": r"\textsuperscript{2}",
    "\u2074": r"\textsuperscript{4}",
    "\u207f": r"\textsuperscript{n}",
    "\u207b": r"\textsuperscript{-}",
    "\u2080": r"\textsubscript{0}",
    "\u2081": r"\textsubscript{1}",
    "\u2082": r"\textsubscript{2}",
    "\u2096": r"\textsubscript{k}",
    "\u2098": r"\textsubscript{m}",
    "\u2099": r"\textsubscript{n}",
    "\u1d40": r"\textsuperscript{T}",
    "\u1d50": r"\textsuperscript{m}",
    "\u1d57": r"\textsuperscript{t}",
    "\u1d62": r"\textsubscript{i}",
    "\u2c7c": r"\textsubscript{j}",
    "\u0302": "",  # combining hat -- drop
}

# --- in code: plain ASCII --------------------------------------------------
CODE_MAP = {
    "\u00b7": "*",  "\u00d7": "x",  "\u00b1": "+/-",
    "\u2248": "~",  "\u2260": "!=",
    "\u2192": "->", "\u2190": "<-",
    "\u2202": "d",  "\u2207": "grad", "\u2208": "in", "\u2218": "o",
    "\u221a": "sqrt", "\u2299": "(*)",
    "\u03a3": "Sum", "\u2211": "Sum",
    "\u03b1": "alpha", "\u03b2": "beta",  "\u03b4": "delta",
    "\u03b5": "eps",   "\u03b7": "eta",   "\u03bc": "mu",
    "\u03c0": "pi",    "\u03c3": "sigma",
    "\u211d": "R",
    "\u2212": "-",     "\u2026": "...",   "\u22ee": "...",
    "\u2070": "^0", "\u00b9": "^1", "\u00b2": "^2",
    "\u2074": "^4", "\u207f": "^n", "\u207b": "^-",
    "\u2080": "_0", "\u2081": "_1", "\u2082": "_2",
    "\u2096": "_k", "\u2098": "_m", "\u2099": "_n",
    "\u1d40": "^T", "\u1d50": "^m", "\u1d57": "^t",
    "\u1d62": "_i", "\u2c7c": "_j",
    "\u0302": "^",
    "\u2502": "|", "\u250c": "+", "\u2510": "+",
    "\u2514": "+", "\u2518": "+", "\u2500": "-",
    "\u251c": "+", "\u2524": "+", "\u252c": "+",
    "\u2534": "+", "\u253c": "+",
    "\u2713": "[x]",
}


def replace_prose(text):
    for k, v in INLINE_MATH.items():
        text = text.replace(k, v)
    for k, v in INLINE_TEXT.items():
        text = text.replace(k, v)
    return text


def replace_code(text):
    for k, v in CODE_MAP.items():
        text = text.replace(k, v)
    return text


# Match e.g. "# 01 -- Title", "## 1. Section", "### 1.2 Sub", remove the
# numeric prefix.  Keep the '#'s intact.
_PREFIX_DASH = re.compile(r"^(#+)\s+\d+\w*\s*[\-\u2014]+\s+")
_PREFIX_DOT  = re.compile(r"^(#+)\s+\d+(?:\.\d+)*\.?\s+")


def strip_heading_prefix(line):
    m = _PREFIX_DASH.match(line)
    if m:
        return m.group(1) + " " + line[m.end():]
    m = _PREFIX_DOT.match(line)
    if m:
        return m.group(1) + " " + line[m.end():]
    return line


def main():
    src = sys.stdin.read()
    parts = re.split(r"(```.*?```|`[^`\n]+`)", src, flags=re.DOTALL)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            lines = part.splitlines(keepends=True)
            lines = [strip_heading_prefix(ln) for ln in lines]
            part = "".join(lines)
            part = replace_prose(part)
        else:
            part = replace_code(part)
        out.append(part)
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
