#!/usr/bin/env python3
"""Resolve Git conflict markers by keeping ours/theirs for a file.

Usage:
  python scripts/resolve_git_conflicts.py app.py --strategy ours
"""
from __future__ import annotations
import argparse
from pathlib import Path


def resolve(text: str, strategy: str) -> tuple[str, int]:
    out: list[str] = []
    i = 0
    lines = text.splitlines(keepends=True)
    resolved = 0

    while i < len(lines):
        line = lines[i]
        if not line.startswith("<<<<<<< "):
            out.append(line)
            i += 1
            continue

        # parse conflict block
        i += 1
        ours: list[str] = []
        theirs: list[str] = []

        while i < len(lines) and not lines[i].startswith("======="):
            ours.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError("Malformed conflict: missing =======")

        i += 1  # skip =======
        while i < len(lines) and not lines[i].startswith(">>>>>>> "):
            theirs.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError("Malformed conflict: missing >>>>>>>")

        i += 1  # skip >>>>>>>
        resolved += 1

        if strategy == "ours":
            out.extend(ours)
        elif strategy == "theirs":
            out.extend(theirs)
        else:  # both
            out.extend(ours)
            out.extend(theirs)

    return "".join(out), resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to conflicted file")
    ap.add_argument("--strategy", choices=["ours", "theirs", "both"], default="ours")
    args = ap.parse_args()

    p = Path(args.path)
    src = p.read_text(encoding="utf-8")
    result, n = resolve(src, args.strategy)
    p.write_text(result, encoding="utf-8")
    print(f"Resolved {n} conflict block(s) in {p} using strategy={args.strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
