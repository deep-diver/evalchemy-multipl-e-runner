#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


def _replace_list_block(src: str, varname: str, new_items: list[str]) -> tuple[str, bool]:
    """
    Replace a top-level assignment like:
        LANUGUAGES = [
            ...
        ]
    with a newly formatted list.

    Returns (new_src, changed?)
    """
    # Find: ^LANUGUAGES\s*=\s*\[
    pat = re.compile(rf"^(?P<indent>[ \t]*){re.escape(varname)}\s*=\s*\[", re.MULTILINE)
    m = pat.search(src)
    if not m:
        return src, False

    start = m.start()
    # Find the '[' position
    bracket_pos = src.find("[", m.end() - 1)
    if bracket_pos < 0:
        return src, False

    # Scan forward to find matching closing bracket for this list
    i = bracket_pos
    depth = 0
    end = None
    in_str = None  # "'" or '"'
    esc = False

    while i < len(src):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        i += 1

    if end is None:
        return src, False

    indent = m.group("indent")
    # Pretty list formatting (one per line, trailing comma)
    lines = [f'{indent}{varname} = [']
    for it in new_items:
        lines.append(f'{indent}    "{it}",')
    lines.append(f"{indent}]")
    replacement = "\n".join(lines)

    new_src = src[:start] + replacement + src[end + 1 :]
    return new_src, True


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Original MultiPLE folder (the one you currently mount)
    src_dir = Path(os.environ.get("HOST_MULTIPLE_DIR", repo_root / "multipl")).resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"HOST_MULTIPLE_DIR not found: {src_dir}")

    langs_raw = os.environ.get("MULTIPLE_LANGUAGES", "").strip()
    if not langs_raw:
        # If not set, just print original dir (no patch)
        print(str(src_dir))
        return

    langs = [x.strip() for x in langs_raw.split(",") if x.strip()]
    if not langs:
        print(str(src_dir))
        return

    # Patched output dir (stable location; overwritten each run)
    out_base = repo_root / ".patched"
    out_base.mkdir(parents=True, exist_ok=True)

    safe_tag = "_".join(langs)
    out_dir = (out_base / f"multipl_{safe_tag}").resolve()

    # Fresh copy (remove old)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    target = out_dir / "eval_instruct.py"
    if not target.exists():
        raise SystemExit(f"eval_instruct.py not found under: {out_dir}")

    src = target.read_text(encoding="utf-8")

    # Patch the typo'd constant name first (LANUGUAGES), and also support LANGUAGES if present.
    src2, changed1 = _replace_list_block(src, "LANUGUAGES", langs)
    src3, changed2 = _replace_list_block(src2, "LANGUAGES", langs)

    if not (changed1 or changed2):
        raise SystemExit("Could not find LANUGUAGES/LANGUAGES assignment block to patch.")

    target.write_text(src3, encoding="utf-8")

    # Print patched directory path for run.sh to consume
    print(str(out_dir))


if __name__ == "__main__":
    main()
