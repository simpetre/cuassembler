"""Probe whether a SASS instruction can be assembled with the current repo.

Given a single SASS instruction string, parse it into (InsKey, vals, modi),
look up the repo entry, and report one of:
    OK                — assembleable
    UnknownInsKey     — shape is absent; feed any sample of it
    UnknownModifiers  — shape known, but a modifier we've never seen
    InsufficientBasis — shape known, but the operand vector falls outside
                        current null space (need more variation, not new shape)

Usage:
    python Tools/probe.py --arch sm_120 "S2R R7, SR_TID.X"
    python Tools/probe.py --arch sm_120 --repo path/to/repo.txt "LDG.E R2, [R4]"
    python Tools/probe.py --arch sm_120 --file failing_lines.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsParser import CuInsParser


# Match the sm_120 dataflow-hint stripper the parser uses on .cuasm input,
# so users can paste lines straight out of nvdisasm output.
import re
_ANNOT_RE = re.compile(r"\s*(?:&[A-Za-z_]\w*=\S+|\?\w+)")


def _default_repo(arch: str) -> Path:
    vnum = int(arch.removeprefix("sm_"))
    return REPO_ROOT / "CuAsm" / "InsAsmRepos" / f"DefaultInsAsmRepos.sm_{vnum}.txt"


def _strip_clutter(line: str) -> str:
    """Strip bracketed control codes, /*offset*/ comments, trailing ';',
    and nvdisasm sm_120 dataflow hints. Idempotent on already-clean input."""
    s = line.strip()
    # bracket-prefix control codes
    s = re.sub(r"^\[[^\]]*\]\s*", "", s)
    # /*0x..*/ offset comments
    s = re.sub(r"/\*[^*]*\*/\s*", "", s)
    # sm_120 dataflow hints
    s = _ANNOT_RE.sub("", s)
    # trailing semicolon
    s = s.rstrip(";").strip()
    return s


def probe(repo: CuInsAssemblerRepos, parser: CuInsParser, line: str) -> tuple[str, str]:
    """Return (status, detail). status in {OK, UnknownInsKey, UnknownModifiers,
    InsufficientBasis, ParseError}."""
    cleaned = _strip_clutter(line)
    try:
        ins_key, ins_vals, ins_modi = parser.parse(cleaned, 0, 0)
    except Exception as e:
        return "ParseError", f"{e}"

    if ins_key not in repo.m_InsAsmDict:
        return "UnknownInsKey", f"shape {ins_key!r} absent; feed any sample"

    ins_asm = repo.m_InsAsmDict[ins_key]
    brief, info = ins_asm.canAssemble(ins_vals, ins_modi)
    if brief is None:
        return "OK", f"shape {ins_key} assembleable; vals={ins_vals}, modi={ins_modi}"
    if brief == "NewModi":
        return "UnknownModifiers", info
    if brief == "NewVals":
        return ("InsufficientBasis",
                f"shape {ins_key} known; vals {ins_vals} outside current null space. "
                f"Feed samples that vary these operand positions.")
    return brief, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True, help="e.g. sm_120")
    ap.add_argument("--repo", type=Path, default=None,
                    help="Repo file; default: DefaultInsAsmRepos.<arch>.txt")
    ap.add_argument("--file", type=Path, default=None,
                    help="Read SASS lines from a file (one per line)")
    ap.add_argument("instruction", nargs="?",
                    help="A single SASS instruction; ignored if --file given")
    args = ap.parse_args()

    repo_path = args.repo or _default_repo(args.arch)
    repo = CuInsAssemblerRepos(str(repo_path), arch=args.arch)
    parser = CuInsParser(args.arch)

    if args.file:
        lines = [ln for ln in args.file.read_text().splitlines() if ln.strip()]
    elif args.instruction:
        lines = [args.instruction]
    else:
        ap.error("provide an instruction or --file")

    width = max((len(_strip_clutter(ln)) for ln in lines), default=0)
    width = min(width, 60)
    for ln in lines:
        status, detail = probe(repo, parser, ln)
        cleaned = _strip_clutter(ln)
        print(f"{cleaned:<{width}}  [{status}]  {detail}")


if __name__ == "__main__":
    main()
