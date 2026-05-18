"""Build a DefaultInsAsmRepos.<arch>.txt from a cleaned nvdisasm feeder file.

The feeder file is plain SASS text (e.g. nvdisasm output) with metadata tokens
stripped (`&wr=...`, `&req={...}`, `?transX`), preserving BRA backtick syntax.
Re-run whenever an agent emits an InsKey the current repo doesn't cover:
disassemble a new sample cubin, clean it, then re-run this to extend the repo.

Example:
    python Tools/build_sm120_repos.py feed.txt --arch sm_120
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("feeder", type=Path, help="Cleaned SASS feeder text file.")
    ap.add_argument("--arch", required=True, help="SM arch, e.g. sm_120.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output repo path (default: CuAsm/InsAsmRepos/DefaultInsAsmRepos.<arch>.txt).",
    )
    args = ap.parse_args()

    out = args.out or (
        REPO_ROOT / "CuAsm" / "InsAsmRepos" / f"DefaultInsAsmRepos.{args.arch}.txt"
    )

    repos = CuInsAssemblerRepos(arch=args.arch)
    feeder = CuInsFeeder(str(args.feeder))
    repos.update(feeder)
    repos.save2file(str(out))

    print(f"saved {out}")
    print(f"repo size: {len(repos)}")


if __name__ == "__main__":
    main()
