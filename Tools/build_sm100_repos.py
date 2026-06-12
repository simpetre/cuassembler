"""Build or extend a DefaultInsAsmRepos.<arch>.txt from a feeder file.

The feeder file is SASS text from `cuobjdump --dump-sass` (or equivalent),
with dataflow annotations (`&wr=...`, `&req=...`, `&rd=...`, `?transX`,
`?WAIT*_END_GROUP`) stripped. See Tools/training_corpus/build_feeder.sh for
the canonical preparation.

By default this is **incremental**: an existing repo at `--out` is loaded
first and the feeder's instructions add to it (shapes already present get
extra samples; new shapes are introduced). Use `--clean` to start from an
empty repo (the previous behavior of this script).

Examples:
    # Bootstrap from a curated training corpus, growing whatever is already there:
    python Tools/build_sm100_repos.py feed.sm100.txt --arch sm_100

    # Start fresh:
    python Tools/build_sm100_repos.py feed.sm100.txt --arch sm_100 --clean

    # Single-shot extension from one cubin's disassembly (e.g. resolving a gap
    # surfaced by an agent failure):
    python Tools/build_sm100_repos.py gap_fix.txt --arch sm_100
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
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Start from an empty repo instead of accumulating into the existing one.",
    )
    args = ap.parse_args()

    out = args.out or (
        REPO_ROOT / "CuAsm" / "InsAsmRepos" / f"DefaultInsAsmRepos.{args.arch}.txt"
    )

    if args.clean or not out.exists():
        repos = CuInsAssemblerRepos(arch=args.arch)
        if args.clean and out.exists():
            print(f"--clean: ignoring existing {out}")
        else:
            print(f"starting fresh (no existing repo at {out})")
    else:
        repos = CuInsAssemblerRepos(str(out), arch=args.arch)
        print(f"loaded {out}  ({len(repos)} shapes)")

    feeder = CuInsFeeder(str(args.feeder))
    repos.update(feeder)
    repos.save2file(str(out))

    print(f"saved {out}")
    print(f"repo size: {len(repos)}")


if __name__ == "__main__":
    main()
