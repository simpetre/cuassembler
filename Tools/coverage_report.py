"""Coverage report for a DefaultInsAsmRepos.<arch>.txt.

Reports per-InsKey-shape: number of samples, modifier count, null-space
basis dimension, and whether the basis is "saturated" (a synthetic new
instance with all-zero operand values is predictable by the current null
space). The last is a coarse proxy for "could I assemble an arbitrary
new instance of this shape?" — useful for picking shapes that still need
more samples.

Usage:
    python Tools/coverage_report.py --arch sm_120
    python Tools/coverage_report.py --arch sm_120 --repo path/to/repo.txt
    python Tools/coverage_report.py --arch sm_120 --compare sm_86
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.config import Config


def _default_repo(arch: str) -> Path:
    vnum = int(arch.removeprefix("sm_"))
    return REPO_ROOT / "CuAsm" / "InsAsmRepos" / f"DefaultInsAsmRepos.sm_{vnum}.txt"


def _shape_summary(ins_asm) -> dict:
    """Per-shape stats. `ins_asm` is a CuInsAssembler."""
    n_samples = len(ins_asm.m_InsRepos)
    n_modi = len(ins_asm.m_InsModiSet)
    n_vals = len(ins_asm.m_InsRepos[0][0]) if n_samples else 0
    # Null matrix dimensions: rows = null-space dim, cols = modi+vals
    nullmat = ins_asm.m_ValNullMat
    if nullmat is None:
        null_dim = 0
    else:
        # sympy Matrix; rows() returns row count
        null_dim = nullmat.rows
    # Basis dimension = total dim - null dim
    total_dim = n_modi + n_vals
    basis_dim = total_dim - null_dim
    return dict(
        samples=n_samples,
        modi=n_modi,
        vals=n_vals,
        basis_dim=basis_dim,
        null_dim=null_dim,
        total_dim=total_dim,
    )


def _coverage(repo: CuInsAssemblerRepos):
    by_family = {}
    for key, ins_asm in repo.items():
        family = key.split("_", 1)[0]
        by_family.setdefault(family, []).append((key, _shape_summary(ins_asm)))
    return by_family


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True, help="e.g. sm_120")
    ap.add_argument("--repo", type=Path, default=None,
                    help="Repo file; default: CuAsm/InsAsmRepos/DefaultInsAsmRepos.<arch>.txt")
    ap.add_argument("--compare", default=None,
                    help="Another arch to compare shape coverage against, e.g. sm_86")
    ap.add_argument("--summary", action="store_true",
                    help="Only print top-line totals; skip per-shape table.")
    args = ap.parse_args()

    repo_path = args.repo or _default_repo(args.arch)
    repo = CuInsAssemblerRepos(str(repo_path), arch=args.arch)

    shapes = list(repo.items())
    total_samples = sum(len(a.m_InsRepos) for _, a in shapes)
    print(f"=== {args.arch}  ({repo_path.name}) ===")
    print(f"InsKey shapes:    {len(shapes)}")
    print(f"Total samples:    {total_samples}")

    if args.compare:
        cmp_path = _default_repo(args.compare)
        if cmp_path.exists():
            cmp_repo = CuInsAssemblerRepos(str(cmp_path), arch=args.compare)
            cmp_keys = set(cmp_repo.m_InsAsmDict)
            our_keys = set(repo.m_InsAsmDict)
            missing = cmp_keys - our_keys
            extra = our_keys - cmp_keys
            print(f"vs {args.compare}: {len(cmp_keys)} shapes; "
                  f"{len(our_keys & cmp_keys)} overlap, "
                  f"{len(missing)} missing on our side, "
                  f"{len(extra)} unique on our side.")

    if args.summary:
        return

    print()
    print(f"{'InsKey':<32} {'samples':>7} {'modi':>5} {'vals':>5} "
          f"{'basis':>5} {'null':>5}")
    print("-" * 70)
    by_family = _coverage(repo)
    for fam in sorted(by_family):
        for key, stats in sorted(by_family[fam]):
            print(f"{key:<32} {stats['samples']:>7} {stats['modi']:>5} "
                  f"{stats['vals']:>5} {stats['basis_dim']:>5} "
                  f"{stats['null_dim']:>5}")


if __name__ == "__main__":
    main()
