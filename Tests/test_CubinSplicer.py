# -*- coding: utf-8 -*-
"""Tests for CuAsm.utils.CubinSplicer.

Run with:
    cd /content/CuAssembler && python -m pytest Tests/test_CubinSplicer.py -v

Requires `nvcc` on PATH to build template cubins. If nvcc is unavailable,
the tests skip cleanly.
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

# Make `CuAsm` importable from a checkout that's not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from elftools.elf.elffile import ELFFile

from CuAsm.CuNVInfo import CuNVInfo
from CuAsm.utils.CubinSplicer import (
    CubinSpliceError,
    splice_kernel_into_cubin,
)


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


_STUB_SRC = """\
extern "C" __global__ void kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] > 0.f ? a[i] : 0.f;
}
"""


def _nvcc_or_skip():
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")


def _build_template(tmp_path: Path, arch: str = "sm_120") -> bytes:
    _nvcc_or_skip()
    src = tmp_path / "stub.cu"
    cubin = tmp_path / "stub.cubin"
    src.write_text(_STUB_SRC)
    r = subprocess.run(
        ["nvcc", f"-arch={arch}", "-cubin", "-o", str(cubin), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"nvcc failed for {arch}: {(r.stderr or r.stdout).strip()}")
    return cubin.read_bytes()


def _section_bytes(cubin: bytes, name: str) -> bytes:
    elf = ELFFile(io.BytesIO(cubin))
    sec = elf.get_section_by_name(name)
    assert sec is not None, f"no section {name!r}"
    return bytes(sec.data())


def _section_header(cubin: bytes, name: str) -> dict:
    elf = ELFFile(io.BytesIO(cubin))
    sec = elf.get_section_by_name(name)
    assert sec is not None, f"no section {name!r}"
    return dict(sec.header)


def _read_exit_offsets(cubin: bytes, kernel_name: str = "kernel") -> list[int]:
    info = _section_bytes(cubin, f".nv.info.{kernel_name}")
    for attr, val in CuNVInfo(info, arch="sm_120"):
        if attr == "EIATTR_EXIT_INSTR_OFFSETS":
            return list(val)
    return []


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_identity_splice_preserves_text_section(tmp_path):
    """Re-splicing a kernel's own bytes back in must leave .text.kernel bytes
    and section sizes unchanged, and the result must still parse as an ELF."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    orig_exits = _read_exit_offsets(tmpl)
    orig_hdr = _section_header(tmpl, ".text.kernel")

    out = splice_kernel_into_cubin(
        tmpl,
        kernel_name="kernel",
        kernel_bytes=orig_text,
        num_regs=(orig_hdr["sh_info"] >> 24) & 0xFF or 10,
        exit_offsets=orig_exits,
    )

    assert len(out) == len(tmpl), "identity splice should not change file size"
    assert _section_bytes(out, ".text.kernel") == orig_text
    assert _read_exit_offsets(out) == orig_exits
    # ELF parses cleanly
    ELFFile(io.BytesIO(out))


def test_same_size_body_swap(tmp_path):
    """Replace the body with a same-length sequence of distinct bytes; section
    sizes / offsets must not change."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    new_text = bytes([0xAA] * len(orig_text))

    out = splice_kernel_into_cubin(
        tmpl,
        kernel_name="kernel",
        kernel_bytes=new_text,
        num_regs=12,
        exit_offsets=[0x70, 0x100],
    )

    assert len(out) == len(tmpl)
    assert _section_bytes(out, ".text.kernel") == new_text
    new_hdr = _section_header(out, ".text.kernel")
    orig_hdr = _section_header(tmpl, ".text.kernel")
    assert new_hdr["sh_offset"] == orig_hdr["sh_offset"]
    assert new_hdr["sh_size"] == orig_hdr["sh_size"]
    # num_regs in sh_info high byte
    assert (new_hdr["sh_info"] >> 24) & 0xFF == 12
    # symbol index in low 24 bits preserved
    assert new_hdr["sh_info"] & 0x00FFFFFF == orig_hdr["sh_info"] & 0x00FFFFFF


def test_larger_body_shifts_downstream_offsets(tmp_path):
    """A body that's larger than the template's must shift every section past
    .text.kernel by exactly the delta (rounded to alignment) and grow the file."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    align = _section_header(tmpl, ".text.kernel")["sh_addralign"]
    delta = align  # one alignment unit larger
    new_text = orig_text + bytes([0xBB] * delta)
    assert len(new_text) % align == 0

    out = splice_kernel_into_cubin(
        tmpl,
        kernel_name="kernel",
        kernel_bytes=new_text,
        num_regs=8,
        exit_offsets=[0x70, 0x100],
    )

    assert len(out) == len(tmpl) + delta
    new_hdr = _section_header(out, ".text.kernel")
    orig_hdr = _section_header(tmpl, ".text.kernel")
    assert new_hdr["sh_size"] == len(new_text)
    assert new_hdr["sh_offset"] == orig_hdr["sh_offset"]  # unchanged
    # A downstream section must have shifted by `delta`
    orig_constant = _section_header(tmpl, ".nv.constant0.kernel")
    new_constant = _section_header(out, ".nv.constant0.kernel")
    assert new_constant["sh_offset"] == orig_constant["sh_offset"] + delta
    # ELF still parses
    ELFFile(io.BytesIO(out))


def test_exit_offsets_round_trip(tmp_path):
    """The EIATTR_EXIT_INSTR_OFFSETS payload must reflect the caller's list."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")

    custom_exits = [0x10, 0x40, 0x80]
    out = splice_kernel_into_cubin(
        tmpl,
        kernel_name="kernel",
        kernel_bytes=orig_text,
        num_regs=10,
        exit_offsets=custom_exits,
    )
    assert _read_exit_offsets(out) == custom_exits


def test_kernel_symbol_size_updated(tmp_path):
    """The .symtab entry for `kernel` must have st_size updated to match the
    new body length."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    align = _section_header(tmpl, ".text.kernel")["sh_addralign"]
    new_text = orig_text + bytes([0xCC] * align)

    out = splice_kernel_into_cubin(
        tmpl,
        kernel_name="kernel",
        kernel_bytes=new_text,
        num_regs=10,
        exit_offsets=[0x70, 0x100],
    )

    elf = ELFFile(io.BytesIO(out))
    symtab = elf.get_section_by_name(".symtab")
    assert symtab is not None
    sym = symtab.get_symbol_by_name("kernel")
    assert sym is not None and len(sym) == 1
    assert sym[0]["st_size"] == len(new_text), \
        f"expected st_size={len(new_text)}, got {sym[0]['st_size']}"


def test_rejects_misaligned_kernel_size(tmp_path):
    """Reject kernel bytes whose length isn't a multiple of sh_addralign."""
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    # +1 byte → definitely not 128-aligned
    bad = orig_text + b"\x00"
    with pytest.raises(CubinSpliceError, match="sh_addralign"):
        splice_kernel_into_cubin(
            tmpl,
            kernel_name="kernel",
            kernel_bytes=bad,
            num_regs=10,
            exit_offsets=[0x70, 0x100],
        )


def test_rejects_missing_kernel(tmp_path):
    tmpl = _build_template(tmp_path)
    orig_text = _section_bytes(tmpl, ".text.kernel")
    with pytest.raises(CubinSpliceError, match="does not contain"):
        splice_kernel_into_cubin(
            tmpl,
            kernel_name="nonesuch",
            kernel_bytes=orig_text,
            num_regs=10,
            exit_offsets=[],
        )
