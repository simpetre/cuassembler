"""Splice agent-assembled kernel bytes into an nvcc-built template cubin.

Bypasses `CuAsmParser.saveAsCubin`'s from-scratch ELF writer (which is broken
on newer arches like sm_120 / Blackwell) by inheriting cubin format
correctness from nvcc and only rewriting the kernel body plus a handful of
metadata fields.

Public API:
    splice_kernel_into_cubin(template_cubin, kernel_name, kernel_bytes,
                             num_regs, exit_offsets) -> bytes
"""

from __future__ import annotations

import struct
from io import BytesIO

from elftools.elf.elffile import ELFFile

from CuAsm.CuNVInfo import CuNVInfo


# ELF64 file header field offsets
_E_PHOFF, _E_SHOFF = 32, 40
_E_PHENTSIZE, _E_PHNUM = 54, 56
_E_SHENTSIZE, _E_SHNUM = 58, 60

# ELF64 section header field offsets (entry size 64)
_SH_OFFSET, _SH_SIZE, _SH_INFO = 24, 32, 44
_SH_ENTSIZE = 64

# ELF64 program header field offsets (entry size 56)
_P_OFFSET, _P_FILESZ, _P_MEMSZ = 8, 32, 40
_P_ENTSIZE = 56

# ELF64 symbol entry offsets (entry size 24)
_ST_SIZE_OFF = 16
_SYMTAB_ENTSIZE = 24

_SHT_NOBITS = 8
_SHT_RELA = 4

# `EIATTR_*_OFFSETS` attribute that we patch with the caller's exit_offsets.
_OFFSET_ATTR_HANDLED = "EIATTR_EXIT_INSTR_OFFSETS"

# Other `*_OFFSETS` attributes that would also be invalidated by replacing
# the kernel body. The conservative scheduler does not emit instructions
# that require these; if any are present in the template, refuse rather
# than silently leaving stale offsets.
_OFFSET_ATTRS_UNSUPPORTED = frozenset({
    "EIATTR_S2RCTAID_INSTR_OFFSETS",
    "EIATTR_LD_CACHEMOD_INSTR_OFFSETS",
    "EIATTR_ATOM_SYS_INSTR_OFFSETS",
    "EIATTR_COOP_GROUP_INSTR_OFFSETS",
    "EIATTR_INT_WARP_WIDE_INSTR_OFFSETS",
    "EIATTR_INDIRECT_BRANCH_TARGETS",
})


class CubinSpliceError(ValueError):
    """Raised when the template cubin can't safely accept the splice."""


def _u64(buf, off):
    return struct.unpack_from("<Q", buf, off)[0]


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def _put_u64(buf, off, val):
    struct.pack_into("<Q", buf, off, val)


def _put_u32(buf, off, val):
    struct.pack_into("<I", buf, off, val)


def _arch_from_elf(template_cubin: bytes) -> str:
    """Recover the sm_NN string from the ELF header's e_flags (low byte is the
    SM version on pre-Blackwell; bits [15:8] extend it on Blackwell+)."""
    e_flags = _u32(template_cubin, 48)
    sm = e_flags & 0xFF
    sm_hi = (e_flags >> 8) & 0xFF
    if sm_hi >= 100:
        sm = sm_hi
    return f"sm_{sm}"


def splice_kernel_into_cubin(
    template_cubin: bytes,
    kernel_name: str,
    kernel_bytes: bytes,
    num_regs: int,
    exit_offsets: list[int],
) -> bytes:
    """Overwrite the body of `kernel_name` in `template_cubin` and return the
    spliced cubin bytes.

    Restrictions (the splice will raise `CubinSpliceError` otherwise):
      * Exactly one `.text.*` section in the template, named `.text.<kernel_name>`.
      * No `.rela.*` section targets the kernel's text section.
      * `.nv.info.<kernel_name>` contains no instruction-offset attributes
        besides `EIATTR_EXIT_INSTR_OFFSETS` (which we rewrite).
      * `len(kernel_bytes)` is a multiple of the template's
        `.text.<kernel_name>` sh_addralign. The caller's scheduler pads to
        0x80; this catches accidental misuse.
    """
    text_name = f".text.{kernel_name}"
    info_name = f".nv.info.{kernel_name}"

    elf = ELFFile(BytesIO(template_cubin))

    # ----- Locate sections we care about ----- #
    text_sec = None
    text_idx = None
    info_sec = None
    info_idx = None
    ginfo_sec = None          # the global `.nv.info` (holds EIATTR_REGCOUNT)
    symtab = None
    symtab_idx = None
    text_section_count = 0

    for idx, sec in enumerate(elf.iter_sections()):
        if sec.name.startswith(".text."):
            text_section_count += 1
            if sec.name == text_name:
                text_sec, text_idx = sec, idx
        elif sec.name == info_name:
            info_sec, info_idx = sec, idx
        elif sec.name == ".nv.info":
            ginfo_sec = sec
        elif sec.name == ".symtab":
            symtab, symtab_idx = sec, idx

    if text_section_count != 1:
        raise CubinSpliceError(
            f"template has {text_section_count} `.text.*` sections; splicer requires exactly 1"
        )
    if text_sec is None:
        raise CubinSpliceError(
            f"template does not contain section {text_name!r}"
        )
    if info_sec is None:
        raise CubinSpliceError(
            f"template does not contain section {info_name!r}"
        )
    if symtab is None:
        raise CubinSpliceError("template has no .symtab")

    # ----- Pre-flight checks ----- #
    addralign = text_sec.header["sh_addralign"] or 1
    if len(kernel_bytes) % addralign != 0:
        raise CubinSpliceError(
            f"kernel_bytes length {len(kernel_bytes)} is not a multiple of "
            f"{text_name}'s sh_addralign={addralign}"
        )

    # No relocations targeting the kernel text
    for sec in elf.iter_sections():
        if sec.header["sh_type"] == _SHT_RELA and sec.header["sh_info"] == text_idx:
            raise CubinSpliceError(
                f"template has relocation section {sec.name!r} targeting {text_name}; "
                "splicer does not support kernels with relocations"
            )

    # ----- Build new .nv.info.<kernel> via CuNVInfo ----- #
    arch = _arch_from_elf(template_cubin)
    nvinfo = CuNVInfo(bytes(info_sec.data()), arch=arch)

    for attr_name, _ in nvinfo:
        if attr_name in _OFFSET_ATTRS_UNSUPPORTED:
            raise CubinSpliceError(
                f"{info_name} contains unsupported offset attribute {attr_name!r}; "
                "the splice would leave its offsets stale. Bailing rather than "
                "produce a silently-corrupt cubin."
            )

    found_exit_attr = False
    new_attrs = []
    for attr_name, val in nvinfo.m_AttrList:
        if attr_name == _OFFSET_ATTR_HANDLED:
            new_attrs.append((attr_name, list(exit_offsets)))
            found_exit_attr = True
        else:
            new_attrs.append((attr_name, val))
    if not found_exit_attr and exit_offsets:
        # Append EIATTR_EXIT_INSTR_OFFSETS if absent. nvcc-built templates
        # almost always have it, so this branch is defensive.
        new_attrs.append((_OFFSET_ATTR_HANDLED, list(exit_offsets)))
    nvinfo.m_AttrList = new_attrs
    new_info_bytes = nvinfo.serialize()

    # ----- Patch EIATTR_REGCOUNT in the global `.nv.info` ----- #
    # The SM honors EIATTR_REGCOUNT for register allocation; the nvcc template
    # was built from an empty kernel (REGCOUNT≈4), so without this every spliced
    # kernel that uses a register past ~R12 gets an out-of-range fault at launch.
    # The value is [kernel_sym_idx, count]; we keep the symbol and set the count.
    # Same byte-size, so it's an in-place overwrite that doesn't shift offsets.
    patched_ginfo_bytes = None
    ginfo_off = None
    if ginfo_sec is not None:
        ginfo = CuNVInfo(bytes(ginfo_sec.data()), arch=arch)
        changed = False
        for i, (attr_name, val) in enumerate(ginfo.m_AttrList):
            if attr_name == 'EIATTR_REGCOUNT' and isinstance(val, list) and len(val) == 2:
                ginfo.m_AttrList[i] = (attr_name, [val[0], num_regs])
                changed = True
        if changed:
            cand = ginfo.serialize()
            if len(cand) != ginfo_sec.header["sh_size"]:
                raise CubinSpliceError(
                    "patching EIATTR_REGCOUNT changed `.nv.info` size "
                    f"({ginfo_sec.header['sh_size']} -> {len(cand)}); refusing to corrupt layout"
                )
            patched_ginfo_bytes = cand
            ginfo_off = ginfo_sec.header["sh_offset"]

    # ----- Compute edits, sorted by file offset ----- #
    text_off = text_sec.header["sh_offset"]
    text_old_size = text_sec.header["sh_size"]
    info_off = info_sec.header["sh_offset"]
    info_old_size = info_sec.header["sh_size"]

    edits = sorted(
        [
            (info_off, info_old_size, new_info_bytes, info_idx),
            (text_off, text_old_size, kernel_bytes, text_idx),
        ],
        key=lambda e: e[0],
    )

    # Apply edits to a bytearray; track cumulative shift for each original offset.
    result = bytearray(template_cubin)
    cumulative = 0
    # remember the *applied* deltas at each original offset for later lookup
    edit_points = []  # list of (orig_off, orig_end, delta_at_or_after_end)
    for orig_off, old_size, new_data, _sec_idx in edits:
        actual_off = orig_off + cumulative
        result[actual_off : actual_off + old_size] = new_data
        delta = len(new_data) - old_size
        edit_points.append((orig_off, orig_off + old_size, delta))
        cumulative += delta

    def delta_at(orig_offset: int) -> int:
        """Cumulative shift applied to any byte that, in the *original* cubin,
        sat at `orig_offset`. A byte that lay strictly past the end of an edit
        moves; a byte inside or before an edit stays where it is (this matters
        only for the small in-edit case, which we don't query)."""
        d = 0
        for _, orig_end, e_delta in edit_points:
            if orig_offset >= orig_end:
                d += e_delta
        return d

    # ----- Rewrite section header table ----- #
    # ELF header tells us where it lives (in the ORIGINAL file); it may itself
    # need to shift, so we read e_shoff from the (still-unmodified) header
    # bytes, compute its new position, and write entries to the new position.
    orig_e_shoff = _u64(template_cubin, _E_SHOFF)
    orig_e_phoff = _u64(template_cubin, _E_PHOFF)
    shnum = struct.unpack_from("<H", template_cubin, _E_SHNUM)[0]
    phnum = struct.unpack_from("<H", template_cubin, _E_PHNUM)[0]

    new_e_shoff = orig_e_shoff + delta_at(orig_e_shoff)
    new_e_phoff = orig_e_phoff + delta_at(orig_e_phoff)

    # First, find which PT_LOAD (if any) covers .text.<kernel> so we update its
    # filesz/memsz rather than just shifting it.
    text_delta = len(kernel_bytes) - text_old_size
    info_delta = len(new_info_bytes) - info_old_size

    for i in range(shnum):
        # section header *i* lives at orig_e_shoff + i*64 in the original cubin,
        # which in `result` is at new_e_shoff + i*64.
        hdr_at = new_e_shoff + i * _SH_ENTSIZE
        sh_offset = _u64(result, hdr_at + _SH_OFFSET)
        sh_size = _u64(result, hdr_at + _SH_SIZE)
        sh_info = _u32(result, hdr_at + _SH_INFO)
        sh_type = _u32(result, hdr_at + 4)

        if i == text_idx:
            _put_u64(result, hdr_at + _SH_SIZE, len(kernel_bytes))
            # sh_info: preserve low 24 bits (symbol index), set high byte = num_regs
            new_sh_info = (sh_info & 0x00FFFFFF) | ((num_regs & 0xFF) << 24)
            _put_u32(result, hdr_at + _SH_INFO, new_sh_info)
            # sh_offset stays the same (no edits precede .text.<kernel> by the
            # time we get here only if .nv.info.<kernel> came after, which is
            # not the typical layout; handle generically):
            _put_u64(result, hdr_at + _SH_OFFSET, sh_offset + delta_at(sh_offset))
        elif i == info_idx:
            _put_u64(result, hdr_at + _SH_SIZE, len(new_info_bytes))
            _put_u64(result, hdr_at + _SH_OFFSET, sh_offset + delta_at(sh_offset))
        else:
            # NOBITS sections don't occupy file space, but their sh_offset is
            # still a valid file position used for sorting; shift it the same
            # way as PROGBITS so layout invariants survive.
            _put_u64(result, hdr_at + _SH_OFFSET, sh_offset + delta_at(sh_offset))

    # ----- Rewrite program header table ----- #
    for i in range(phnum):
        hdr_at = new_e_phoff + i * _P_ENTSIZE
        p_offset = _u64(result, hdr_at + _P_OFFSET)
        p_filesz = _u64(result, hdr_at + _P_FILESZ)
        p_memsz = _u64(result, hdr_at + _P_MEMSZ)

        # Does this segment cover .text.<kernel> in the ORIGINAL file?
        covers_text = (p_offset <= text_off < p_offset + p_filesz)
        covers_info = (p_offset <= info_off < p_offset + p_filesz)
        new_p_offset = p_offset + delta_at(p_offset)
        new_p_filesz = p_filesz
        new_p_memsz = p_memsz
        if covers_text:
            new_p_filesz += text_delta
            new_p_memsz += text_delta
        if covers_info:
            new_p_filesz += info_delta
            new_p_memsz += info_delta

        _put_u64(result, hdr_at + _P_OFFSET, new_p_offset)
        _put_u64(result, hdr_at + _P_FILESZ, new_p_filesz)
        _put_u64(result, hdr_at + _P_MEMSZ, new_p_memsz)

    # ----- Update kernel symbol st_size ----- #
    # symtab moved to new_offset_for(orig_symtab_offset). Walk entries; the
    # entry whose st_shndx == text_idx and whose st_size matches the old kernel
    # size is the kernel symbol (cubins usually have only one FUNC at that
    # section, but we don't rely on the name string here).
    symtab_orig_off = symtab.header["sh_offset"]
    symtab_size = symtab.header["sh_size"]
    symtab_new_off = symtab_orig_off + delta_at(symtab_orig_off)
    nsyms = symtab_size // _SYMTAB_ENTSIZE
    for i in range(nsyms):
        ent_at = symtab_new_off + i * _SYMTAB_ENTSIZE
        st_shndx = struct.unpack_from("<H", result, ent_at + 6)[0]
        st_size = _u64(result, ent_at + _ST_SIZE_OFF)
        if st_shndx == text_idx and st_size == text_old_size:
            _put_u64(result, ent_at + _ST_SIZE_OFF, len(kernel_bytes))

    # ----- Patch ELF header's e_shoff / e_phoff ----- #
    _put_u64(result, _E_SHOFF, new_e_shoff)
    _put_u64(result, _E_PHOFF, new_e_phoff)

    # ----- Overwrite global `.nv.info` with the REGCOUNT-patched bytes ----- #
    # Same size, so its content merely shifted with the surrounding edits; write
    # the patched bytes at its shifted file offset.
    if patched_ginfo_bytes is not None:
        at = ginfo_off + delta_at(ginfo_off)
        result[at:at + len(patched_ginfo_bytes)] = patched_ginfo_bytes

    return bytes(result)
