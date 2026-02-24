#!/usr/bin/env python3
"""Clear PT_GNU_STACK executable flag from ELF shared objects.

ORT MLAS assembly marks the stack as executable (RWE). Newer kernels
(6.18+) refuse to dlopen such libraries. This script patches the ELF
program header to change RWE -> RW (clears PF_X flag).
"""
import glob
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1

for pattern in sys.argv[1:]:
    for so in glob.glob(pattern, recursive=True):
        with open(so, "r+b") as f:
            magic = f.read(4)
            if magic != b"\x7fELF":
                continue
            ei_class = f.read(1)[0]
            if ei_class != 2:  # 64-bit only
                continue
            f.seek(32)
            e_phoff = struct.unpack("<Q", f.read(8))[0]
            f.seek(54)
            e_phentsize = struct.unpack("<H", f.read(2))[0]
            e_phnum = struct.unpack("<H", f.read(2))[0]
            for i in range(e_phnum):
                off = e_phoff + i * e_phentsize
                f.seek(off)
                p_type = struct.unpack("<I", f.read(4))[0]
                if p_type != PT_GNU_STACK:
                    continue
                p_flags = struct.unpack("<I", f.read(4))[0]
                if p_flags & PF_X:
                    f.seek(off + 4)
                    f.write(struct.pack("<I", p_flags & ~PF_X))
                    print(f"Cleared execstack: {so}")
                break
