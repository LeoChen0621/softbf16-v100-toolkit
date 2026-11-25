"""Turn `dumpbin /exports` output for cublas64_12.dll into a .def file.

    python make_cublas_def.py <cudalibs dir>

The import library is needed because the kernel calls cublasGemmEx for its
repair path, and the PyPI CUDA wheels ship DLLs but no .lib on Windows.
"""
import re
import sys
from pathlib import Path

libs = Path(sys.argv[1])
txt = (libs / "cublas_exports.txt").read_text(errors="ignore")
names = []
for line in txt.splitlines():
    m = re.match(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$", line)
    if m:
        names.append(m.group(1))
names = sorted(set(names))
if not names:
    print("no exports parsed - is cublas_exports.txt valid?")
    sys.exit(1)
(libs / "cublas.def").write_text(
    "LIBRARY cublas64_12.dll\nEXPORTS\n" + "".join("  %s\n" % n for n in names),
    encoding="ascii")
print("cublas.def: %d exports" % len(names))
