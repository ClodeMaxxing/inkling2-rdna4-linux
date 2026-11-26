#!/usr/bin/env python3
"""Linux stand-in for scripts/inkling2_requant.py --plan '^output\\.weight$=Q4_K' (which needs a Windows ggml DLL and a
module the repo does not ship). Requantises one Q8_0 tensor of a GGUF to Q4_K with ggml_quantize_chunk from a
libggml-base.so, and rewrites the file with every other tensor copied byte for byte. 2026-11-26.
  ink_requant_head.py SRC.gguf DST.gguf LIBGGML_BASE.so [TENSOR_NAME]
Run with a Python that has numpy."""
import ctypes, os, struct, sys
import numpy as np

src, dst, lib = sys.argv[1], sys.argv[2], sys.argv[3]
target = sys.argv[4] if len(sys.argv) > 4 else "output.weight"
f = open(src, "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr():
    return f.read(rd("Q"))
S = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skip(t):
    if t in S: f.seek(S[t], 1)
    elif t == 8: f.seek(rd("Q"), 1)
    else:
        et, n = rd("I"), rd("Q")
        if et in S: f.seek(S[et] * n, 1)
        else:
            for _ in range(n): skip(et)
assert f.read(4) == b"GGUF"
ver, nt, nk = rd("I"), rd("Q"), rd("Q")
kv0 = f.tell()
for _ in range(nk):
    rstr(); skip(rd("I"))
kv1 = f.tell()
tens = []
for _ in range(nt):
    name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
    tens.append([name, dims, ty, off])
ALIGN = 32
data0 = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
size = os.path.getsize(src)
order = sorted(range(nt), key=lambda i: tens[i][3])
nbytes = {}
for a, i in enumerate(order):
    end = tens[order[a + 1]][3] if a + 1 < nt else size - data0
    nbytes[i] = end - tens[i][3]  # includes padding up to the next tensor

ti = next(i for i in range(nt) if tens[i][0] == target.encode())
name, dims, ty, off = tens[ti]
cols, rows = dims[0], dims[1]
assert ty == 8 and cols % 256 == 0, (ty, dims)
f.seek(data0 + off)
blk = np.frombuffer(f.read(rows * (cols // 32) * 34), np.uint8).reshape(rows, cols // 32, 34)
d = blk[:, :, :2].copy().view(np.float16).astype(np.float32)
w = np.ascontiguousarray((d * blk[:, :, 2:].view(np.int8).astype(np.float32)).reshape(rows, cols))
g = ctypes.CDLL(lib)
g.ggml_quantize_init.argtypes = [ctypes.c_int]; g.ggml_quantize_init(12)
q = g.ggml_quantize_chunk
q.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
q.restype = ctypes.c_size_t
rb = cols // 256 * 144
out = (ctypes.c_uint8 * (rows * rb))()
n = q(12, w.ctypes.data, ctypes.addressof(out), 0, rows, cols, None)
assert n == rows * rb, (n, rows * rb)
new = bytes(out)
print("%s: Q8_0 %d MB -> Q4_K %d MB" % (target, rows * (cols // 32) * 34 >> 20, len(new) >> 20))

infos, blobs, o = [], [], 0
for i in range(nt):  # keep the table order; lay the data out again in that order
    nm, dm, t, of = tens[i]
    ln = len(new) if i == ti else nbytes[i]
    if i == ti: t = 12
    pad = (-ln) % ALIGN
    infos.append(struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dm)) + b"".join(struct.pack("<Q", x) for x in dm) + struct.pack("<IQ", t, o))
    blobs.append((i, of, ln, pad)); o += ln + pad
f.seek(kv0); kvs = f.read(kv1 - kv0)
head = b"GGUF" + struct.pack("<IQQ", ver, nt, nk) + kvs + b"".join(infos)
head += b"\0" * ((-len(head)) % ALIGN)
with open(dst + ".part", "wb") as w_:
    w_.write(head)
    for i, of, ln, pad in blobs:  # streamed: never more than 64 MB of a copied tensor in memory
        if i == ti:
            w_.write(new)
        else:
            f.seek(data0 + of); left = ln
            while left:
                b = f.read(min(left, 64 << 20)); w_.write(b); left -= len(b)
        w_.write(b"\0" * pad)
os.replace(dst + ".part", dst)
print("wrote", dst, len(head) + o, "bytes")
