#!/usr/bin/env python3
"""AtomicChat inkling2_loop GGUF -> the layout rdna4-amd-infer's loader reads (trial converter, 2026-11-26).
  ink_loopconv.py SRC.gguf DST.gguf A|B
Adds the uint32 keys loop.core_start / loop.core_end / num_loops (from the file's loop.start / loop.end / loop.count)
and two global F32 tensors, loop_res_query.weight and loop_kv_res_query.weight (2 * n_core * dim each), built from the
per-block blk.N.loop_msg_query.weight [dim, 4] tensors. Variant A: rows (0,1) -> res, (2,3) -> kv. B: (0,2) -> res,
(1,3) -> kv. All existing tensor data is copied byte for byte."""
import struct, sys, os

src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(src, "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr():
    return f.read(rd("Q"))
S = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rv(t):
    if t in S: return rd(S[t])
    if t == 8: return rstr()
    if t == 9:
        et, n = rd("I"), rd("Q")
        if et in S:
            f.seek(struct.calcsize(S[et]) * n, 1); return None
        return [rv(et) for _ in range(n)]
    raise SystemExit("unknown kv type %d" % t)

assert f.read(4) == b"GGUF"
ver, nt, nk = rd("I"), rd("Q"), rd("Q")
kv0 = f.tell(); kv = {}; align = 32
for _ in range(nk):
    k = rstr().decode(); t = rd("I"); kv[k] = rv(t)
    if k == "general.alignment": align = kv[k]
kv1 = f.tell()
tens = {}
for _ in range(nt):
    name = rstr().decode(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
    tens[name] = (dims, ty, off)
ti1 = f.tell()
data0 = (ti1 + align - 1) // align * align
size = os.path.getsize(src)
A = "inkling2_loop."
cs, ce, nl, dim = kv[A + "loop.start"], kv[A + "loop.end"], kv[A + "loop.count"], kv[A + "embedding_length"]
res, kvq = b"", b""
for i in range(cs, ce):
    dims, ty, off = tens["blk.%d.loop_msg_query.weight" % i]
    assert ty == 0 and dims == [dim, 4], (i, dims, ty)
    f.seek(data0 + off); rows = [f.read(dim * 4) for _ in range(4)]
    a, b = [tuple(int(c) for c in part) for part in variant.split("-")]  # e.g. 03-12: rows 0,3 -> res; 1,2 -> kv
    res += rows[a[0]] + rows[a[1]]; kvq += rows[b[0]] + rows[b[1]]
nc = ce - cs
data_len = size - data0
pad = (-data_len) % align
off_res = data_len + pad
off_kv = off_res + len(res) + ((-len(res)) % align)
def u32kv(k, v):
    k = (A + k).encode(); return struct.pack("<Q", len(k)) + k + struct.pack("<II", 4, v)
def tinfo(name, n, off):
    name = name.encode(); return struct.pack("<Q", len(name)) + name + struct.pack("<I", 2) + struct.pack("<QQ", dim, n) + struct.pack("<IQ", 0, off)
f.seek(kv0); kvs = f.read(kv1 - kv0); tis = f.read(ti1 - kv1)
head = (b"GGUF" + struct.pack("<IQQ", ver, nt + 2, nk + 3) + kvs + u32kv("loop.core_start", cs) + u32kv("loop.core_end", ce) +
        u32kv("num_loops", nl) + tis + tinfo("loop_res_query.weight", 2 * nc, off_res) + tinfo("loop_kv_res_query.weight", 2 * nc, off_kv))
head += b"\0" * ((-len(head)) % align)
with open(dst + ".part", "wb") as o:
    o.write(head); f.seek(data0)
    while True:
        b = f.read(64 << 20)
        if not b: break
        o.write(b)
    o.write(b"\0" * pad + res + b"\0" * ((-len(res)) % align) + kvq)
os.replace(dst + ".part", dst)
print("variant", variant, "core", cs, ce, "loops", nl, "added", len(res) + len(kvq), "bytes; wrote", dst)
