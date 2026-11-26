#!/usr/bin/env python3
"""Unsloth llama.cpp-style DSpark drafter (general.architecture=dflash, dspark_* tensors) -> the layout
rdna4-amd-infer's model_load_dspark reads (general.architecture=dspark). Trial converter, 2026-11-26.
  ink_dsparkconv.py SRC.gguf DST.gguf [--keep-bf16]
Renames metadata keys (dflash.* -> dspark.*) and tensors, derives dspark.attention.local_layer_ids from
sliding_window_pattern, drops the tokenizer keys (the engine uses the target model's), and stores BF16 tensors as F32
(same values) unless --keep-bf16. Run with a Python that has numpy."""
import struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
keep_bf16 = "--keep-bf16" in sys.argv
f = open(src, "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr():
    return f.read(rd("Q")).decode("utf-8")
S = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rv(t):
    if t in S: return rd(S[t])
    if t == 8: return rstr()
    if t == 9:
        et, n = rd("I"), rd("Q")
        return (et, [rv(et) for _ in range(n)])
    raise SystemExit("kv type %d" % t)
def wstr(s):
    b = s.encode(); return struct.pack("<Q", len(b)) + b
def wv(t, v):
    if t in S: return struct.pack("<" + S[t], v)
    if t == 8: return wstr(v)
    et, items = v
    return struct.pack("<IQ", et, len(items)) + b"".join(wv(et, x) for x in items)

assert f.read(4) == b"GGUF"
ver, nt, nk = rd("I"), rd("Q"), rd("Q")
kvs = []
for _ in range(nk):
    k = rstr(); t = rd("I"); kvs.append((k, t, rv(t)))
tens = []
for _ in range(nt):
    name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
    tens.append([name, dims, ty, off])
ALIGN = 32
data0 = (f.tell() + ALIGN - 1) // ALIGN * ALIGN

out_kv = []
pattern = None
for k, t, v in kvs:
    if k.startswith("tokenizer."): continue
    if k == "general.architecture": v = "dspark"
    if k == "dflash.attention.sliding_window_pattern": pattern = v[1]
    if k == "dflash.dspark.markov_rank": k = "dspark.markov_rank"
    elif k.startswith("dflash."): k = "dspark." + k[len("dflash."):]
    out_kv.append((k, t, v))
if pattern is None: raise SystemExit("no sliding_window_pattern")
local = [i for i, w in enumerate(pattern) if w]
out_kv.append(("dspark.attention.local_layer_ids", 9, (5, local)))

REN = {"dspark_fuse.weight": "fuse_weight", "mask_embd.weight": "mask_embed", "dspark_norm.weight": "markov.dflare_norm.weight",
       "joint_proj.bias": "markov.rnn_proj.bias", "joint_proj.weight": "markov.rnn_proj.weight",
       "dspark_hidden_out.bias": "markov.hidden_out.bias", "dspark_hidden_out.weight": "markov.hidden_out.weight",
       "markov_w1.weight": "markov.w1.weight"}
BYTES = {0: lambda n: n * 4, 1: lambda n: n * 2, 30: lambda n: n * 2, 8: lambda n: n // 32 * 34}
blobs, infos, off = [], [], 0
for name, dims, ty, o in tens:
    n = 1
    for d in dims: n *= d
    f.seek(data0 + o); raw = f.read(BYTES[ty](n))
    if ty == 30 and not keep_bf16:  # BF16 -> F32: the 16 bits are the upper half of the float32
        x = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        if "--q8" in sys.argv and len(dims) == 2 and dims[0] % 32 == 0:  # ggml Q8_0: per 32 values an f16 scale + 32 int8
            xb = x.reshape(-1, 32); d = np.abs(xb).max(axis=1) / 127.0
            inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0)
            q = np.rint(xb * inv[:, None]).astype(np.int8)
            rec = np.zeros(len(d), dtype=[("d", "<f2"), ("q", "i1", 32)]); rec["d"] = d.astype("<f2"); rec["q"] = q
            raw = rec.tobytes(); ty = 8
        else:
            raw = x.tobytes(); ty = 0
    new = REN.get(name, name)
    infos.append(wstr(new) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off))
    pad = (-len(raw)) % ALIGN
    blobs.append(raw + b"\0" * pad); off += len(raw) + pad
    if new != name: print("  %-28s -> %-28s %s type %d" % (name, new, dims, ty))
head = b"GGUF" + struct.pack("<IQQ", ver, len(tens), len(out_kv)) + b"".join(wstr(k) + struct.pack("<I", t) + wv(t, v) for k, t, v in out_kv) + b"".join(infos)
head += b"\0" * ((-len(head)) % ALIGN)
with open(dst, "wb") as o:
    o.write(head)
    for b in blobs: o.write(b)
print("local layers", local, "| keys", len(out_kv), "| tensors", len(tens), "| wrote", dst, off + len(head), "bytes")
