#!/usr/bin/env python3
"""Fill ie-serve's context in steps and read card memory after each (2026-11-26).
  ctx_probe.py LIMIT_MIB SIZE_K [SIZE_K ...] -- SRC [SRC ...]
Each step is a fresh prompt of about SIZE_K thousand tokens with a code word at the top that the model must repeat.
Stops before a step when card memory already exceeds LIMIT_MIB, or on the first error."""
import glob, json, sys, time, urllib.request

a = sys.argv[1:]
limit = int(a[0]); cut = a.index("--"); sizes = [int(x) for x in a[1:cut]]; srcs = a[cut + 1:]
big = "".join(open(f, encoding="utf-8", errors="replace").read() for f in srcs)
vf = [p for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")][0]
def vram():
    return int(open(vf).read()) // 1048576
print("start vram", vram(), "MiB; source chars", len(big), flush=True)
for k in sizes:
    if vram() > limit:
        print("stop: card already at", vram(), "MiB before the", k, "K step"); break
    word = "PELICAN-%d" % (7000 + k)
    prompt = ("The secret code word is %s. Remember it.\n\nBelow is C source code.\n\n```c\n" % word + big[: int(k * 1000 * 2.44)] +
              "\n```\n\nFirst state the secret code word given at the very top. Then say in one sentence what this code is.")
    body = {"model": "inkling2", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 500,
            "stream": False, "reasoning_effort": "low"}
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); peak = vram()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:
        print(json.dumps({"k": k, "error": str(e), "body": getattr(e, "read", lambda: b"")()[:300].decode("utf-8", "replace"), "vram": vram()})); break
    u = d["usage"]; m = d["choices"][0]["message"]; text = (m.get("content") or ""); word_seen = word in ((m.get("reasoning_content") or "") + text)
    print(json.dumps({"k": k, "prompt_tokens": u["prompt_tokens"], "cached": u["prompt_tokens_details"]["cached_tokens"],
                      "read_tps": u.get("prompt_token/s"), "gen_tps": u.get("response_token/s"), "wall_s": round(time.time() - t0, 1),
                      "vram_after": vram(), "recall": word in text, "recall_any": word_seen, "answer": text[:140]}), flush=True)
