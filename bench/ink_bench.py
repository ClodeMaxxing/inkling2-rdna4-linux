#!/usr/bin/env python3
"""Same benchmark for llama-server and ie-serve (shape from the r/llamacpp post): three short prompts
(code, edit, prose) up to 512 tokens twice, a ~14K-token prompt twice, a ~56K-token prompt once. Greedy.
Speeds are each server's own timings; wall clock is also recorded.
  ink_bench.py URL LABEL OUT.json [--key-env NAME] [--model ID] [--parts short,14k,56k] [--src FILE ...]
"""
import json, os, sys, time, urllib.request

a = sys.argv[1:]
url, label, out = a[0], a[1], a[2]
opt = a[3:]
def val(name, d=None):
    return opt[opt.index(name) + 1] if name in opt else d
key = os.environ.get(val("--key-env", "INK_KEY"), "")
model = val("--model", "inkling")
parts = val("--parts", "short,14k,56k").split(",")
srcs = opt[opt.index("--src") + 1:] if "--src" in opt else []
CPT = float(val("--cpt", "3.2"))  # characters per token for C source, calibrated from the first long run

SHORT = [
    ("code", "Write a Python function that parses an ISO 8601 date-time string (with optional timezone offset) "
             "into a UTC datetime without using third-party libraries. Include a few doctests."),
    ("edit", "Rewrite this function so it is iterative instead of recursive and handles negative n by raising "
             "ValueError. Return only the code.\n\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n"),
    ("prose", "Explain in plain language, in about 350 words, why a mixture-of-experts language model can be faster "
              "to run than a dense model with the same number of parameters."),
]

def call(name, prompt, max_tokens=512):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1,
            "max_tokens": max_tokens, "stream": False, "reasoning_effort": "low"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            d = json.load(r)
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        row = {"name": name, "error": str(e), "body": body[:400].decode("utf-8", "replace"), "wall_s": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        return row
    wall = time.time() - t0
    u, t = d.get("usage", {}), d.get("timings", {})
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    row = {"name": name, "wall_s": round(wall, 1), "prompt_tokens": u.get("prompt_tokens"),
           "completion_tokens": u.get("completion_tokens"), "finish": d["choices"][0].get("finish_reason"),
           "chars_out": len(text), "tail": text[-160:]}
    if t:  # llama-server
        row.update(read_tps=round(t.get("prompt_per_second", 0), 1), read_n=t.get("prompt_n"),
                   cached=t.get("cache_n"), gen_tps=round(t.get("predicted_per_second", 0), 1),
                   draft_n=t.get("draft_n"), draft_acc=t.get("draft_n_accepted"))
    else:  # ie-serve
        row.update(read_tps=u.get("prompt_token/s"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   gen_tps=u.get("response_token/s"), prompt_ms=u.get("prompt_ms"), response_ms=u.get("response_ms"),
                   tok_per_step=u.get("mtp_tokens_per_step"))
    print(json.dumps(row), flush=True)
    return row

rows = []
if "short" in parts:
    for rnd in (1, 2):
        for n, p in SHORT:
            rows.append(call("%s#%d" % (n, rnd), p))
big = "".join(open(f, encoding="utf-8", errors="replace").read() for f in srcs)
def long_prompt(tokens):
    return ("Below is C source code from an inference engine.\n\n```c\n" + big[: int(tokens * CPT)] +
            "\n```\n\nSummarise in about 300 words what this code does and name its main functions.")
if "14k" in parts:
    for rnd in (1, 2):
        rows.append(call("14k#%d" % rnd, long_prompt(13500)))
if "56k" in parts:
    rows.append(call("56k#1", long_prompt(54500)))
json.dump({"label": label, "url": url, "rows": rows}, open(out, "w"), indent=1)
print("saved", out)
