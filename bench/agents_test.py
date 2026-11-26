#!/usr/bin/env python3
"""N agents sharing one ie-serve (2026-11-26): each holds its own conversation of about K thousand tokens and
takes three turns; in every round all agents send at once (the server queues them).
  agents_test.py URL MODEL OUT.jsonl N:K [N:K ...] -- SRC [SRC ...]
Per turn: wait (send to answer, queue time included), prompt and cached tokens, the server's own read and write
speeds, and whether the agent's code word (given once, at the top of its first message) came back."""
import glob, json, sys, threading, time, urllib.request

a = sys.argv[1:]
url, model, out = a[0], a[1], a[2]
cut = a.index("--"); cols = [tuple(int(x) for x in c.split(":")) for c in a[3:cut]]; srcs = a[cut + 1:]
big = "".join(open(f, encoding="utf-8", errors="replace").read() for f in srcs)
vf = glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")[0]
def vram():
    return int(open(vf).read()) // 1048576
FOLLOW = ["Repeat the secret code word exactly, then name one function that is defined in the code above.",
          "What was the secret code word? Answer with only the code word."]
log = open(out, "a")
lock = threading.Lock()

def turn(n, k, i, t, msgs, word):
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 400, "stream": False, "reasoning_effort": "low"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); row = {"agents": n, "k": k, "agent": i, "turn": t}
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        u = d["usage"]; m = d["choices"][0]["message"]; text = m.get("content") or ""
        row.update(wait_s=round(time.time() - t0, 1), prompt_tokens=u["prompt_tokens"], cached=u["prompt_tokens_details"]["cached_tokens"],
                   read_tps=u.get("prompt_token/s"), gen_tps=u.get("response_token/s"), out_tokens=u.get("completion_tokens"),
                   prompt_ms=u.get("prompt_ms"), recall=word in text, finish=d["choices"][0].get("finish_reason"), answer=text[:100])
        msgs.append({"role": "assistant", "content": text})
    except Exception as e:
        row.update(wait_s=round(time.time() - t0, 1), error=str(e)[:200], recall=False)
    row["vram"] = vram()
    with lock:
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row)[:300], flush=True)

for n, k in cols:
    convs = []
    for i in range(n):
        word = "HERON-%d%d%d" % (n, i, k)
        o = i * 20000
        convs.append((word, [{"role": "user", "content": "The secret code word is %s. Remember it.\n\nBelow is C source code.\n\n```c\n%s\n```\n\nFirst state the secret code word given at the very top. Then say in one sentence what this code is." % (word, big[o: o + int(k * 1000 * 2.44)])}]))
    t_col = time.time()
    for t in range(3):
        if t: [c[1].append({"role": "user", "content": FOLLOW[t - 1]}) for c in convs]
        th = [threading.Thread(target=turn, args=(n, k, i, t + 1, convs[i][1], convs[i][0])) for i in range(n)]
        for i, x in enumerate(th):
            x.start(); time.sleep(0.3)  # fixed order: agent 0 first
        [x.join() for x in th]
    with lock:
        log.write(json.dumps({"agents": n, "k": k, "column_total_s": round(time.time() - t_col, 1), "vram": vram()}) + "\n"); log.flush()
print("ALLDONE"); log.write("ALLDONE\n"); log.close()
