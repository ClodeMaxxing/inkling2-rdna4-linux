# Findings

What a Linux user runs into, in the order we did, and what we would suggest upstream. Engine commit `90c7d21`.

## 1. Which weight files work

The engine reads Inkling files whose experts are `IQ3_XXS` (gate/up) and `IQ4_NL` (down). Unsloth's "UD" dynamic quants
mix several other types and are refused. AtomicChat's `Inkling-2-Mini-AD-IQ3_XXS.gguf` has exactly the right mix.

## 2. The silent no-loop trap

Inkling 2 runs 12 of its 24 blocks three times per token. AtomicChat's file describes that with the keys
`inkling2_loop.loop.start`, `loop.end` and `loop.count`. The engine looks for `loop.core_start`, `loop.core_end` and
`num_loops`, and if they are missing it defaults to **no loop**, without a warning.

The result looks like a success: the model loads, answers simple questions, and runs about twice as fast as it should
(229 tok/s plain decode, 271 GPU launches per token). The text is degraded and loops. Perplexity on a short test was 26.4
instead of 10.7. If your numbers beat the engine author's by 2x, this is why. A correct run shows `2222 ops` in the
`plan:` line at start-up, not `702`.

## 3. Loop tensors and the open question

With the loop keys present the engine also wants two global tensors, `loop_res_query.weight` and
`loop_kv_res_query.weight`, each holding two vectors per looped block. AtomicChat's file stores four vectors per block in
`blk.N.loop_msg_query.weight`. [`convert/ink_loopconv.py`](../convert/ink_loopconv.py) splits the four into two pairs.

**Which pairing is right is not established.** On 1,200 tokens the three possible pairings scored perplexity 10.658
(`01-23`), 10.661 (`03-12`) and 10.677 (`02-13`). That is too close to call. We use `01-23`. Someone who knows the
reference implementation, or the engine author, could settle this in a line.

## 4. The drafter

The engine wants a drafter GGUF with `general.architecture = dspark` and its own tensor names. Two llama.cpp-style
drafters are in circulation:

- **Unsloth's** (`dspark_fuse`, `dspark_hidden_out`, `joint_proj`, `markov_w1`, ...) maps one-to-one onto the engine's
  names. [`convert/ink_dsparkconv.py`](../convert/ink_dsparkconv.py) converts it.
- **AtomicChat's** (`markov_w2`, `ctx_mix`, ...) is a different variant and does not map.

One trap: the three matrices stored as BF16 must be written as `Q8_0`. Written as F32, the file loads, the prompt is read,
and the first draft step dies with a GPU memory fault. Use `--q8`.

## 5. The output head and the shortlist

The engine's fastest drafting path (a vocabulary shortlist) requires the model's `output.weight` to be `Q4_K`. The engine's
script for that depends on a Windows ggml DLL and a helper module that is not in the repository.
[`convert/ink_requant_head.py`](../convert/ink_requant_head.py) does the same job with `libggml-base.so` from any
llama.cpp build. It saves about 220 MB of card memory and costs almost nothing in accuracy (see RESULTS).

## 6. Memory on Linux

- The engine's guard against over-filling the card reads the Windows memory budget. On Linux there is no guard: an
  over-full card ends in `HIP error 2 (out of memory)` and the process exits.
- The context cache grows as a conversation grows, so that failure arrives mid-conversation. `IE_KV_GROW=0` (one fixed
  array) is refused together with the 4-bit global cache. **`IE_KV_PREMAP=<context>` with `IE_KV_SHRINK=0`** reserves
  everything at load instead; a shortage then fails the start, where a service manager can see it.
- Above 131,072 context the prompt chunk is lowered from 1024 to 512 unless the Windows budget check passes.
  `IE_CHUNK_FORCE=1` restores 1024; it read 12-14% faster for about 140 MB here.
- Recent kernels have a device-memory cgroup controller (`dmem`), which reports the card as a region and could cap what a
  desktop session takes. Not tested here.

## 7. Serving behaviour worth knowing

- One request at a time, one cached conversation (see RESULTS section 6).
- If output ends inside an unfinished tool call (for instance the client's output cap is reached while the model writes a
  large file in one call), the server reports `finish_reason: network_error`. Clients may treat that as transient and
  retry a request that cannot succeed. Give the model a generous output cap.
- The server ignores the `model` field of a request, so a client entry pointing at the same port under an old model name
  still reaches it, with that entry's limits.

## 8. Suggestions for upstream

Offered with thanks, as issues or pull requests to the engine's author, not as a fork:

1. Read `loop.start` / `loop.end` / `loop.count` as aliases of the loop keys, and warn when an `inkling2_loop` file would
   run without its loop.
2. Accept `blk.N.loop_msg_query.weight` directly, which would also settle the pairing question.
3. Accept the llama.cpp-style DSpark drafter names directly.
4. A Linux memory check (for example `hipMemGetInfo`, already loaded) behind the chunk clamp and before cache growth.
5. Makefile: `-ffreestanding` for the vision kernels, and a note that the Inkling kernels need AMD clang 24.
6. A Linux path for the output-head requant, or a documented `libggml-base.so` option.
7. A licence, whenever the upstream situation allows one.
