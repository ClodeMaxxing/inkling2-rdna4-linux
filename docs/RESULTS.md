# Results

All measurements are from one machine on 2026-11-26: RX 9070 XT 16 GB, Ubuntu 26.04, kernel 7.0, ROCm 7.1 runtime,
30 GB RAM, a GNOME desktop on the same card. Engine `rdna4-amd-infer` at commit `90c7d21`, built as in
[`BUILD.md`](BUILD.md). Speeds are each server's own timings. Most figures are single runs; treat differences under about
10% as noise. Raw output is in [`../results`](../results).

## 1. Against the previous setup

Benchmark: [`../bench/ink_bench.py`](../bench/ink_bench.py). Three short prompts (code, edit, prose) of up to 512 output
tokens, a 13.6K-token prompt and a 59K-token prompt, greedy.

Baseline: Unsloth's llama.cpp fork, HIP build, `Inkling-2-Mini-UD-IQ3_XXS.gguf`, DSpark drafter (3 drafts), 8-bit
context cache, 65,536 context.

| Test | llama.cpp fork | Engine, no drafter, 65K f16 | Engine, drafter, 128K 4-bit global |
|---|---|---|---|
| Code, writing | 51 tok/s | 116 | 155-164 |
| Edit, writing | 65 tok/s | 117 | 249 |
| Prose, writing | 38 tok/s | 117 | 138 |
| 13.6K prompt, reading | 352 tok/s | 5,406 | 4,569 |
| 13.6K prompt, writing | 52 tok/s | 101 | 148-160 |
| 59K prompt, reading | 236 tok/s | 4,698 | 3,959 |
| 59K prompt, writing | 38 tok/s | 82 | 126 |

The two setups use different weight files, so this compares speed, not answer quality.

## 2. The profile in daily use

196,608 context, 4-bit rotated cache on the 8 global layers, all experts on the card, DSpark drafter (7 drafts,
p-min 0.45) with the vocabulary shortlist, output head at Q4_K, prompt chunk 1024, cache reserved at start.

| Measure | Value |
|---|---|
| Writing: code / edit / prose | 195 / 224 / 139 tok/s |
| Writing at a 190,703-token context | 126-129 tok/s |
| Reading, 60K prompt | 4,664 tok/s |
| Reading, 190K prompt | 3,430 tok/s (59 s end to end) |
| Follow-up turn on a 190K conversation | 1.4-2.1 s |
| Card memory | 15.66 GB from load, unchanged through a 190K prompt |
| Recall of a code word placed at the top of the prompt | correct at 60K, 94K, 127K, 166K and 190K |

## 3. How far the context goes

Probe: [`../bench/ctx_probe.py`](../bench/ctx_probe.py).

| Context cache | Largest prompt that worked | What happened beyond it |
|---|---|---|
| f16 on all layers, experts on the card | 77,197 tokens (15.7 GB) | out of memory at about 88K; the server process exits |
| 4-bit rotated on global layers, drafter on | 190,703 tokens at a 196,608 context | not pushed further |

With the 4-bit global cache, context costs about 9 MiB of card memory per 1,000 tokens.

## 4. What the compression costs

`ie-run --ppl --kld` over the same 30,000 tokens of mixed prose and C source, against the engine at f16 with the
original 8-bit output head. This measures the reading path only; it says nothing about the 3-bit weights themselves.

| Configuration | Perplexity | Mean KL | Median KL | 99th pct | Same top token |
|---|---|---|---|---|---|
| f16 cache, Q8_0 head (reference) | 3.0494 | - | - | - | - |
| 4-bit global cache, Q8_0 head | 3.0575 | 0.0100 | 0.0020 | 0.101 | 96.63% |
| 4-bit global cache, Q4_K head (in use) | 3.0589 | 0.0109 | 0.0023 | 0.105 | 96.55% |

On an earlier 1,200-token text: 8-bit cache on all layers 0.0055 mean KL, 96.4% same top token; 4-bit global 0.0096, 95.2%.

Not measured: the drafter's effect. In one 200-token greedy test drafted and undrafted output diverged after 22 tokens,
both continuing sensibly.

## 5. Drafter settings

Sweep: [`../bench/sweep.sh`](../bench/sweep.sh), 400 tokens from a code prompt and a prose prompt
([`../results/sweep.log`](../results/sweep.log)).

- Plain decode 116 tok/s (121 with the Q4_K head).
- Drafts 5, 7 or 8 with p-min 0.3, 0.45 or 0.6: 168-200 tok/s on code, 198-225 on prose, no setting best on both.
- Q4_K head with the shortlist at 7 / 0.45: 199 on code, 227 on prose. This is what the profile uses.
- The engine's opt-in switches (`IE_MK_INKLING`, `IE_INKLING_GRAPH=2`, `IE_DS_TREE`, `IE_DS_FORESPARK`,
  `IE_DS_OVERLAP`, `IE_DS_GLOB_WIN=auto16k`) changed nothing measurable; the megakernel was slightly slower.

## 6. Several agents on one server

Test: [`../bench/agents_test.py`](../bench/agents_test.py). The engine serves one request at a time and caches one
conversation, so this is turn-taking, not parallel decoding. Run before the last tuning pass.

| | 1 agent x 190K | 2 agents x 95K | 4 agents x 47K |
|---|---|---|---|
| Turns / errors | 3 / 0 | 6 / 0 | 12 / 0 |
| Code word recalled | 3 of 3 | 6 of 6 | 12 of 12 |
| First turn | 65 s | 28 s and 56 s | 13 s to 55 s |
| Follow-up turns | 1.4-2.1 s | 26 s and 52-54 s | 12 s to 51 s |
| Context reused on follow-ups | all | none | none |
| Card memory | flat | flat | flat |

The engine documents a disk cache (`IE_DISK_CACHE`) meant for exactly this; it was not tested here.
