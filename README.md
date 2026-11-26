# Inkling 2 Mini on a Radeon RX 9070 XT under Linux

A working recipe, converters and test kit for running Thinking Machines' **Inkling 2 Mini** (26B mixture-of-experts)
on one 16 GB **RX 9070 XT** under Linux with **SaloQT's `rdna4-amd-infer`** engine: 196K context, about 130-220 tokens
per second writing and 3,400-4,700 tokens per second prompt reading, inside 16 GB.

This repository contains **no engine code and no model weights**. It is the missing glue for Linux users: how to build
the engine, how to convert the publicly available weight and drafter files into the layout the engine reads, a tested
launcher, and the scripts and numbers behind every claim here.

## Credit where it is due

- **The engine** is [`SaloQT/rdna4-amd-infer`](https://github.com/SaloQT/rdna4-amd-infer) by SaloQT: 217 hand-written
  Inkling kernels for RDNA4, DSpark drafting, the 4-bit global context cache and the long-context work. All of the speed
  is theirs.
- It is built on [`hanw/amd.c`](https://github.com/hanw/amd.c) by Han Wang, a from-scratch C11 inference engine for the
  same GPU with a formally verified indexing and quantisation core.
- **Weights**: [`AtomicChat/Inkling-2-Mini-GGUF`](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) (Apache-2.0).
  **Drafter**: the DSpark file from `unsloth/Inkling-2-Mini-GGUF`.
- The starting point was a benchmark post on r/llamacpp comparing this engine with llama.cpp on the same card.

Neither engine repository carries a licence file at the time of writing, so their code is not redistributed here. Get it
from the authors, and please star their repositories. See [CREDITS.md](CREDITS.md).

## What you get

| Path | What it is |
|---|---|
| [`docs/BUILD.md`](docs/BUILD.md) | Building the engine on Ubuntu: which compiler works, which do not |
| [`convert/ink_loopconv.py`](convert/ink_loopconv.py) | AtomicChat weights -> the loop layout the engine reads |
| [`convert/ink_dsparkconv.py`](convert/ink_dsparkconv.py) | Unsloth DSpark drafter -> the engine's drafter format |
| [`convert/ink_requant_head.py`](convert/ink_requant_head.py) | Output head Q8_0 -> Q4_K (Linux stand-in for the engine's Windows-only step) |
| [`serve/inkling-ie`](serve/inkling-ie), [`serve/inkling-ie.service`](serve/inkling-ie.service) | The launcher and systemd unit in daily use |
| [`serve/run-ie.sh`](serve/run-ie.sh) | Trial profiles used while testing |
| [`bench/`](bench) | Speed benchmark, context fill-and-recall probe, multi-agent test, drafter sweep |
| [`results/`](results) | Raw output of the runs quoted below |
| [`docs/RESULTS.md`](docs/RESULTS.md) | All measurements, with how each was taken |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | The traps we hit, and fixes worth offering upstream |

## Results in one table

Same machine, same prompts, greedy decoding. "Before" is Unsloth's llama.cpp fork (HIP build) with the Unsloth
UD-IQ3_XXS weights, DSpark drafter and an 8-bit context cache at 65,536 context.

| | llama.cpp fork (before) | rdna4-amd-infer (this recipe) |
|---|---|---|
| Context | 65,536 | 196,608 |
| Writing, short prompts | 38-65 tok/s | 139-224 tok/s |
| Writing at a full context | 38 tok/s at 59K | 126-129 tok/s at 190K |
| Prompt reading | 236-352 tok/s | 3,430-4,664 tok/s |
| A 59K-token prompt, end to end | 206 s | about 16 s |
| Card memory | 15.2-15.9 GB | 15.7 GB, reserved at start, flat |

Accuracy of the compressed pieces against the same engine at full precision, over 30,000 tokens: mean KL divergence
0.0109, same top token 96.55% of the time, perplexity 3.0589 against 3.0494. Details and caveats in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

1. Build the engine with AMD's clang 24: [`docs/BUILD.md`](docs/BUILD.md).
2. Download `Inkling-2-Mini-AD-IQ3_XXS.gguf` (12.6 GB) from AtomicChat and `dspark-Inkling-2-Mini-Q8_0.gguf` from Unsloth.
3. Convert (CPU only, a few minutes, needs Python with numpy for the last two):

   ```sh
   python3 convert/ink_loopconv.py Inkling-2-Mini-AD-IQ3_XXS.gguf ie-01-23.gguf 01-23
   python3 convert/ink_requant_head.py ie-01-23.gguf ie-01-23-fast.gguf /path/to/libggml-base.so
   python3 convert/ink_dsparkconv.py dspark-Inkling-2-Mini-Q8_0.gguf ie-dspark-q8.gguf --q8
   ```

4. Edit the paths at the top of `serve/inkling-ie`, install it with the unit file, and start it.
5. Check it with `bench/ctx_probe.py` before trusting it: it fills the context in steps and asks the model to repeat a
   code word from the top of the prompt.

## Know before you rely on it

- **One request at a time.** The engine has no parallel slots and keeps one conversation in its cache. Several agents
  sharing it take turns and each re-reads its whole context ([`docs/RESULTS.md`](docs/RESULTS.md)).
- **No vision on Linux.** The engine's image decoding is Windows-only, and the vision kernels were not built here.
- **The loop row order is unproven.** `ink_loopconv.py` has to guess how four per-block vectors map onto the engine's two
  tensors. Variant `01-23` scored best, but only by a hair. See [`docs/FINDINGS.md`](docs/FINDINGS.md).
- **Drafted output is not token-identical** to undrafted output on this setup, although every answer we checked was sound.
- **Memory is tight.** With a desktop session on the same card about 0.6 GB is spare. Run headless if you can.
- **No quality comparison with llama.cpp yet.** The speed comparison is solid; the two setups use different weight files
  and were never scored on the same text.
- Tested on one machine: RX 9070 XT 16 GB, Ubuntu 26.04, kernel 7.0, ROCm 7.1 runtime, 30 GB RAM, engine commit `90c7d21`.

## Licence

The scripts and documents in this repository are released under the [Apache License 2.0](LICENSE).

That licence covers this repository's own files only. It does not include, and does not cover, any code from
`rdna4-amd-infer` or `amd.c`, which remain their authors' and carry no licence at the time of writing, nor any model
weights, which keep the licences of their publishers.
