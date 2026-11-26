# Credits

This repository documents a way to run other people's work. The achievement is theirs.

| Who | What | Where |
|---|---|---|
| **SaloQT** | `rdna4-amd-infer`: the Windows port, 310 custom `gfx1201` kernels (217 for Inkling 2), DSpark drafting, vocabulary shortlist, 4-bit rotated global context cache, expert cache, 256K long-context work, vision kernels, OpenAI-compatible server | https://github.com/SaloQT/rdna4-amd-infer |
| **Han Wang** | `amd.c`: the base C11 engine, formally verified indexing and quantisation core, base kernels and the `ie-serve` architecture that `rdna4-amd-infer` builds on | https://github.com/hanw/amd.c |
| **Thinking Machines** | Inkling 2 Mini, the model | https://huggingface.co/thinkingmachines |
| **AtomicChat** | The `AD-IQ3_XXS` GGUF conversion used here (Apache-2.0) | https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF |
| **Unsloth** | The DSpark drafter GGUF used here, and the llama.cpp fork used as the baseline | https://huggingface.co/unsloth/Inkling-2-Mini-GGUF |
| **ggml / llama.cpp authors** | `ggml_quantize_chunk`, used by `ink_requant_head.py` through `libggml-base.so` | https://github.com/ggml-org/llama.cpp |
| **u/BhatSahab** | The r/llamacpp benchmark post that pointed at the engine | r/llamacpp |

## What is and is not in this repository

- **In it:** our own converter scripts, launcher, test scripts, measurements and notes.
- **Not in it:** any source file, kernel or binary from `rdna4-amd-infer` or `amd.c`, and no model weights.

Both engine repositories are published without a licence file. Their authors keep all rights to their code, and SaloQT's
README says the repository deliberately applies no licence over Han Wang's upstream code. Fetch the engine from its
authors. If either author asks for something here to be changed or removed, it will be.

## How the work here was done

The measurements, converters and notes were produced by the repository owner working with Claude (Anthropic's AI
assistant) on the owner's own hardware. Errors in them are ours, not the engine authors'.
