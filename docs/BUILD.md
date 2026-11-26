# Building rdna4-amd-infer on Ubuntu

Tested on Ubuntu 26.04, kernel 7.0, with the distribution's ROCm 7.1 runtime packages (`libamdhip64-7`) and engine commit
`90c7d21`. The engine loads the HIP runtime with `dlopen` at run time, so only the GPU kernels need a special compiler.

## The compiler is the whole problem

| Compiler | Result |
|---|---|
| Ubuntu `clang-18` | Fails: the Inkling kernels use builtins it does not have (`__builtin_amdgcn_raw_buffer_load_*`, `__amdgpu_buffer_rsrc_t`) |
| Ubuntu `clang-21` | Fails: backend error in `ie_inkling_gemm_rq_exp_iq3xxs_r3` ("illegal VGPR to SGPR copy") |
| **AMD clang 24** from the ROCm pip SDK | **Works.** 232 Inkling kernels, 0 register spills, 0 scratch memory |

A build that stops part-way still leaves `ie-serve` and the base kernel files behind, and `ie-serve` will start with them.
Check that `build/ie_inkling_kernels.hsaco` exists before believing a build.

## Steps

1. Get AMD's LLVM in user space (no system ROCm install needed):

   ```sh
   python3.12 -m venv ~/venv-rocm
   ~/venv-rocm/bin/pip install rocm-sdk-core rocm-sdk-libraries rocm-sdk-device-gfx1201
   L=~/venv-rocm/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin
   $L/clang --version      # expect: AMD clang version 24
   ```

   We used the 10.1.0 wheels in a Python 3.12 environment; use whichever Python version the wheels are published for.
   This is the same route the engine's own Windows build script takes.

2. Clone the engine and build with that compiler, leaving the source untouched:

   ```sh
   git clone https://github.com/SaloQT/rdna4-amd-infer.git && cd rdna4-amd-infer/engine
   make -k -j6 GPU_CC=$L/clang GPU_LD=$L/ld.lld OBJDUMP=$L/llvm-objdump
   ```

   `-k` lets the build finish past the one known failure below.

3. Known failure, harmless for text: `ie_vision_kernels` does not compile on Linux as shipped (`stdint.h` pulls in a libc
   header that is not there for the GPU target). It needs `-ffreestanding` on that compile step. Image decoding in the
   engine is Windows-only anyway, so this recipe runs without vision.

4. Check the result:

   ```sh
   ls build/ie-serve build/ie_kernels.hsaco build/ie_inkling_kernels.hsaco
   $L/llvm-readelf --notes build/ie_inkling_kernels.hsaco | grep -c 'vgpr_spill_count: 0'   # 232
   ```

5. Optional tools used by this repository (host compiler only):

   ```sh
   make build/tok_check      # tokenise a text file with the engine's own tokenizer: tok_check MODEL.gguf --file TEXT
   ```

## Running it

`ie-serve MODEL.gguf --backend gpu --hsaco build/ie_kernels.hsaco --port 8090` is the engine's documented Linux start.
The profile this repository recommends is in [`../serve/inkling-ie`](../serve/inkling-ie). The settings there are a
translation of the engine's Windows launcher `serve-inkling2.ps1` plus the changes described in
[`FINDINGS.md`](FINDINGS.md).
