#!/bin/sh
# Trial profiles used while testing (see docs/RESULTS.md). Edit E and M below. The daily profile is serve/inkling-ie.
#   run-ie.sh plain|std|256k     (127.0.0.1:8090, no API key, loopback only)
E=/home/amd-ai/models/rdna4-amd-infer/clean-90c7d21/engine
M=/home/amd-ai/models/llm/inkling-2-mini-atomicchat
MODEL=$M/ie-01-23.gguf
DS=$M/dspark-Inkling-2-Mini-Q8_0.gguf
COMMON="--backend gpu --hsaco $E/build-amd24/ie_kernels.hsaco --host 127.0.0.1 --port 8090 --name inkling2"
PS1ARGS="--think-temp 1.0 --top-k 20 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 24 --tool-temp 0 --effort high"
cd $E || exit 2
case "$1" in
  plain) exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 8192 --draft 0 ;;
  std)   export IE_DS_SPLIT=1 IE_DS_SHORTLIST=1
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 98304 --chunk 1024 $PS1ARGS --dspark $DS --draft 5 --draft-pmin 0.6 ;;
  256k)  export IE_DS_SPLIT=1 IE_DS_SHORTLIST=1 IE_INK_GLOB_KV=q4rot IE_DS_GLOB_WIN=8192 IE_DS_ROPE_OTF=1 IE_MOE_CACHE_SLOTS=auto
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 262144 --chunk 1024 $PS1ARGS --dspark $DS --draft 7 --draft-pmin 0.45 ;;
  std-nodraft) exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 98304 --chunk 1024 $PS1ARGS --draft 0 ;;
  256k-nodraft) export IE_INK_GLOB_KV=q4rot IE_MOE_CACHE_SLOTS=auto
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 262144 --chunk 1024 $PS1ARGS --draft 0 ;;
  like-current) exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 65536 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --draft 0 ;;
  like-current-q8) export IE_KV_Q8=1
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 65536 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --draft 0 ;;
  f16-probe) exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 131072 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --draft 0 ;;
  128k-q4rot) export IE_INK_GLOB_KV=q4rot
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 131072 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --draft 0 ;;
  128k-ds) export IE_INK_GLOB_KV=q4rot IE_DS_SPLIT=1 IE_DS_SHORTLIST=1 IE_DS_GLOB_WIN=8192 IE_DS_ROPE_OTF=1
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 131072 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --dspark $M/ie-dspark-q8.gguf --draft 7 --draft-pmin 0.45 ;;
  196k-ds) export IE_INK_GLOB_KV=q4rot IE_DS_SPLIT=1 IE_DS_GLOB_WIN=8192 IE_DS_ROPE_OTF=1 IE_KV_PREMAP=196608 IE_KV_SHRINK=0
         exec ./build-amd24/ie-serve $MODEL $COMMON --ctx 196608 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --dspark $M/ie-dspark-q8.gguf --draft 7 --draft-pmin 0.45 ;;
  196k-fast|196k-fast-force) [ "$1" = 196k-fast-force ] && export IE_CHUNK_FORCE=1
         export IE_INK_GLOB_KV=q4rot IE_DS_SPLIT=1 IE_DS_SHORTLIST=1 IE_DS_GLOB_WIN=8192 IE_DS_ROPE_OTF=1 IE_KV_PREMAP=196608 IE_KV_SHRINK=0
         exec ./build-amd24/ie-serve $M/ie-01-23-fast.gguf $COMMON --ctx 196608 --chunk 1024 --think-temp 1.0 --top-k 20 --effort low --dspark $M/ie-dspark-q8.gguf --draft 7 --draft-pmin 0.45 ;;
  *) echo "usage: run-ie.sh plain|std|256k"; exit 2 ;;
esac
