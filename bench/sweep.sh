#!/bin/bash
# Drafter sweep and opt-in experiments with ie-run (2026-11-26). Stops inkling-ie, restarts it on exit.
E=/home/amd-ai/models/rdna4-amd-infer/clean-90c7d21/engine; M=/home/amd-ai/models/llm/inkling-2-mini-atomicchat; B=/home/amd-ai/models/rdna4-amd-infer/bench
trap "sudo -n systemctl start inkling-ie" EXIT
sudo -n systemctl stop inkling-ie; sleep 3; cd $E
export IE_INK_GLOB_KV=q4rot IE_DS_SPLIT=1 IE_DS_GLOB_WIN=8192 IE_DS_ROPE_OTF=1
one(){ # label model tokens extra-env... -- args
  local label=$1 model=$2 tok=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local out; out=$(env "${envs[@]}" timeout 80 ./build-amd24/ie-run $M/$model --backend gpu --hsaco build-amd24/ie_kernels.hsaco --tokens-file $B/$tok --n 400 --ctx 16384 "$@" 2>&1)
  local tps=$(echo "$out" | grep -oE "[0-9.]+ tokens/s \(spec[^)]*\)|steps, [0-9.]+ ms/token, [0-9.]+ tokens/s" | grep -oE "[0-9.]+ tokens/s" | tail -1)
  local tpst=$(echo "$out" | grep -oE "[0-9.]+ tokens per step" | tail -1)
  local ids=$(echo "$out" | grep "^ids:" | md5sum | cut -c1-6)
  local uniq=$(echo "$out" | grep "^ids:" | tr " " "\n" | tail -400 | sort -u | wc -l)
  local err=$(echo "$out" | grep -iE "^ie: |HIP error|not found|refused|keeping full" | head -1 | cut -c1-110)
  echo "$label | $tok | ${tps:-FAIL} | ${tpst:--} | ids $ids uniq $uniq | $err"
}
DS="--dspark $M/ie-dspark-q8.gguf"
for t in code-tokens.txt gen-tokens.txt; do
  one "plain" ie-01-23.gguf $t -- --draft 0
  for d in 5 7 8; do for p in 0.3 0.45 0.6; do one "q8head d$d p$p" ie-01-23.gguf $t -- $DS --draft $d --draft-pmin $p; done; done
  one "FAST plain" ie-01-23-fast.gguf $t -- --draft 0
  one "FAST d7 p0.45 noshort" ie-01-23-fast.gguf $t IE_DS_SHORTLIST=0 -- $DS --draft 7 --draft-pmin 0.45
  one "FAST d7 p0.45 shortlist" ie-01-23-fast.gguf $t IE_DS_SHORTLIST=1 -- $DS --draft 7 --draft-pmin 0.45
  one "FAST d8 p0.3 shortlist" ie-01-23-fast.gguf $t IE_DS_SHORTLIST=1 -- $DS --draft 8 --draft-pmin 0.3
done
for x in IE_MK_INKLING=1 IE_INKLING_GRAPH=2 IE_DS_TREE=1 IE_DS_FORESPARK=1 IE_DS_OVERLAP=1 IE_DS_GLOB_WIN=auto16k; do
  one "exp $x" ie-01-23.gguf code-tokens.txt $x -- $DS --draft 7 --draft-pmin 0.45
done
echo SWEEPDONE
