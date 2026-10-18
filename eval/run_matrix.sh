#!/usr/bin/env bash
# Sequential experiment matrix. One model resident at a time: 16 GB cannot hold two.
# Each run scores the same 8 gold chunks with the same metrics.
cd "D:/Extras/Deecogs/Medical RAG" || exit 1
PY="medical_rag/Scripts/python.exe"

export LOCAL_LLM_JSON_GRAMMAR=true
export LOCAL_LLM_TEMPERATURE=0.0
export LOCAL_LLM_N_THREADS=          # let it default to physical cores
export EXTRACTION_FEW_SHOT=true

run () {
  label="$1"; model="$2"
  echo "=== RUN $label ($model) ==="
  $PY -u eval/run_extraction_eval.py --label "$label" --model "$model" --n 8 2>&1 \
    | grep -E "CHUNK |RESULT |Traceback|EXCEPTION|error"
  echo "=== END $label ==="
}

# Prompt effect on the incumbent model, holding the model fixed.
run medgemma4b_v2_negation "medgemma-1.5-4b-it-Q4_K_M.gguf"

# Does general instruction tuning beat medical tuning at this task?
run qwen3b_v2 "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# Main candidate.
run qwen7b_v2 "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# Ablation: same model, worked example removed, to price the few-shot separately.
export EXTRACTION_FEW_SHOT=false
run qwen7b_v2_noshot "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

echo "MATRIX COMPLETE"
