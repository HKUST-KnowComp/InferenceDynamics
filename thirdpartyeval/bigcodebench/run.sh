DATASET=bigcodebench
MODEL=xai/grok-2-latest
BACKEND=openai
NUM_GPU=2
SPLIT=complete
SUBSET=hard
export E2B_API_KEY="e2b_0a231fa3b0a2b01690ab6c66a23b55c0979ce4ee"

python  bigcodebench/generate.py\
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND