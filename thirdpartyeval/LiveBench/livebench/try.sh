python run_livebench.py \
    --model qwen-max \
    --bench-name live_bench \
    --question-source jsonl \
    --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key sk-a22c26442cb046e6bfdde08382772dfa \
    --skip-inference \
    --ignore-missing-answers \
    --livebench-release-option 2024-11-25 \
    --remove-existing-judgment-file

# python show_livebench_result.py --bench-name live_bench/math --model-list qwen-max --question-source jsonl --livebench-release-option 2024-11-25