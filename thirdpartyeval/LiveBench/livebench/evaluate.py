import json
import subprocess
import pandas as pd
import os
if __name__ == "__main__":
    # If model list is not provided, evaluate all models.
    model_list = []
    if len(model_list) == 0:
        model_list = [filename.split('.jsonl')[0] for filename in os.listdir('/home/hshiah/LLM_index/thirdpartyeval/LiveBench/livebench/data/live_bench/coding/model_answer')]
        model_list = ['gemini-2.0-flash-001', 'deepseek-chat-v3-0324', 'claude-3.5-sonnet',
        'grok-2-latest','nova-pro-v1','qwen-plus',
        'glm-4-plus','gpt-4o-2024-11-20','gemini-pro-1.5','hunyuan-turbos-latest',]
        model_list = ['qwen-2.5-72b-instruct']
    for model in model_list:
        subprocess.run([
            'python', 'run_livebench.py',
            '--model', model,
            '--bench-name', 'live_bench',
            '--question-source', 'jsonl',
            '--api-base', 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '--api-key', 'sk-a22c26442cb046e6bfdde08382772dfa',
            '--skip-inference',
            '--ignore-missing-answers',
            '--livebench-release-option', '2024-11-25',
            '--remove-existing-judgment-file'
        ])