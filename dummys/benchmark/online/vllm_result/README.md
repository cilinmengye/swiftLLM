测试命令:

启动:
```
vllm serve /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

单个测试:
```
vllm bench serve \
  --backend openai \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/completions \
  --model /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B \
  --dataset-name sharegpt \
  --dataset-path /mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 4 \
  --burstiness 1.0 \
  --max-concurrency 256 \
  --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-filename /home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/vllm_result/sharegpt_rate_4.json
```

整个测试:
```
python vllm_benchmark.py
```