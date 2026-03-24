# 测试

启动
```

```

整个测试
```
python swiftllm_benchmark.py   --host 127.0.0.1   --port 8000   --endpoint /generate   --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B   --dataset-path /mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json   --num-prompts 500   --request-rates 1,2,3,4,5,6   --burstiness 1.0   --max-concurrency 256   --seed 0   --disable-shuffle   --save-result --result-filepath /home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/test_swiftllm_result/ --max-tokens 120
```

但是还是差好多和vllm结果相比，我没招了...