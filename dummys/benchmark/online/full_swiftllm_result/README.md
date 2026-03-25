相比 swiftllm_result/ 此处没有限制max-tokens，而是根据sharegpt里面的对话决定

启动:
```
 python api_server.py --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B --host 0.0.0.0 --port 8000
```

测试：
```
python swiftllm_benchmark.py   --host 127.0.0.1   --port 8000   --endpoint /generate   --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B   --dataset-path /mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json   --num-prompts 500   --request-rates 1,2,3,4,5,6   --burstiness 1.0   --max-concurrency 256   --seed 0   --disable-shuffle   --save-result --result-filepath /home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/full_swiftllm_result/
```