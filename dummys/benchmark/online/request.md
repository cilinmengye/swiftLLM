此项目swiftLLM是一个轻量级大模型推理框架，现在我想要测试此推理框架在Online Serving的性能，依据swiftLLM/README.md下的实验:

The second scenario is "online serving", where we start an API server, sample prompts from a real-world dataset, and let the model generate completions. This is the scenario where LLM is used in real-world applications like chatbots or code completions.

我想要使用RTX 4090 GPU under FP16 precision复现出swiftLLM的性能数据实验，我希望你帮我写出benchmark代码，你需要复现swiftLLM/README.md中Online Serving的做法:
Here we use the [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) to sample prompts, and use a poisson process with different lambdas to simulate different request arrival rates. 

我希望你在swiftLLM/dummys/benchmark/online/plan.md文件中创建详细的实施计划。内容包括：
* 分步实施方案:
    * 我应该如何下载并使用ShareGPT dataset
    * 我如何实现use the ShareGPT dataset to sample prompts, and use a poisson process with different lambdas to simulate different request arrival rates.
    * 我应该如何复现swiftLLM/docs/assets/online-llama-3-7b-4090.png中图片在RTX 4090 GPU under FP16 precision下的性能数据实验？你能读图片吗？如果你不能读，我来告诉你:
    swiftLLM/docs/assets/online-llama-3-7b-4090.png中有三张图: 1. Per-Token Latency, Per Input Token Latency, Per Output Token Latency, 图的横坐标为Request Rate, 纵坐标为Latency(ms)
* 展示实际更改的代码片段
* 需要修改的文件路径
* 不同方案之间的优缺点

暂勿实施。

tips: 
1. 你可以参照 swiftLLM/examples/online.py 代码的写法
2. 如果有更好的想法你也可以打开思路，不一定要按照我上述步骤实现，但是最终目的一定要达到

我希望你相关代码和数据实现在swiftLLM/dummys/benchmark/online下