此项目swiftLLM是一个轻量级大模型推理框架，现在我想要测试此推理框架在A Single Forward Operation的性能，依据swiftLLM/README.md下的实验:

The first scenario is "a single forward operation", where we feed the model with a batch of inputs and let it generate one output token (equivelant to one "forward" operation). This is the basic operation of LLM inference (both online and offline) so its performance is crucial.

我想要使用RTX 4090 GPU under FP16 precision复现出swiftLLM的性能数据实验，我希望你帮我写出benchmark代码，按照要求你需要进行如下步骤:

1. 生成测试数据: 你需要生成在batch_size = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]下对应input_len=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]的input data
2. 你需要使用生成的input data测试swiftLLM 的 Prefill Time Usage and Decoding Time Usage, 并记录下数据，利用python绘制出在不同batch_size和input_len下Prefill Time Usage and Decoding Time Usage的性能图

tips: 
1. 你可以参照 swiftLLM/examples/offline.py 代码的写法
2. 如果有更好的想法你也可以打开思路，不一定要按照我上述步骤实现，但是最终目的一定要达到

我希望你相关代码和数据实现在swiftLLM/dummys/benchmark/offline下