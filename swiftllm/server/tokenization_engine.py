import ray
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig

# 在 Ray 框架中，@ray.remote 是一个核心装饰器，它的作用是将一个普通的 Python 类或函数转换成一个分布式的对象或任务。
# 当你看到 @ray.remote class TokenizationEngine 时, 你正定义一个 Ray Actor。加上装饰器后，这个类就从一个“只能在
# 本地运行的工具”变成了“可以在集群中任何地方运行的服务”。它的主要职责包括：1. 状态保持：与普通的远程函数不同，Actor
# 会在内存中保留自己的状态（比如加载好的分词模型字典）。2. 并发处理：你可以同时启动多个 TokenizationEngine 实例，
# 分别处理不同的数据流。3. 在复杂的 AI 流水线中，你可能希望“分词”和“模型推理”同时进行。使用 Ray Actor，主程序可
# 以把文本丢给分词引擎后立刻去处理别的事情，等分词好了再通过 ray.get() 拿回结果。
#
# 启动 Actor: engine = TokenizationEngine.remote()
# 异步调用: obj_ref = engine.tokenize.remote("Hello World")
# 获取结果: print(ray.get(obj_ref))


@ray.remote
class TokenizationEngine:
    def __init__(self, engine_config: EngineConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(engine_config.model_path)

    def batched_tokenize(self, prompts: list[str]) -> list[list[int]]:
        prompt_token_ids = self.tokenizer(prompts, return_attention_mask=False)['input_ids']
        return prompt_token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id
