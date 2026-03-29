"""
旨在创建出一种通用的(对于任何模型都适用的)权重加载方案, 我们会提供模型架构, 提供safetensors文件

新的思路是：
1. model / submodule 中不再保存临时权重 tensor
2. 而是只保存 "weight_name -> target_loader_module" 的映射
3. loader 逐个读取 safetensors 中的权重, 读取后立刻分发给对应模块处理
4. 具体如何消费权重(直接copy / 合并QKV / 合并gate+up)由模块自己决定
"""

import os
import glob
from torch import nn
from safetensors import safe_open


def load_weight(model: nn.Module, path: str):
    """
    要求：
    1. model 的某些模块实现了 name_to_module: dict[str, nn.Module]
       表示某个 safetensors 权重名应该交给哪个模块处理
    2. 真正消费权重的模块需要实现:
           load_weight(weight: torch.Tensor, weight_name: str)
       其中模块可根据 weight_name 决定如何处理当前权重

    优点：
    - 不再把所有权重暂存在 CPU dict 上
    - 从文件读取一个 tensor 后, 立刻交给目标模块处理
    - 特殊模块(如QKV/GateUP)可在模块内部仅缓存少量必要张量
    """
    if not os.path.isdir(path):
        raise ValueError(f"Invalid weight path: {path}")

    # 第一步：一次性建立全局索引，避免每个权重都遍历整棵模型树
    name_to_loader_module: dict[str, nn.Module] = {}

    for _, module in model.named_modules():
        if hasattr(module, "name_to_module"):
            mapping = module.name_to_module
            if not isinstance(mapping, dict):
                raise TypeError(
                    f"{module.__class__.__name__}.name_to_module must be a dict, "
                    f"but got {type(mapping)}"
                )

            for weight_name, target_module in mapping.items():
                if weight_name in name_to_loader_module:
                    raise RuntimeError(
                        f"Duplicated weight mapping detected for key: {weight_name}"
                    )
                if not hasattr(target_module, "load_weight"):
                    raise TypeError(
                        f"Target module for {weight_name} does not implement load_weight"
                    )
                name_to_loader_module[weight_name] = target_module

    if len(name_to_loader_module) == 0:
        raise RuntimeError("No name_to_module mapping found in model.")

    # 用于最后校验是否所有需要的权重都被加载到了
    remaining_keys = set(name_to_loader_module.keys())

    # 第二步：逐文件逐权重读取，并立即分发
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if len(files) == 0:
        raise RuntimeError(f"No .safetensors files found under: {path}")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                target_module = name_to_loader_module.get(weight_name, None)
                if target_module is None:
                    # 不属于当前模型/当前TP rank需要的权重，直接跳过
                    continue

                weight = f.get_tensor(weight_name)
                target_module.load_weight(weight, weight_name)
                remaining_keys.discard(weight_name)

                # 显式删除局部引用，帮助 Python 尽快释放
                del weight

    # 第三步：校验是否有缺失权重
    if len(remaining_keys) > 0:
        missing_keys = sorted(list(remaining_keys))
        error_msg = f"检测到有 {len(missing_keys)} 个权重未被加载！\n"
        error_msg += f"缺失的 Key 如下: {missing_keys[:32]}"
        if len(missing_keys) > 32:
            error_msg += " ..."
        raise RuntimeError(error_msg)

    # 第四步：如果某些模块需要在全部权重加载完成后做最终校验/清理，可调用 finalize_weight_loading
    # for _, module in model.named_modules():
    #     if hasattr(module, "finalize_weight_loading"):
    #         module.finalize_weight_loading()