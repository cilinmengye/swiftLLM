#include <torch/extension.h>

#include "block_swapping.h"

// PyTorch 的 C++ 扩展系统 (ATen 库 + Pybind11)
// 内嵌的 Pybind11: PyTorch 源码中自带了 Pybind11。当你 #include <torch/extension.h> 时，你其实已经间接引入了 Pybind11 的所有功能。
// 如下代码的作用就是创建一张“映射表”，告诉 Python：“当你调用 swiftllm_c.swap_blocks 时，请去执行 C++ 里的那个 swap_blocks 函数”。

PYBIND11_MODULE(swiftllm_c, m) {
  m.def("swap_blocks", &swap_blocks);
}
