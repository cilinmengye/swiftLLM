from setuptools import setup
from torch.utils import cpp_extension

__version__ = "0.0.1"

# 把 C++/CUDA 源代码编译成一个可以被 Python import 的动态链接库（.so 文件）。
ext_modules = [
    cpp_extension.CUDAExtension(
        "swiftllm_c",               # ① 模块名
        [                           # ② 源文件列表
            "src/entrypoints.cpp",
			"src/block_swapping.cpp"
        ],
        extra_compile_args={        # ③ 编译参数
            'cxx': ['-O3'],                     # 给 C++ 编译器的参数
            'nvcc': ['-O3', '--use_fast_math']  # 给 CUDA 编译器的参数
        }
    ),
]

setup(
    name="swiftllm_c",
    version=__version__,
    author="Shengyu Liu",
    author_email="shengyu.liu@stu.pku.edu.cn",
    url="",
    description="Some C++/CUDA sources for SwiftLLM.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    zip_safe=False,
    python_requires=">=3.9",
)
