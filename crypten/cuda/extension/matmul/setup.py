from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mat_mul",
    ext_modules=[
        CUDAExtension(
            "mat_mul",
            ["mat_mul.cpp", "matmul_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)