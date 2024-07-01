from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="c_sub",
    ext_modules=[
        CUDAExtension(
            "c_sub",
            ["c_sub.cpp", "csub_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)