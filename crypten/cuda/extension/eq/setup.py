from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="eq",
    ext_modules=[
        CUDAExtension(
            "eq",
            ["eq.cpp", "eq_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    } 
)