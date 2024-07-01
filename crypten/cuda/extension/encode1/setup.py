from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="encode1",
    ext_modules=[
        CUDAExtension(
            "encode1",
            ["encode1.cpp", "encode1_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    } 
)