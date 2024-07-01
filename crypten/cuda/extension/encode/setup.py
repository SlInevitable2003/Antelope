from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="encode",
    ext_modules=[
        CUDAExtension(
            "encode",
            ["encode.cpp", "encode_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    } 
)