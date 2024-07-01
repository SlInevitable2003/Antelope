#include <torch/extension.h>
#include "mat_mul.h"

void torch_launch_matmul(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t m,
                       int64_t n,
                       int64_t k) {
    launch_matmul((long long *)c.data_ptr(),
                (const long long *)a.data_ptr(),
                (const long long *)b.data_ptr(),
                m,n,k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_matmul",
          &torch_launch_matmul,
          "matmul kernel warpper");
}