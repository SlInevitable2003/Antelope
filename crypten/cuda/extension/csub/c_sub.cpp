#include <torch/extension.h>
#include "c_sub.h"

void torch_launch_csub(torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n,
                       int64_t rank) {
    launch_csub((long long *)a.data_ptr(),
                (const long long *)b.data_ptr(),
                n,rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_csub",
          &torch_launch_csub,
          "csub kernel warpper");
}