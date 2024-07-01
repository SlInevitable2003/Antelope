#include <torch/extension.h>
#include "eq.h"

void torch_launch_eq(torch::Tensor &a,
                         int64_t n) {
     launch_eq((int *)a.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_eq",
          &torch_launch_eq,
          "eq kernel warpper");
}