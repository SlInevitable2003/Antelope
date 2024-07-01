#include <torch/extension.h>
#include "encode1.h"

void torch_launch_encode1(torch::Tensor &a,
                         torch::Tensor &r,
                         int64_t n) {
     launch_encode1((int *)a.data_ptr(),
                (int *)r.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_encode1",
          &torch_launch_encode1,
          "encode1 kernel warpper");
}