#include <torch/extension.h>
#include "encode.h"

void torch_launch_encode(torch::Tensor &a,
                         torch::Tensor &r,
                         int64_t n) {
     launch_encode((int *)a.data_ptr(),
                (int *)r.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_encode",
          &torch_launch_encode,
          "encode kernel warpper");
}