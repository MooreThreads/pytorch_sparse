#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/saint_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__saint_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__saint_cpu(void) { return NULL; }
#endif
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
         torch::Tensor col) {
  if (at::musa::is_musa(idx)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return subgraph_cpu(idx, rowptr, row, col);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::saint_subgraph", &subgraph);
