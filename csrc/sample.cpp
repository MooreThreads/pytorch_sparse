#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__sample_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__sample_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
           int64_t num_neighbors, bool replace) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return sample_adj_cpu(rowptr, col, idx, num_neighbors, replace);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::sample_adj", &sample_adj);
