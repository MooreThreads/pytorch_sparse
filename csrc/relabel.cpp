#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/relabel_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__relabel_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__relabel_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
                                                 torch::Tensor idx) {
  if (at::musa::is_musa(col)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return relabel_cpu(col, idx);
  }
}

SPARSE_API std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop(torch::Tensor rowptr, torch::Tensor col,
                std::optional<torch::Tensor> optional_value,
                torch::Tensor idx, bool bipartite) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return relabel_one_hop_cpu(rowptr, col, optional_value, idx, bipartite);
  }
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::relabel", &relabel)
        .op("torch_sparse::relabel_one_hop", &relabel_one_hop);
