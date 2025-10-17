#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/ego_sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__ego_sample_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__ego_sample_cpu(void) { return NULL; }
#endif
#endif
#endif

// Returns `rowptr`, `col`, `n_id`, `e_id`, `ptr`, `root_n_id`
SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
ego_k_hop_sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                     int64_t depth, int64_t num_neighbors, bool replace) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return ego_k_hop_sample_adj_cpu(rowptr, col, idx, depth, num_neighbors,
                                    replace);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::ego_k_hop_sample_adj", &ego_k_hop_sample_adj);
