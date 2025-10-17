#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/diag_cpu.h"

#ifdef WITH_MUSA
#include "musa/diag_musa.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__diag_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__diag_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor non_diag_mask(torch::Tensor row, torch::Tensor col, int64_t M,
                            int64_t N, int64_t k) {
  if (at::musa::is_musa(row)) {
#ifdef WITH_MUSA
    return non_diag_mask_musa(row, col, M, N, k);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return non_diag_mask_cpu(row, col, M, N, k);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::non_diag_mask", &non_diag_mask);
