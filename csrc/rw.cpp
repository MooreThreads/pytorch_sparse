#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/rw_cpu.h"

#ifdef WITH_MUSA
#include "musa/rw_musa.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__rw_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__rw_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor random_walk(torch::Tensor rowptr, torch::Tensor col,
                          torch::Tensor start, int64_t walk_length) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    return random_walk_musa(rowptr, col, start, walk_length);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return random_walk_cpu(rowptr, col, start, walk_length);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::random_walk", &random_walk);
