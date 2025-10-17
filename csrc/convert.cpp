#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/convert_cpu.h"

#ifdef WITH_MUSA
#include "musa/convert_musa.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__convert_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__convert_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor ind2ptr(torch::Tensor ind, int64_t M) {
  if (at::musa::is_musa(ind)) {
#ifdef WITH_MUSA
    return ind2ptr_musa(ind, M);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return ind2ptr_cpu(ind, M);
  }
}

SPARSE_API torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E) {
  if (at::musa::is_musa(ptr)) {
#ifdef WITH_MUSA
    return ptr2ind_musa(ptr, E);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return ptr2ind_cpu(ptr, E);
  }
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::ind2ptr", &ind2ptr)
                           .op("torch_sparse::ptr2ind", &ptr2ind);
