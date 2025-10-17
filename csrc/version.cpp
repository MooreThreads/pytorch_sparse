#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#ifdef WITH_MUSA
#ifdef USE_ROCM
#include <hip/hip_version.h>
#else
#include <musa.h>
#endif
#endif

#include "macros.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__version_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif
#endif

namespace sparse {
SPARSE_API int64_t musa_version() noexcept {
#ifdef WITH_MUSA
#ifdef USE_ROCM
  return HIP_VERSION;
#else
  return MUSA_VERSION;
#endif
#else
  return -1;
#endif
}
} // namespace sparse

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::musa_version", [] { return sparse::musa_version(); });
