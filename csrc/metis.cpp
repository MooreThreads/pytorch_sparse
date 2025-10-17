#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "cpu/metis_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_MUSA
PyMODINIT_FUNC PyInit__metis_musa(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__metis_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor partition(torch::Tensor rowptr, torch::Tensor col,
                        std::optional<torch::Tensor> optional_value,
                        int64_t num_parts, bool recursive) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return partition_cpu(rowptr, col, optional_value, std::nullopt, num_parts,
                         recursive);
  }
}

SPARSE_API torch::Tensor partition2(torch::Tensor rowptr, torch::Tensor col,
                         std::optional<torch::Tensor> optional_value,
                         std::optional<torch::Tensor> optional_node_weight,
                         int64_t num_parts, bool recursive) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return partition_cpu(rowptr, col, optional_value, optional_node_weight,
                         num_parts, recursive);
  }
}

SPARSE_API torch::Tensor mt_partition(torch::Tensor rowptr, torch::Tensor col,
                           std::optional<torch::Tensor> optional_value,
                           std::optional<torch::Tensor> optional_node_weight,
                           int64_t num_parts, bool recursive,
                           int64_t num_workers) {
  if (at::musa::is_musa(rowptr)) {
#ifdef WITH_MUSA
    AT_ERROR("No MUSA version supported");
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  } else {
    return mt_partition_cpu(rowptr, col, optional_value, optional_node_weight,
                            num_parts, recursive, num_workers);
  }
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::partition", &partition)
                           .op("torch_sparse::partition2", &partition2)
                           .op("torch_sparse::mt_partition", &mt_partition);
