#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
spmm_musa(torch::Tensor rowptr, torch::Tensor col,
          std::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

torch::Tensor spmm_value_bw_musa(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce);
