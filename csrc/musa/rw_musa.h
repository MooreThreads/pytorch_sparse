#pragma once

#include "../extensions.h"

torch::Tensor random_walk_musa(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor start, int64_t walk_length);
