/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_event.hpp"

#include <torch/types.h>

#include <cuda_runtime.h>

#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace nixl_ep {
namespace detail {
    inline void
    push_back(std::vector<torch::Tensor> &tensors, const torch::Tensor &tensor) {
        tensors.push_back(tensor);
    }

    inline void
    push_back(std::vector<torch::Tensor> &tensors, const std::optional<torch::Tensor> &tensor) {
        if (tensor) {
            tensors.push_back(*tensor);
        }
    }
} // namespace detail

// Emulates torch::Tensor::record_stream(). The method is not exposed in stable torch ABI.
// This is an intermediate step of porting to stable torch ABI. It is required to get rid of
// at::cuda::CUDAStream. And later we can swap torch::Tensor for torch::stable::Tensor.
class TensorHolders {
public:
    template<typename... Tensors>
    void
    record(const std::vector<cudaStream_t> &streams, const Tensors &...args) {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(sizeof...(Tensors));
        (detail::push_back(tensors, args), ...);
        for (cudaStream_t stream : streams) {
            update(tensors, stream);
        }
    }

    // Must only be called after synchronizing the streams passed to record().
    void
    clear();

private:
    void
    update(const std::vector<torch::Tensor> &tensors, cudaStream_t stream);

    std::mutex mutex;
    std::vector<std::pair<cuda::Event, std::vector<torch::Tensor> *>> holders;
};
} // namespace nixl_ep
