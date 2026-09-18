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

#include "kernels/exception.cuh"

#include <torch/types.h>

#include <cuda_runtime.h>

#include <optional>
#include <vector>

namespace nixl_ep {

namespace detail {

    inline void
    append_tensor(std::vector<torch::Tensor> &tensors, const torch::Tensor &tensor) {
        tensors.push_back(tensor);
    }

    inline void
    append_tensor(std::vector<torch::Tensor> &tensors, const std::optional<torch::Tensor> &tensor) {
        if (tensor) {
            tensors.push_back(*tensor);
        }
    }

    inline void
    record_stream_impl(const std::vector<torch::Tensor> &tensors, cudaStream_t stream) {
        EP_HOST_ASSERT(!tensors.empty());
        auto *holder = new std::vector<torch::Tensor>(tensors);
        CUDA_CHECK(cudaLaunchHostFunc(
            stream,
            [](void *data) { delete static_cast<std::vector<torch::Tensor> *>(data); },
            holder));
    }

} // namespace detail

// Emulates torch::Tensor::record_stream(). The method is not exposed in stable torch ABI.
// This is an intermediate step of porting to stable torch ABI. It is required to get rid of
// at::cuda::CUDAStream. And later we can swap torch::Tensor for torch::stable::Tensor.
template<typename... Tensors>
void
record_stream(const std::vector<cudaStream_t> &streams, const Tensors &...tensors) {
    std::vector<torch::Tensor> out;
    out.reserve(sizeof...(Tensors));
    (detail::append_tensor(out, tensors), ...);
    for (cudaStream_t stream : streams) {
        detail::record_stream_impl(out, stream);
    }
}

} // namespace nixl_ep
