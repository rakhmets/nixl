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

#include "tensor_holders.hpp"

#include "kernels/exception.cuh"

namespace nixl_ep {
void
TensorHolders::clear() {
    for (auto &entry : holders) {
        delete entry.second;
    }
    holders.clear();
}

void
TensorHolders::update(const std::vector<torch::Tensor> &tensors, cudaStream_t stream) {
    EP_HOST_ASSERT(!tensors.empty());

    // The deletion of the holder below is not safe under graph capture.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    EP_HOST_ASSERT(capture_status == cudaStreamCaptureStatusNone);

    auto *holder = new std::vector<torch::Tensor>(tensors);
    cuda::Event event;
    event.record(stream);

    const std::lock_guard<std::mutex> lock(mutex);

    // Keep links to tensors until the event is ready.
    auto it = holders.begin();
    while (it != holders.end()) {
        if (it->first.is_ready()) {
            delete it->second;
            it = holders.erase(it);
        } else {
            ++it;
        }
    }

    holders.emplace_back(std::move(event), holder);
}
} // namespace nixl_ep
