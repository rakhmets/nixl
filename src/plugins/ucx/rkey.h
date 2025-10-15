/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_UTILS_UCX_RKEY_H
#define NIXL_SRC_UTILS_UCX_RKEY_H

#include <memory>

extern "C" {
#include <ucp/api/ucp.h>
}

class nixlUcxEp;

namespace nixl::ucx {
class rkey {
public:
    rkey() = delete;
    rkey(const nixlUcxEp &, const void *rkey_buffer);

    [[nodiscard]] ucp_rkey_h
    get() const noexcept {
        return rkey_.get();
    }

private:
    [[nodiscard]] static ucp_rkey_h
    unpackUcpRkey(const nixlUcxEp &, const void *rkey_buffer);

    const std::unique_ptr<ucp_rkey, void (*)(ucp_rkey_h)> rkey_;
};
} // namespace nixl::ucx
#endif
