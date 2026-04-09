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

#ifndef NIXL_SRC_PLUGINS_UCX_CONNECTION_H
#define NIXL_SRC_PLUGINS_UCX_CONNECTION_H

#include "backend/backend_aux.h"

#include <memory>
#include <vector>

class nixlUcxEp;

namespace nixl::ucx {
class connection : public nixlBackendConnMD {
public:
    explicit connection(std::vector<std::unique_ptr<nixlUcxEp>> eps) : eps_(std::move(eps)) {}

    [[nodiscard]] const nixlUcxEp &
    getEp(size_t ep_id) const noexcept {
        return *eps_[ep_id];
    }

private:
    const std::vector<std::unique_ptr<nixlUcxEp>> eps_;
};
} // namespace nixl::ucx
#endif
