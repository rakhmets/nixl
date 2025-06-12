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

#ifndef NIXL_SRC_UTILS_UCX_UCX_CONFIG_H
#define NIXL_SRC_UTILS_UCX_UCX_CONFIG_H

extern "C"
{
#include <ucp/api/ucp.h>
}

#include <string_view>

namespace nixl {

class UcxConfig {
public:
    UcxConfig();
    ~UcxConfig();

    UcxConfig(const UcxConfig&) = delete;
    UcxConfig& operator=(const UcxConfig&) = delete;
    UcxConfig(UcxConfig&&) = delete;
    UcxConfig& operator=(UcxConfig&&) = delete;

    [[nodiscard]] ucp_config_t* get() const;
    void modify(std::string_view name, std::string_view value);

private:
    ucp_config_t* ucp_config;
};

}

#endif
