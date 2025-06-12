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

#include "ucx_config.h"

#include "common/nixl_log.h"

#include <stdexcept>

namespace nixl {

UcxConfig::UcxConfig()
{
    const auto status = ucp_config_read(NULL, NULL, &ucp_config);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to read UCX config");
    }
}

UcxConfig::~UcxConfig()
{
    ucp_config_release(ucp_config);
}

ucp_config_t* UcxConfig::get() const
{
    return ucp_config;
}

void UcxConfig::modify(std::string_view name, std::string_view value)
{
    const auto status = ucp_config_modify(ucp_config, name.data(), value.data());
    if (status != UCS_OK) {
        NIXL_WARN << "Failed to modify " << name << ": " << ucs_status_string(status);
    }
}

}
