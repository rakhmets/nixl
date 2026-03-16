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

#ifndef NIXL_SRC_PLUGINS_UCX_CONTEXT_H
#define NIXL_SRC_PLUGINS_UCX_CONTEXT_H

#include "nixl_types.h"
#include "ucx_utils.h"

extern "C" {
#include <ucp/api/ucp.h>
}

#include <string>
#include <string_view>
#include <vector>

namespace nixl::ucx {
class context {
public:
    context(const std::vector<std::string> &devs,
                   bool prog_thread,
                   unsigned long num_workers,
                   nixl_thread_sync_t sync_mode,
                   size_t num_device_channels,
                   const std::string &engine_conf = "");
    ~context();

    context(context &&) = delete;
    context(const context &) = delete;

    void
    operator=(context &&) = delete;
    void
    operator=(const context &) = delete;

    int
    memReg(void *addr, size_t size, nixlUcxMem &mem, nixl_mem_t nixl_mem_type);

    [[nodiscard]] std::string
    packRkey(nixlUcxMem &mem);

    void
    memDereg(nixlUcxMem &mem);

    void
    warnAboutHardwareSupportMismatch() const;

private:
    ucp_context_h ctx;
    const nixl_ucx_mt_t mtType_;
    const unsigned ucpVersion_;
};
} // namespace nixl::ucx
#endif
