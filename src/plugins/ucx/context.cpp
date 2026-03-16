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

#include "context.h"

#include "common/hw_info.h"
#include "common/nixl_log.h"
#include "config.h"
#include "serdes/serdes.h"
#include "ucx_utils.h"

namespace {
[[nodiscard]] unsigned
makeUcpVersion() noexcept {
    unsigned major_version, minor_version, release_number;
    ucp_get_version(&major_version, &minor_version, &release_number);
    return UCP_VERSION(major_version, minor_version);
}

[[nodiscard]] nixl_ucx_mt_t
makeMtType(const bool prog_thread, const nixl_thread_sync_t sync_mode) noexcept {
    // With strict synchronization model nixlAgent serializes access to backends, with more
    // permissive models backends need to account for concurrent access and ensure their internal
    // state is properly protected. Progress thread creates internal concurrency in UCX backend
    // irrespective of nixlAgent synchronization model.
    return (sync_mode == nixl_thread_sync_t::NIXL_THREAD_SYNC_RW || prog_thread) ?
        nixl_ucx_mt_t::WORKER :
        nixl_ucx_mt_t::SINGLE;
}
} // namespace

namespace nixl::ucx {
context::context(const std::vector<std::string> &devs,
                 bool prog_thread,
                 unsigned long num_workers,
                 nixl_thread_sync_t sync_mode,
                 size_t num_device_channels,
                 const std::string &engine_config)
    : mtType_(makeMtType(prog_thread, sync_mode)),
      ucpVersion_(makeUcpVersion()) {

    ucp_params_t ucp_params;
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    ucp_params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
#ifdef HAVE_UCX_GPU_DEVICE_API
    ucp_params.features |= UCP_FEATURE_DEVICE;
#endif

    if (prog_thread) ucp_params.features |= UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = num_workers > 1 ? 1 : 0;

    nixl::ucx::config config;

    /* If requested, restrict the set of network devices */
    if (devs.size()) {
        /* TODO: check if this is the best way */
        std::string devs_str;
        for (const auto &dev : devs) {
            devs_str += dev + ":1,";
        }
        devs_str.pop_back(); // to remove odd comma after the last device
        config.modifyAlways("NET_DEVICES", devs_str.c_str());
    }

    config.modify("ADDRESS_VERSION", "v2");
    config.modify("RNDV_THRESH", "inf");
    config.modify("MAX_RMA_RAILS", "2");
    config.modify("IB_PCI_RELAXED_ORDERING", "try");
    config.modify("RCACHE_MAX_UNRELEASED", "1024");

    if (ucpVersion_ >= UCP_VERSION(1, 21)) {
        config.modify("RC_GDA_NUM_CHANNELS", std::to_string(num_device_channels));
    }

    if (ucpVersion_ >= UCP_VERSION(1, 19)) {
        config.modify("MAX_COMPONENT_MDS", "32");
    } else {
        NIXL_WARN << "UCX version is less than 1.19, CUDA support is limited, "
                  << "including the lack of support for multi-GPU within a single process.";
    }

    std::string elem;
    std::stringstream stream(engine_config);

    while (std::getline(stream, elem, ',')) {
        std::string_view elem_view = elem;
        size_t pos = elem_view.find('=');

        if (pos != std::string::npos) {
            config.modify(elem_view.substr(0, pos), elem_view.substr(pos + 1));
        }
    }

    const auto status = ucp_init(&ucp_params, config.getUcpConfig(), &ctx);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to create UCX context: " +
                                 std::string(ucs_status_string(status)));
    }
}

context::~context() {
    ucp_cleanup(ctx);
}

int
context::memReg(void *addr, size_t size, nixlUcxMem &mem, nixl_mem_t nixl_mem_type) {
    mem.base = addr;
    mem.size = size;

    ucp_mem_map_params_t mem_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
        .address = mem.base,
        .length = mem.size,
    };

    ucs_status_t status = ucp_mem_map(ctx, &mem_params, &mem.memh);
    if (status != UCS_OK) {
        NIXL_ERROR << "Failed to ucp_mem_map: " << ucs_status_string(status);
        return -1;
    }

    if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
        ucp_mem_attr_t attr;
        attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        status = ucp_mem_query(mem.memh, &attr);
        if (status != UCS_OK) {
            NIXL_ERROR << "Failed to ucp_mem_query: " << ucs_status_string(status);
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }

        if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
            NIXL_ERROR << "VRAM memory is detected as host by UCX. "
                          "UCX is likely not configured with CUDA support. "
                          "VRAM registration cannot proceed.";
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }
    }

    return 0;
}

std::string
context::packRkey(nixlUcxMem &mem) {
    void *rkey_buf;
    std::size_t size;

    const ucs_status_t status = ucp_rkey_pack(ctx, mem.memh, &rkey_buf, &size);
    if (status != UCS_OK) {
        NIXL_ERROR << "Failed to ucp_rkey_pack: " << ucs_status_string(status);
        return {};
    }
    const std::string result = nixlSerDes::_bytesToString(rkey_buf, size);
    ucp_rkey_buffer_release(rkey_buf);
    return result;
}

void
context::memDereg(nixlUcxMem &mem) {
    ucp_mem_unmap(ctx, mem.memh);
}

void
context::warnAboutHardwareSupportMismatch() const {
    ucp_context_attr_t attr = {
        .field_mask = UCP_ATTR_FIELD_MEMORY_TYPES,
    };
    const auto status = ucp_context_query(ctx, &attr);
    if (status != UCS_OK) {
        NIXL_WARN << "Failed to query UCX context: " << ucs_status_string(status) << ", "
                  << "hardware support mismatch check will be skipped";
        return;
    }

    const nixl::hwInfo hw_info;

    NIXL_DEBUG << "hwInfo { "
               << "numNvidiaGpus=" << hw_info.numNvidiaGpus << ", "
               << "numIbDevices=" << hw_info.numIbDevices << " }";

    if (hw_info.numNvidiaGpus > 0 && !UCS_BIT_GET(attr.memory_types, UCS_MEMORY_TYPE_CUDA)) {
        NIXL_WARN << hw_info.numNvidiaGpus
                  << " NVIDIA GPU(s) were detected, but UCX CUDA support was not found! "
                  << "GPU memory is not supported.";
    }

    if (ucpVersion_ >= UCP_VERSION(1, 21)) {
        // `UCS_MEMORY_TYPE_RDMA` is included in `memory_types` only from UCX 1.21
        if (hw_info.numIbDevices > 0 && !UCS_BIT_GET(attr.memory_types, UCS_MEMORY_TYPE_RDMA)) {
            NIXL_WARN << hw_info.numIbDevices
                      << " IB device(s) were detected, but accelerated IB support was not found! "
                         "Performance may be degraded.";
        }
    }
}
} // namespace nixl::ucx
