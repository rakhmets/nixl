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

#ifndef __GDS_MT_BACKEND_H
#define __GDS_MT_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <backend/backend_engine.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <cufile.h>
#include "gds_mt_utils.h"
#include "taskflow/core/executor.hpp"

class nixlGdsMtEngine : public nixlBackendEngine {
public:
    nixlGdsMtEngine (const nixlBackendInitParams *init_params);
    // Note: The destructor of the TaskFlow executor runs wait_for_all() to
    // wait for all submitted taskflows to complete and then notifies all worker
    // threads to stop and join these threads.
    ~nixlGdsMtEngine() = default;

    nixlGdsMtEngine (const nixlGdsMtEngine &) = delete;
    nixlGdsMtEngine &
    operator= (const nixlGdsMtEngine &) = delete;

    bool
    supportsNotif() const override {
        return false;
    }
    bool
    supportsRemote() const override {
        return false;
    }
    bool
    supportsLocal() const override {
        return true;
    }
    bool
    supportsProgTh() const override {
        return false;
    }

    nixlMemList
    getSupportedMems() const override {
        return {DRAM_SEG, VRAM_SEG, FILE_SEG};
    }

    nixlStatus
    connect (const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus
    disconnect (const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus
    loadLocalMD (nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixlStatus
    unloadMD (nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }
    nixlStatus
    registerMem (const nixlBlobDesc &mem, const nixlMemType &nixl_mem, nixlBackendMD *&out) override;
    nixlStatus
    deregisterMem (nixlBackendMD *meta) override;

    nixlStatus
    prepXfer (const nixlXferOp &operation,
              const nixl_meta_dlist_t &local,
              const nixl_meta_dlist_t &remote,
              const std::string &remote_agent,
              nixlBackendReqH *&handle,
              const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixlStatus
    postXfer (const nixlXferOp &operation,
              const nixl_meta_dlist_t &local,
              const nixl_meta_dlist_t &remote,
              const std::string &remote_agent,
              nixlBackendReqH *&handle,
              const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixlStatus
    checkXfer (nixlBackendReqH *handle) const override;
    nixlStatus
    releaseReqH (nixlBackendReqH *handle) const override;

private:
    gdsMtUtil gds_mt_utils_;
    std::unordered_map<int, std::weak_ptr<gdsMtFileHandle>> gds_mt_file_map_;
    size_t thread_count_;
    std::unique_ptr<tf::Executor> executor_;
};
#endif
