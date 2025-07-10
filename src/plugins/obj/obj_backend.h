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

#ifndef OBJ_BACKEND_H
#define OBJ_BACKEND_H

#include "obj_executor.h"
#include "obj_s3_client.h"
#include <string>
#include <memory>
#include <unordered_map>
#include "backend/backend_engine.h"

class nixlObjEngine : public nixlBackendEngine {
public:
    nixlObjEngine(const nixlBackendInitParams *init_params);
    nixlObjEngine(const nixlBackendInitParams *init_params, std::shared_ptr<IS3Client> s3_client);
    virtual ~nixlObjEngine();

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    bool
    supportsProgTh() const override {
        return false;
    }

    nixlMemList
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixlStatus
    registerMem(const nixlBlobDesc &mem, const nixlMemType &nixl_mem, nixlBackendMD *&out) override;

    nixlStatus
    deregisterMem(nixlBackendMD *meta) override;

    nixlStatus
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixlStatus
    prepXfer(const nixlXferOp &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixlStatus
    postXfer(const nixlXferOp &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixlStatus
    checkXfer(nixlBackendReqH *handle) const override;
    nixlStatus
    releaseReqH(nixlBackendReqH *handle) const override;

    nixlStatus
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

private:
    std::shared_ptr<AsioThreadPoolExecutor> executor_;
    std::shared_ptr<IS3Client> s3Client_;
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
};

#endif // OBJ_BACKEND_H
