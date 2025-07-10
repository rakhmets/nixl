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

#ifndef POSIX_BACKEND_H
#define POSIX_BACKEND_H

#include <memory>
#include <string>
#include <vector>
#include <absl/strings/str_format.h>
#include "backend/backend_engine.h"
#include "posix_queue.h"

class nixlPosixBackendReqH : public nixlBackendReqH {
private:
    const nixlXferOp            &operation;      // The transfer operation (read/write)
    const nixl_meta_dlist_t         &local;          // Local memory descriptor list
    const nixl_meta_dlist_t         &remote;         // Remote memory descriptor list
    const nixl_opt_b_args_t         *opt_args;       // Optional backend-specific arguments
    const nixlBParams           *custom_params_; // Custom backend parameters
    const int                       queue_depth_;    // Queue depth for async I/O
    std::unique_ptr<nixlPosixQueue> queue;           // Async I/O queue instance
    const nixlPosixQueue::queue_t   queue_type_;     // Type of queue used

    nixlStatus initQueues();                      // Initialize async I/O queue

public:
    nixlPosixBackendReqH(const nixlXferOp &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const nixl_opt_b_args_t* opt_args,
                         const nixlBParams* custom_params);
    ~nixlPosixBackendReqH() {};

    nixlStatus postXfer();
    nixlStatus prepXfer();
    nixlStatus checkXfer();

    // Exception classes
    class exception: public std::exception {
        private:
            const nixlStatus code_;
        public:
            exception(const std::string& msg, nixlStatus code)
                : std::exception(), code_(code) {}
            nixlStatus code() const noexcept { return code_; }
    };
};

class nixlPosixEngine : public nixlBackendEngine {
private:
    const nixlPosixQueue::queue_t queue_type_;

public:
    nixlPosixEngine(const nixlBackendInitParams* init_params);
    virtual ~nixlPosixEngine() = default;

    bool supportsRemote() const override {
        return false;
    }

    bool supportsLocal() const override {
        return true;
    }

    bool supportsNotif() const override {
        return false;
    }

    bool supportsProgTh() const override {
        return false;
    }

    nixlMemList getSupportedMems() const override {
        return {FILE_SEG, DRAM_SEG};
    }

    nixlStatus registerMem(const nixlBlobDesc &mem,
                              const nixlMemType &nixl_mem,
                              nixlBackendMD* &out) override;

    nixlStatus deregisterMem(nixlBackendMD* meta) override;

    nixlStatus connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixlStatus unloadMD(nixlBackendMD* input) override {
        return NIXL_SUCCESS;
    }

    nixlStatus prepXfer(const nixlXferOp &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixlStatus postXfer(const nixlXferOp &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixlStatus checkXfer(nixlBackendReqH* handle) const override;
    nixlStatus releaseReqH(nixlBackendReqH* handle) const override;

    nixlStatus loadLocalMD(nixlBackendMD* input, nixlBackendMD* &output) override {
        output = input;
        return NIXL_SUCCESS;
    }
};

#endif // POSIX_BACKEND_H
