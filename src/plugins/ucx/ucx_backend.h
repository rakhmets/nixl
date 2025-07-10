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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H
#define NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <poll.h>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "ucx/rkey.h"
#include "ucx/ucx_utils.h"
#include "common/list_elem.h"

enum ucx_cb_op_t {CONN_CHECK, NOTIF_STR, DISCONNECT};

class nixlUcxConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        std::vector<std::unique_ptr<nixlUcxEp>> eps;

    public:
        [[nodiscard]] const std::unique_ptr<nixlUcxEp>& getEp(size_t ep_id) const noexcept {
            return eps[ep_id];
        }

    friend class nixlUcxEngine;
};

using ucx_connection_ptr_t = std::shared_ptr<nixlUcxConnection>;

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
    private:
        nixlUcxMem mem;
        nixlBlob rkeyStr;

    public:
        nixlUcxPrivateMetadata() : nixlBackendMD(true) {
        }

        [[nodiscard]] const std::string& get() const noexcept {
            return rkeyStr;
        }

    friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {
public:
    nixlUcxPublicMetadata() : nixlBackendMD(false) {}

    [[nodiscard]] const nixl::ucx::rkey &
    getRkey(size_t id) const {
        return *rkeys_[id];
    }

    void
    addRkey(const nixlUcxEp &ep, const void *rkey_buffer) {
        rkeys_.emplace_back(std::make_unique<nixl::ucx::rkey>(ep, rkey_buffer));
    }

    ucx_connection_ptr_t conn;

private:
    std::vector<std::unique_ptr<nixl::ucx::rkey>> rkeys_;
};

// Forward declaration of CUDA context
// It is only visible in ucx_backend.cpp to ensure that
// HAVE_CUDA works properly
// Once we will introduce static config (i.e. config.h) that
// will be part of NIXL installation - we can have
// HAVE_CUDA in h-files
class nixlUcxCudaCtx;
class nixlUcxCudaDevicePrimaryCtx;
using nixlUcxCudaDevicePrimaryCtxPtr = std::shared_ptr<nixlUcxCudaDevicePrimaryCtx>;

class nixlUcxEngine
    : public nixlBackendEngine {
    private:
        /* UCX data */
        std::unique_ptr<nixlUcxContext> uc;
        std::vector<std::unique_ptr<nixlUcxWorker>> uws;
        std::string workerAddr;

        /* Progress thread data */
        std::mutex pthrActiveLock;
        std::condition_variable pthrActiveCV;
        bool pthrActive;
        bool pthrOn;
        std::thread pthr;
        std::chrono::milliseconds pthrDelay;
        int pthrControlPipe[2];
        std::vector<pollfd> pollFds;

        /* CUDA data*/
        std::unique_ptr<nixlUcxCudaCtx> cudaCtx; // Context matching specific device
        bool cuda_addr_wa;

        // Context to use when current context is missing
        nixlUcxCudaDevicePrimaryCtxPtr m_cudaPrimaryCtx;

        /* Notifications */
        notif_list_t notifMainList;
        std::mutex  notifMtx;
        notif_list_t notifPthrPriv, notifPthr;

        // Map of agent name to saved nixlUcxConnection info
        std::unordered_map<std::string, ucx_connection_ptr_t,
                           std::hash<std::string>, strEqual> remoteConnMap;


        void vramInitCtx();
        void vramFiniCtx();
        int vramUpdateCtx(void *address, uint64_t devId, bool &restart_reqd);
        int vramApplyCtx();

        // Threading infrastructure
        //   TODO: move the thread management one outside of NIXL common infra
        void progressFunc();
        void progressThreadStart();
        void progressThreadStop();
        void progressThreadRestart();
        bool isProgressThread() const noexcept {
            return std::this_thread::get_id() == pthr.get_id();
        }

        // Connection helper
        static ucs_status_t
        connectionCheckAmCb(void *arg, const void *header,
                            size_t header_length, void *data,
                            size_t length,
                            const ucp_am_recv_param_t *param);

        static ucs_status_t
        connectionTermAmCb(void *arg, const void *header,
                           size_t header_length, void *data,
                           size_t length,
                           const ucp_am_recv_param_t *param);

        // Memory management helpers
        nixlStatus internalMDHelper (const nixlBlob &blob,
                                        const std::string &agent,
                                        nixlBackendMD* &output);

        // Notifications
        static ucs_status_t notifAmCb(void *arg, const void *header,
                                      size_t header_length, void *data,
                                      size_t length,
                                      const ucp_am_recv_param_t *param);
        nixlStatus notifSendPriv(const std::string &remote_agent,
                                    const std::string &msg,
                                    nixlUcxReq &req,
                                    size_t worker_id) const;
        void notifProgress();
        void notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt);

    public:
        nixlUcxEngine(const nixlBackendInitParams* init_params);
        ~nixlUcxEngine();

        bool supportsRemote() const override { return true; }
        bool supportsLocal() const override { return true; }
        bool supportsNotif() const override { return true; }
        bool supportsProgTh() const override { return pthrOn; }

        nixlMemList getSupportedMems() const override;

        /* Object management */
        nixlStatus getPublicData (const nixlBackendMD* meta,
                                     std::string &str) const override;
        nixlStatus getConnInfo(std::string &str) const override;
        nixlStatus loadRemoteConnInfo (const std::string &remote_agent,
                                          const std::string &remote_conn_info) override;

        nixlStatus connect(const std::string &remote_agent) override;
        nixlStatus disconnect(const std::string &remote_agent) override;

        nixlStatus registerMem (const nixlBlobDesc &mem,
                                   const nixlMemType &nixl_mem,
                                   nixlBackendMD* &out) override;
        nixlStatus deregisterMem (nixlBackendMD* meta) override;

        nixlStatus loadLocalMD (nixlBackendMD* input,
                                   nixlBackendMD* &output) override;

        nixlStatus loadRemoteMD (const nixlBlobDesc &input,
                                    const nixlMemType &nixl_mem,
                                    const std::string &remote_agent,
                                    nixlBackendMD* &output) override;
        nixlStatus unloadMD (nixlBackendMD* input) override;

        // Data transfer
        nixlStatus prepXfer (const nixlXferOp &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr) const override;

        nixlStatus estimateXferCost(const nixlXferOp &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* const &handle,
                                       std::chrono::microseconds &duration,
                                       std::chrono::microseconds &err_margin,
                                       nixlCost &method,
                                       const nixlAgentOptionalArgs* opt_args=nullptr) const override;

        nixlStatus postXfer (const nixlXferOp &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr) const override;

        nixlStatus checkXfer (nixlBackendReqH* handle) const override;
        nixlStatus releaseReqH(nixlBackendReqH* handle) const override;

        int progress() override;

        nixlStatus getNotifs(notif_list_t &notif_list);
        nixlStatus genNotif(const std::string &remote_agent, const std::string &msg) const override;

        //public function for UCX worker to mark connections as connected
        nixlStatus checkConn(const std::string &remote_agent);
        nixlStatus endConn(const std::string &remote_agent);

        const std::unique_ptr<nixlUcxWorker> &getWorker(size_t worker_id) const {
            return uws[worker_id];
        }

        size_t getWorkerId() const {
            return std::hash<std::thread::id>{}(std::this_thread::get_id()) % uws.size();
        }
};

#endif
