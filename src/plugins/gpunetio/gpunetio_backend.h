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

#ifndef GPUNETIO_BACKEND_H
#define GPUNETIO_BACKEND_H

#include "gpunetio_backend_aux.h"

class nixlDocaEngine : public nixlBackendEngine {
public:
    CUcontext main_cuda_ctx;
    int oob_sock_server;
    std::mutex notifLock;
    std::mutex qpLock;
    mutable std::mutex connectLock;
    std::vector<std::pair<uint32_t, struct doca_gpu *>> gdevs; /* List of DOCA GPUNetIO device handlers */
    struct doca_dev *ddev; /* DOCA device handler associated to queues */
    nixlStatus addRdmaQp (const std::string &remote_agent);
    nixlStatus connectServerRdmaQp (int oob_sock_client, const std::string &remote_agent);
    nixlStatus nixlDocaInitNotif (const std::string &remote_agent, struct doca_dev *dev, struct doca_gpu *gpu);

    volatile uint8_t pthrStop, pthrActive;
    nixlDocaEngine (const nixlBackendInitParams *init_params);
    ~nixlDocaEngine();

    bool supportsRemote() const {
        return true;
    }
    bool supportsLocal() const {
        return false;
    }
    bool supportsNotif() const {
        return true;
    }
    bool supportsProgTh() const {
        return false;
    }

    nixlMemList
    getSupportedMems() const;

    /* Object management */
    nixlStatus
    getPublicData (const nixlBackendMD *meta, std::string &str) const override;
    nixlStatus
    getConnInfo (std::string &str) const override;
    nixlStatus
    loadRemoteConnInfo (const std::string &remote_agent,
                        const std::string &remote_conn_info) override;

    nixlStatus
    connect (const std::string &remote_agent) override;
    nixlStatus
    disconnect (const std::string &remote_agent) override;

    nixlStatus
    registerMem (const nixlBlobDesc &mem, const nixlMemType &nixl_mem, nixlBackendMD *&out) override;
    nixlStatus
    deregisterMem (nixlBackendMD *meta) override;

    nixlStatus
    loadRemoteMD (const nixlBlobDesc &input,
                  const nixlMemType &nixl_mem,
                  const std::string &remote_agent,
                  nixlBackendMD *&output) override;
    nixlStatus
    unloadMD (nixlBackendMD *input) override;

    // Data transfer
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

    nixlStatus
    getNotifs (notif_list_t &notif_list);
    nixlStatus
    genNotif (const std::string &remote_agent, const std::string &msg) const override;

    void addConnection (struct doca_rdma_connection *connection);
    uint32_t getConnectionLast();
    void removeConnection (uint32_t connection_idx);
    uint32_t getGpuCudaId();

    nixlStatus sendLocalAgentName (int oob_sock_client);
    nixlStatus recvRemoteAgentName (int oob_sock_client, std::string &remote_agent);

private:
    struct doca_log_backend *sdk_log;
    std::string msg_tag_start = "DOCAS";
    std::string msg_tag_end = "DOCAE";
    std::vector<struct nixlDocaRdmaQp> rdma_qp_v;
    int nstreams;

    uint32_t local_port;
    int noSyncIters;
    uint8_t ipv4_addr[4];
    std::thread pthr;
    uint64_t *last_rsvd_flags;
    uint64_t *last_posted_flags;
    cudaStream_t post_stream[DOCA_POST_STREAM_NUM];
    cudaStream_t wait_stream;
    mutable std::atomic<uint32_t> xferStream;
    mutable std::atomic<uint32_t> lastPostedReq;

    struct docaXferReqGpu *xferReqRingGpu;
    struct docaXferReqGpu *xferReqRingCpu;
    mutable std::atomic<uint32_t> xferRingPos;

    struct docaXferCompletion *completion_list_gpu;
    struct docaXferCompletion *completion_list_cpu;
    uint32_t *wait_exit_gpu;
    uint32_t *wait_exit_cpu;
    int oob_sock_client;
    struct docaNotifRecv *notif_fill_gpu;
    struct docaNotifRecv *notif_fill_cpu;
    struct docaNotifRecv *notif_progress_gpu;
    struct docaNotifRecv *notif_progress_cpu;

    struct docaNotifSend *notif_send_gpu;
    struct docaNotifSend *notif_send_cpu;

    // Map of agent name to saved nixlDocaConnection info
    std::unordered_map<std::string, nixlDocaConnection> remoteConnMap;
    std::unordered_map<std::string, struct nixlDocaRdmaQp *> qpMap;
    std::unordered_map<std::string, int> connMap;
    std::unordered_map<std::string, struct nixlDocaNotif *> notifMap;

    pthread_t server_thread_id;

    class nixlDocaBckndReq : public nixlBackendReqH {
    private:
    public:
        cudaStream_t stream;
        uint32_t devId;
        uint32_t start_pos;
        uint32_t end_pos;
        uintptr_t backendHandleGpu;

        nixlDocaBckndReq() : nixlBackendReqH() {}

        ~nixlDocaBckndReq() {}
    };

    nixlStatus progressThreadStart();
    void progressThreadStop();


    nixlStatus
    connectClientRdmaQp (int oob_sock_client, const std::string &remote_agent);
    nixlStatus
    nixlDocaDestroyNotif (struct doca_gpu *gpu, struct nixlDocaNotif *notif);

    mutable std::mutex notifSendLock;

};

#endif
