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
#ifndef __UCX_MO_BACKEND_H
#define __UCX_MO_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <cassert>
#include <memory>

#include "nixl.h"
#include "ucx_backend.h"

// Local includes
#include <common/nixl_time.h>
#include <common/list_elem.h>
#include <ucx/ucx_utils.h>

class nixlUcxMoConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        uint32_t num_engines;

    public:
        // Extra information required for UCX connections

    friend class nixlUcxMoEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlUcxMoPrivateMetadata : public nixlBackendMD
{
private:
    uint32_t eidx;
    nixlBackendMD *md;
    nixlMemType  memType;
    nixlBlob rkeyStr;
public:
    nixlUcxMoPrivateMetadata() : nixlBackendMD(true) {
    }

    ~nixlUcxMoPrivateMetadata(){
    }

    std::string get() const {
        return rkeyStr;
    }

    friend class nixlUcxMoEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxMoPublicMetadata : public nixlBackendMD
{
    uint32_t eidx;
    nixlUcxMoConnection conn;
    std::vector<nixlBackendMD*> int_mds;

public:
    nixlUcxMoPublicMetadata() : nixlBackendMD(false) {}

    ~nixlUcxMoPublicMetadata(){
    }

    friend class nixlUcxMoEngine;
};



class nixlUcxMoEngine : public nixlBackendEngine {
private:
    uint32_t _engineCnt;
    uint32_t _gpuCnt;
    int setEngCnt(uint32_t host_engines);
    uint32_t getEngCnt();
    int32_t getEngIdx(nixlMemType type, uint64_t devId);
    std::string getEngName(const std::string &baseName, uint32_t eidx) const;
    std::string getEngBase(const std::string &engName);
    bool pthrOn;

    // UCX backends data
    std::vector<std::unique_ptr<nixlBackendEngine>> engines;
    // Map of agent name to saved nixlUcxConnection info
    using remote_conn_map_t = std::map<std::string, nixlUcxMoConnection>;
    using remote_comm_it_t = remote_conn_map_t::iterator;
    remote_conn_map_t remoteConnMap;

    // Memory helper
    nixlStatus internalMDHelper (const nixlBlob &blob,
                                    const nixlMemType &nixl_mem,
                                    const std::string &agent,
                                    nixlBackendMD* &output);

public:
    nixlUcxMoEngine(const nixlBackendInitParams* init_params);
    ~nixlUcxMoEngine() = default;

    bool supportsRemote () const { return true; }
    bool supportsLocal  () const { return false; }
    bool supportsNotif  () const { return true; }
    bool supportsProgTh () const { return pthrOn; }

    nixlMemList getSupportedMems () const;

    /* Object management */
    nixlStatus getPublicData (const nixlBackendMD* meta,
                                 std::string &str) const;
    nixlStatus getConnInfo(std::string &str) const;
    nixlStatus loadRemoteConnInfo (const std::string &remote_agent,
                                        const std::string &remote_conn_info);

    nixlStatus connect(const std::string &remote_agent);
    nixlStatus disconnect(const std::string &remote_agent);

    nixlStatus registerMem (const nixlBlobDesc &mem,
                               const nixlMemType &nixl_mem,
                               nixlBackendMD* &out);
    nixlStatus deregisterMem (nixlBackendMD* meta);

    nixlStatus loadLocalMD (nixlBackendMD* input,
                               nixlBackendMD* &output);

    nixlStatus loadRemoteMD (const nixlBlobDesc &input,
                                const nixlMemType &nixl_mem,
                                const std::string &remote_agent,
                                nixlBackendMD* &output);
    nixlStatus unloadMD (nixlBackendMD* input);

    // Data transfer
    nixlStatus prepXfer (const nixlXferOp &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH* &handle,
                            const nixl_opt_b_args_t* opt_args=nullptr) const;

    nixlStatus postXfer (const nixlXferOp &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH* &handle,
                            const nixl_opt_b_args_t* opt_args=nullptr) const;
    nixlStatus checkXfer (nixlBackendReqH* handle) const;
    nixlStatus releaseReqH(nixlBackendReqH* handle) const;

    int progress();

    nixlStatus getNotifs(notif_list_t &notif_list);
    nixlStatus genNotif(const std::string &remote_agent, const std::string &msg) const;

    //public function for UCX worker to mark connections as connected
    nixlStatus checkConn(const std::string &remote_agent);
    nixlStatus endConn(const std::string &remote_agent);
};

#endif
