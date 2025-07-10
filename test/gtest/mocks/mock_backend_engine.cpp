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
#include "mock_backend_engine.h"
#include "gmock_engine.h"

namespace mocks {

MockBackendEngine::MockBackendEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      gmock_backend_engine(GMockBackendEngine::GetFromParams(init_params->customParams)),
      sharedState(1) {}

nixlStatus
MockBackendEngine::registerMem(const nixlBlobDesc &mem,
                               const nixlMemType &nixl_mem,
                               nixlBackendMD *&out) {
    sharedState++;
    return gmock_backend_engine->registerMem(mem, nixl_mem, out);
}

nixlStatus
MockBackendEngine::deregisterMem(nixlBackendMD *meta) {
    sharedState++;
    return gmock_backend_engine->deregisterMem(meta);
}

nixlStatus
MockBackendEngine::connect(const std::string &remote_agent) {
    sharedState++;
    return gmock_backend_engine->connect(remote_agent);
}

nixlStatus
MockBackendEngine::disconnect(const std::string &remote_agent) {
    sharedState++;
    return gmock_backend_engine->disconnect(remote_agent);
}

nixlStatus
MockBackendEngine::unloadMD(nixlBackendMD *input) {
    sharedState++;
    return gmock_backend_engine->unloadMD(input);
}

nixlStatus
MockBackendEngine::prepXfer(const nixlXferOp &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    assert(sharedState > 0);
    return gmock_backend_engine->prepXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixlStatus
MockBackendEngine::postXfer(const nixlXferOp &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    assert(sharedState > 0);
    return gmock_backend_engine->postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixlStatus
MockBackendEngine::checkXfer(nixlBackendReqH *handle) const {
    assert(sharedState > 0);
    return gmock_backend_engine->checkXfer(handle);
}

nixlStatus
MockBackendEngine::releaseReqH(nixlBackendReqH *handle) const {
    assert(sharedState > 0);
    return gmock_backend_engine->releaseReqH(handle);
}

nixlStatus
MockBackendEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                      const std::string &remote_conn_info) {
    sharedState++;
    return gmock_backend_engine->loadRemoteConnInfo(remote_agent, remote_conn_info);
}

nixlStatus
MockBackendEngine::loadRemoteMD(const nixlBlobDesc &input,
                                const nixlMemType &nixl_mem,
                                const std::string &remote_agent,
                                nixlBackendMD *&output) {
    sharedState++;
    return gmock_backend_engine->loadRemoteMD(input, nixl_mem, remote_agent, output);
}

nixlStatus
MockBackendEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    sharedState++;
    return gmock_backend_engine->loadLocalMD(input, output);
}

nixlStatus
MockBackendEngine::getNotifs(notif_list_t &notif_list) {
    sharedState++;
    return gmock_backend_engine->getNotifs(notif_list);
}

nixlStatus
MockBackendEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    assert(sharedState > 0);
    return gmock_backend_engine->genNotif(remote_agent, msg);
}

int
MockBackendEngine::progress() {
    sharedState++;
    return gmock_backend_engine->progress();
}
} // namespace mocks
