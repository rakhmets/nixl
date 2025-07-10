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
#include "common.h"

namespace mocks {
namespace backend_plugin {

    static nixlBackendEngine *
    create_engine(const nixlBackendInitParams *params) {
        return new MockBackendEngine(params);
    }

    static void
    destroy_engine(nixlBackendEngine *engine) {
        delete engine;
    }

    static const char *
    get_plugin_name() {
        return gtest::GetMockBackendName();
    }

    static const char *
    get_plugin_version() {
        return "0.0.1";
    }

    static nixlBParams
    get_backend_options() {
        return nixlBParams();
    }

    static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                       create_engine,
                                       destroy_engine,
                                       get_plugin_name,
                                       get_plugin_version,
                                       get_backend_options};
} // namespace backend_plugin

} // namespace mocks

extern "C" nixlBackendPlugin *
nixl_plugin_init() {
    return &mocks::backend_plugin::plugin;
}

extern "C" void
nixl_plugin_fini() {}
