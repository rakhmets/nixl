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

#include "backend/backend_plugin.h"
#include "gds_mt_backend.h"
#include "common/nixl_log.h"
#include <exception>

static const char *PLUGIN_NAME = "GDS_MT";
static const char *PLUGIN_VERSION = "0.1.0";

static nixlBackendEngine *
create_gds_mt_engine (const nixlBackendInitParams *init_params) {
    try {
        return new nixlGdsMtEngine (init_params);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "GDS_MT: Failed to create engine: " << e.what();
        return nullptr;
    }
}

static void
destroy_gds_mt_engine (nixlBackendEngine *engine) {
    delete engine;
}

static const char *
get_plugin_name() {
    return PLUGIN_NAME;
}

static const char *
get_plugin_version() {
    return PLUGIN_VERSION;
}

static nixlBParams
get_backend_options() {
    return {};
}

static nixlMemList
get_backend_mems() {
    return {DRAM_SEG, VRAM_SEG, FILE_SEG};
}

static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                   create_gds_mt_engine,
                                   destroy_gds_mt_engine,
                                   get_plugin_name,
                                   get_plugin_version,
                                   get_backend_options,
                                   get_backend_mems};

#ifdef STATIC_PLUGIN_GDS_MT

nixlBackendPlugin *
createStaticGdsMtPlugin() {
    return &plugin;
}

#else

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &plugin;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif
