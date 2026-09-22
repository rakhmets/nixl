# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads nixl_ep_cpp with RTLD_DEEPBIND so it binds its own bundled UCX
symbols instead of a UCX already loaded globally in the process (e.g. by
HPC-X/OpenMPI). Opt-in via NIXL_UCX_DEEPBIND (default off)."""

import importlib
import os
import sys

_deepbind = getattr(os, "RTLD_DEEPBIND", 0)
_enabled = (
    bool(_deepbind)
    and hasattr(sys, "getdlopenflags")
    and os.getenv("NIXL_UCX_DEEPBIND", "0").strip().lower()
    not in {"0", "false", "no", "off", "disable", "disabled"}
)

if _enabled:
    _old_flags = sys.getdlopenflags()
    sys.setdlopenflags(_old_flags | _deepbind)
try:
    nixl_ep_cpp = importlib.import_module(".nixl_ep_cpp", __package__)
finally:
    if _enabled:
        sys.setdlopenflags(_old_flags)
