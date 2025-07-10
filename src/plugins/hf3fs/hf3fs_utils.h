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
#ifndef __HF3FS_UTILS_H
#define __HF3FS_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include "hf3fs_usrbio.h"



class hf3fsFileHandle {
public:
    int            fd;
    // -1 means inf size file?
    size_t         size;
    std::string    metadata;
    std::string    mount_point;
};

class hf3fsUtil {
public:
    hf3fsUtil() {}
    ~hf3fsUtil() {}
    nixlStatus registerFileHandle(int fd, int *ret);
    void deregisterFileHandle(int fd);
    nixlStatus openHf3fsDriver();
    void closeHf3fsDriver();
    nixlStatus createIOR(struct hf3fs_ior *ior, int num_ios, bool is_read);
    nixlStatus createIOV(struct hf3fs_iov *iov, void *addr, size_t size, size_t block_size);
    nixlStatus wrapIOV(struct hf3fs_iov *iov, void *addr, size_t size, size_t block_size);
    void destroyIOV(struct hf3fs_iov *iov);
    nixlStatus destroyIOR(struct hf3fs_ior *ior);
    nixlStatus prepIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr,
                         size_t fd_offset, size_t size, int fd, bool is_read, void *user_data);
    nixlStatus postIOR(struct hf3fs_ior *ior);
    nixlStatus waitForIOs(struct hf3fs_ior *ior, struct hf3fs_cqe *cqes, int num_cqes,
                             int min_cqes, struct timespec *ts, int *num_completed);
    std::string mount_point;
};

#endif
