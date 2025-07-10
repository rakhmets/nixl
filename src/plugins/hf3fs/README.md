# NIXL HF3FS Plugin

This plugin utilizes `hf3fs_usrbio.so` as an I/O backend for NIXL.

## Usage
1. Build and install [3FS](https://github.com/deepseek-ai/3FS/).
2. Ensure that 3FS libraries,`hf3fs_usrbio.so` and `libhf3fs_api_shared.so`, are installed under `/usr/lib/`.
3. Ensure that 3FS headers are installed under `/usr/include/hf3fs`.
4. Build NIXL.
5. Once the HF3FS Backend is built, you can use it in your data transfer task by specifying the backend name as "HF3FS":

```cpp
nixlStatus ret1;
std::string ret_s1;
nixlAgentConfig cfg(true);
nixlBParams init1;
nixlMemList mems1;
nixlBackendH      *hf3fs;
nixlAgent A1(agent1, cfg);
ret1 = A1.getPluginParams("HF3FS", mems1, init1);
assert (ret1 == NIXL_SUCCESS);
ret1 = A1.createBackend("HF3FS", init1, hf3fs);
...
```

