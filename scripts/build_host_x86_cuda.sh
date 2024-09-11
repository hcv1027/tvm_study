# !/bin/bash

cd /workspace/tvm/
if [ ! -d "build_host_x86" ]; then
    mkdir build_host_x86
fi
rm -rf build_host_x86/*
cp cmake/config.cmake build_host_x86/
cd build_host_x86/

# Enable OpenCL
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_NVTX ON\) >> config.cmake
echo set\(USE_GRAPH_EXECUTOR_CUDA_GRAPH ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_CUDNN_FRONTEND OFF\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_CURAND ON\) >> config.cmake
# Enable OpenCL
echo set\(USE_OPENCL ON\) >> config.cmake
# Enable RPC capability to communicate to remote device.
echo set\(USE_RPC ON\) >> config.cmake
# Enable llvm
echo set\(USE_LLVM ON\) >> config.cmake
echo set\(BACKTRACE_ON_SEGFAULT ON\) >> config.cmake
echo set\(SUMMARIZE ON\) >> config.cmake

cmake .. \
    -DCMAKE_INSTALL_PREFIX=/workspace/tvm/tvm_host_x86/ \
    -DCMAKE_BUILD_TYPE=Release
make -j24
make install


