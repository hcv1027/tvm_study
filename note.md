# Compile `libtvm.so` and `libtvm_runtime.so`

## 1. Compile 

### Modify CMakeLists.txt
```cmake
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
```

### Modify cmake/modules/contrib/ArmComputeLib.cmake
```cmake
# Modify this line
set(ACL_INCLUDE_DIRS ${ACL_PATH}/include ${ACL_PATH})
# To this line 
set(ACL_INCLUDE_DIRS ${ACL_PATH}/include)
```

## 2. Prepare Docker Container

### Qualcomm SA8295
```bash
docker build --no-cache -t tvm:v3 -f Dockerfile.cross_qualcomm .
docker run -it --name tvm_env --volume="/home/josper/workspace/:/workspace/" tvm:v3 /bin/bash
```

### X86 OpenCL with NV's GPU
```bash
docker build --no-cache -t tvm_x86_cl:v1 -f Dockerfile.x86_nv_cl .
docker run --gpus all -it --name tvm_x86_cl -p 9191:9190 --volume="/home/josper/workspace/:/workspace/" tvm_x86_cl:v1 /bin/bash
```

## 3. Install TVM Package in Docker Container

```bash
# The following three commands can be done in Dockerfile and skipped in docker container
# /opt/conda/bin/conda init bash
# /opt/conda/bin/conda init zsh
# source ~/.bashrc

conda env create --file conda/build-environment.yaml
conda activate tvm-build
pip3 install numpy decorator attrs typing-extensions psutil scipy tornado 'xgboost>=1.1.0' cloudpickle onnx onnxoptimizer

mv /opt/conda/envs/tvm-build/lib/libstdc++.so.6 /opt/conda/envs/tvm-build/lib/libstdc++.so.6.bak
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/envs/tvm-build/lib/libstdc++.so.6

conda build --output-folder=conda/pkg conda/recipe
conda install tvm -c ./conda/pkg
```

## 4. Build `libtvm.so` and `libtvm_runtime.so`
### Cross compile case:
Build `libtvm.so` for host machine:
```bash 
# !/bin/bash
cd /home/josper/workspace/tvm/
mkdir build_host
cp cmake/SA8295_host.cmake build_host/config.cmake
cd build_host
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/home/josper/workspace/tvm/tvm_host

make -j8 # Build both libtvm.so and libtvm_runtime.so (But we don't need libtvm_runtime.so)
```

Build `libtvm_runtime.so` for target device:
```bash 
# !/bin/bash
mkdir build_runtime
cd /home/josper/workspace/tvm/
cp cmake/SA8295_runtime.cmake build_runtime/config.cmake
cd build_runtime
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 \
    -DCMAKE_INSTALL_PREFIX=/home/josper/workspace/tvm/tvm_runtime

make runtime # Only build libtvm_runtime.so
```

## 5. Auto-tuning on SA8295

### Host Machine
```bash
export TVM_TRACKER_HOST=172.19.134.19
export TVM_TRACKER_PORT=9190
```

### Tracker Machine (Can be the same as Host Machine)
```bash
python3 -m tvm.exec.rpc_tracker --port 9190
```

### Target Device
```bash
adb reverse tcp:9190 tcp:9190
adb forward tcp:5000 tcp:5000
./tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=172.19.134.19:9190 --key=android --port-end=5100
```
