Certainly! Understanding the roles of TVM's `rpc_tracker` and `tvm_rpc` is crucial for distributed tuning and deployment scenarios. Let's break down their functions and see how they can work together in your setup.

### Roles of `rpc_tracker` and `tvm_rpc`

#### `rpc_tracker`

- **Role**: The `rpc_tracker` is a centralized service that helps manage and keep track of remote devices (e.g., edge devices, remote servers) that can execute TVM tasks.
- **Function**: It maintains a registry of available devices, their capabilities, and their network addresses. This allows TVM to distribute tasks to the appropriate remote devices.
- **Usage**: You run the `rpc_tracker` on a central machine that will coordinate communication between your host machine and remote devices.

#### `tvm_rpc`

- **Role**: The `tvm_rpc` is an RPC (Remote Procedure Call) server that runs on the target device (e.g., Android edge device). It receives commands from the host machine (through the `rpc_tracker`) to execute specific tasks like model inference or tuning.
- **Function**: It listens for incoming requests from the `rpc_tracker` and executes the tasks on the target device, returning the results to the host machine.
- **Usage**: You run the `tvm_rpc` on each target device that you want to use for model execution or tuning.

### Example Setup

In your scenario, you want to run the `rpc_tracker` on `host_b` (with low CPU) and perform auto-tuning on `host_a` (with high CPU), with the target device being an Android edge device.

### Steps to Set Up

1. **Run `rpc_tracker` on Host B**:

   ```sh
   host_b$ python3 -m tvm.exec.rpc_tracker --port 9190
   ```

   This command starts the `rpc_tracker` on `host_b` listening on port 9190.

2. **Run `tvm_rpc` on the Android Edge Device**:

   ```sh
   adb reverse tcp:9190 tcp:9190
   adb forward tcp:5000 tcp:5000
   adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=`host_b_ip_address`:9190 --key=android --port-end=5100
   ```

   This sets up port forwarding and starts the `tvm_rpc` server on the Android device, registering it with the `rpc_tracker` on `host_b`.

3. **Perform Auto-Tuning on Host A**:

   On `host_a`, set the environment variables to point to the `rpc_tracker` running on `host_b`, and then run the auto-tuning script:

   ```sh
   export TVM_TRACKER_HOST=host_b_ip_address
   export TVM_TRACKER_PORT=9190

   python3 -m tvm.driver.tvmc tune \
       --target="opencl -device=adreno" \
       --target-host="llvm -mtriple=aarch64-linux-gnu" \
       --rpc-key android \
       --rpc-tracker $TVM_TRACKER_HOST:$TVM_TRACKER_PORT \
       /path/to/your/model.onnx --output /path/to/tuning_results.log
   ```

   Replace `host_b_ip_address` with the actual IP address of `host_b`.

### Summary

- **`rpc_tracker` on Host B**: Manages and tracks the remote devices available for task execution.
- **`tvm_rpc` on Android Edge Device**: Executes tasks sent by the host through the `rpc_tracker`.
- **Auto-Tuning on Host A**: Uses `host_b` as the `rpc_tracker` to find and communicate with the Android device for tuning tasks.

### Benefits

- **Load Distribution**: Running the `rpc_tracker` on a low-CPU host (`host_b`) offloads the tracking tasks, allowing the high-CPU host (`host_a`) to focus on the compute-intensive tuning tasks.
- **Scalability**: This setup allows you to add more devices or hosts as needed, managed by the central `rpc_tracker`.

This configuration helps in effectively utilizing different resources and distributing the workload appropriately. If you have any specific questions or need further customization, feel free to ask!

---

### Q: I see a tutorial which says to run these command:
```
adb reverse tcp:9190 tcp:9190
adb forward tcp:5000 tcp:5000
adb forward tcp:5002 tcp:5001
adb forward tcp:5003 tcp:5002
adb forward tcp:5004 tcp:5003
```
It run three more `adb forward` than your version, what's different?

### Ans:
The additional `adb forward` commands in the tutorial are used to set up port forwarding for multiple ports between the host machine and the Android device. This is often done to handle multiple RPC sessions or multiple parallel tuning tasks. Let's break down the purpose and differences:

### Purpose of Each Command

1. **Reverse Port Forwarding**:
   - `adb reverse tcp:9190 tcp:9190`
     - This command sets up reverse port forwarding so that the Android device can connect to the `rpc_tracker` running on the host machine on port 9190.

2. **Forward Port Forwarding**:
   - `adb forward tcp:5000 tcp:5000`
   - `adb forward tcp:5002 tcp:5001`
   - `adb forward tcp:5003 tcp:5002`
   - `adb forward tcp:5004 tcp:5003`

   These commands forward multiple ports from the host machine to the Android device. This is useful for handling multiple RPC server instances or sessions. Each forwarded port pair (e.g., `5000:5000`, `5002:5001`, etc.) allows a separate RPC server instance on the Android device to communicate with the host.

### Comparison

- **Simpler Setup (Single Port)**:
  - The simpler setup with just one `adb forward` command (e.g., `adb forward tcp:5000 tcp:5000`) is sufficient if you only need one RPC session or if you're only running a single tuning task at a time.

- **Multiple Ports Setup**:
  - The additional `adb forward` commands in the tutorial allow for multiple concurrent RPC sessions. This can be useful if you plan to run multiple tuning tasks in parallel or if the RPC server uses multiple ports for different purposes.

### Example Scenario

- **Single Port Forwarding**:
  ```sh
  adb reverse tcp:9190 tcp:9190
  adb forward tcp:5000 tcp:5000
  ```

  - Suitable for a single RPC session or a single tuning task.

- **Multiple Ports Forwarding**:
  ```sh
  adb reverse tcp:9190 tcp:9190
  adb forward tcp:5000 tcp:5000
  adb forward tcp:5002 tcp:5001
  adb forward tcp:5003 tcp:5002
  adb forward tcp:5004 tcp:5003
  ```

  - Suitable for multiple RPC sessions or multiple parallel tuning tasks.

### How to Choose

- **Use the simpler setup** if:
  - You are running a single tuning task.
  - You do not require multiple concurrent RPC sessions.

- **Use the extended setup** if:
  - You plan to run multiple tuning tasks in parallel.
  - You have an advanced setup that requires multiple RPC sessions.

### Adjusted Command for Multiple Port Forwarding

If you decide to use the multiple port forwarding setup, here’s how you can adjust the `tvm_rpc` server command on the Android device:

```sh
adb reverse tcp:9190 tcp:9190
adb forward tcp:5000 tcp:5000
adb forward tcp:5002 tcp:5001
adb forward tcp:5003 tcp:5002
adb forward tcp:5004 tcp:5003

adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=host_b_ip_address:9190 --key=android --port-end=5100
```

In the above command, replace `host_b_ip_address` with the actual IP address of `host_b`.

### Running Auto-Tuning on Host A

```sh
export TVM_TRACKER_HOST=host_b_ip_address
export TVM_TRACKER_PORT=9190

python3 -m tvm.driver.tvmc tune \
    --target="opencl -device=adreno" \
    --target-host="llvm -mtriple=aarch64-linux-gnu" \
    --rpc-key android \
    --rpc-tracker $TVM_TRACKER_HOST:$TVM_TRACKER_PORT \
    /path/to/your/model.onnx --output /path/to/tuning_results.log
```

By setting up multiple port forwarding, you can handle multiple RPC sessions, which can be useful for more complex tuning and deployment scenarios.