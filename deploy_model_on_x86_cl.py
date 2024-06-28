# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
import os
import tvm
from tvm import te
from tvm import relay, rpc
from tvm.contrib import utils, ndk
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import clml
from tvm import autotvm
import onnx

# Configuration
calculation_dtype = "float16"
acc_dtype = "float32"

# Set to True if you want to run on a real device over RPC, else False for local run
local_demo = False

# Set the target to OpenCL
test_target = "opencl"
# Set the target to intel x86 CPU
# test_target = "llvm -mcpu=core-avx2"

# Change target configuration for x86-64 architecture
target = tvm.target.Target("llvm")
target_host = tvm.target.Target("llvm")

# Set to True to enable auto-tuning
is_tuning = False
is_tuning = True
tune_log = "/workspace/gallopwave/tvm/example/auto_tune_x86_cl.json"

# Path to the ONNX model
onnx_model_path = "/workspace/gallopwave/tvm/models/g2208_b_4_output_concat.onnx"
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to Relay module
input_name = "input"
shape_dict = {input_name: (4, 3, 640, 640)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Apply mixed precision transformations
from tvm.driver.tvmc.transform import apply_graph_transforms
# mod = apply_graph_transforms(
#     mod,
#     {
#         "mixed_precision": True,
#         "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
#         "mixed_precision_calculation_type": calculation_dtype,
#         "mixed_precision_acc_type": acc_dtype,
#     },
# )

# Prepare TVM target
if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.Target(test_target, host=target_host)

print(f"Compilation target: {target}")

# AutoTuning
rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "localhost")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
# Target is opencl run on NVIDIA GPU
key = "opencl"
# key = "x86_opencl"

if is_tuning:
    tasks = autotvm.task.extract_from_program(
        mod, target=target, target_host=target_host, params=params
    )
    tmp_log_file = tune_log + ".tmp"

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default", timeout=15),  # 修改點2: 使用 LocalBuilder 而不是 ndk.create_shared
        runner=autotvm.LocalRunner(number=10, repeat=1, timeout=4, min_repeat_ms=150)  # 修改點3: 使用 LocalRunner 進行本地調優
    )

    # measure_option = autotvm.measure_option(
    #     builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
    #     runner=autotvm.RPCRunner(
    #         key,
    #         host=rpc_tracker_host,
    #         port=rpc_tracker_port,
    #         number=3,
    #         timeout=600,
    #     ),
    # )
    n_trial = 1024
    early_stopping = False

    from tvm.autotvm.tuner import XGBTuner

    for i, tsk in enumerate(reversed(tasks[:3])):
        print("Task:", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type="reg")

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    autotvm.record.pick_best(tmp_log_file, tune_log)

# Compilation
print("Compiling the model...")
if os.path.exists(tune_log):
    print(f"Using tuning log: {tune_log}")
    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
else:
    print("No tuning log found. Compiling without tuning...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
print("Compilation done...")

# Save the compiled model
model_dir = "/workspace/gallopwave/tvm/example/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

lib.export_library(os.path.join(model_dir, "compiled_model_x86.so"))
with open(os.path.join(model_dir, "compiled_model_x86.json"), "w") as f:
    f.write(lib.get_graph_json())
param_bytes = tvm.runtime.save_param_dict(lib.get_params())
with open(os.path.join(model_dir, "compiled_model_x86.params"), "wb") as f:
    f.write(param_bytes)

print(f"Model artifacts saved to {model_dir}")

