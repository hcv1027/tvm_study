import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import numpy as np
import onnx
import time

# Step 1: Load your ONNX model
onnx_model = onnx.load('g2210_b_4.onnx')

# Step 2: Prepare input data, shape, and data types
input_name = 'input'
input_shape = (4, 3, 640, 640)
input_dtype = 'uint8'

input_data = np.random.randint(
    low=0,
    high=256,
    size=input_shape,
    dtype=input_dtype
)

# Step 3: Convert the ONNX model to Relay IR
shape_dict = {input_name: input_shape}
dtype_dict = {input_name: input_dtype}
mod, params = relay.frontend.from_onnx(
    onnx_model, shape=shape_dict, dtype=dtype_dict
)

# Step 4: Define the compilation target
target = tvm.target.Target("cuda")

# Step 5: Compile and export the model before tuning
print("Compiling without tuning...")
with tvm.transform.PassContext(opt_level=3):
    lib_before = relay.build(
        mod,
        target=target,
        params=params
    )

# Export the un-tuned model
lib_before.export_library("g2210_b_4_lib_before.so")
with open("g2210_b_4_graph_before.json", "w") as f:
    f.write(lib_before.get_graph_json())
with open("g2210_b_4_param_before.params", "wb") as f:
    param_bytes = tvm.runtime.save_param_dict(lib_before.get_params())
    f.write(param_bytes)

# Measure performance before tuning
dev = tvm.cuda(0)
module_before = graph_executor.GraphModule(lib_before["default"](dev))
module_before.set_input(input_name, tvm.nd.array(input_data, device=dev))

# Warm up
module_before.run()

# Time the execution
print("Measuring performance before tuning...")
ftimer = module_before.module.time_evaluator("run", dev, number=10, repeat=3)
timing_result = ftimer().results
mean_time_before = np.mean(timing_result) * 1000  # Convert to milliseconds
print(f"Mean inference time (before tuning): {mean_time_before:.2f} ms")

# Step 6: Perform AutoScheduler tuning
print("Starting AutoScheduler tuning...")

# Define the tuning tasks
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

# Define the tuning options
tuning_option = auto_scheduler.TuningOptions(
    num_measure_trials=3000,  # You can adjust this number based on time constraints
    early_stopping=1500,  # Stop if no improvement after 3000 trials
    runner=auto_scheduler.LocalRunner(repeat=5, min_repeat_ms=200, timeout=20),
    measure_callbacks=[auto_scheduler.RecordToFile('autoscheduler_tuning_log.json')],
    verbose=2,
)

# Create a task scheduler and tune
task_scheduler = auto_scheduler.TaskScheduler(tasks, task_weights)
task_scheduler.tune(tuning_option)

# Step 7: Compile and export the model after tuning
print("Compiling with tuning...")

with auto_scheduler.ApplyHistoryBest('autoscheduler_tuning_log.json'):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib_after = relay.build(
            mod,
            target=target,
            params=params
        )

# Export the tuned model
lib_after.export_library("g2210_b_4_lib_after.so")
with open("g2210_b_4_graph_after.json", "w") as f:
    f.write(lib_after.get_graph_json())
with open("g2210_b_4_param_after.params", "wb") as f:
    param_bytes = tvm.runtime.save_param_dict(lib_after.get_params())
    f.write(param_bytes)

# Measure performance after tuning
module_after = graph_executor.GraphModule(lib_after["default"](dev))
module_after.set_input(input_name, tvm.nd.array(input_data, device=dev))

# Warm up
module_after.run()

# Time the execution
print("Measuring performance after tuning...")
ftimer = module_after.module.time_evaluator("run", dev, number=10, repeat=3)
timing_result = ftimer().results
mean_time_after = np.mean(timing_result) * 1000  # Convert to milliseconds
print(f"Mean inference time (after tuning): {mean_time_after:.2f} ms")

# Step 8: Compare the performance
print(f"Performance improvement: {((mean_time_before - mean_time_after) / mean_time_before) * 100:.2f}%")

