import tvm
from tvm import runtime
from tvm.contrib import graph_executor
import numpy as np
import time

def load_model(lib_path, graph_json_path, params_path, dev):
    """
    Load a compiled TVM model.

    Parameters:
        lib_path (str): Path to the compiled shared library (.so file).
        graph_json_path (str): Path to the graph JSON file.
        params_path (str): Path to the parameters file.
        dev (tvm.device): TVM device where the model will run.

    Returns:
        graph_executor.GraphModule: The loaded TVM module.
    """
    # Load the compiled library
    lib = runtime.load_module(lib_path)

    # Load the graph
    with open(graph_json_path, "r") as f:
        graph_json = f.read()

    # Load the parameters
    with open(params_path, "rb") as f:
        params = f.read()

    # Create the graph executor
    module = graph_executor.create(graph_json, lib, dev)

    # Load the parameters into the module
    module.load_params(params)

    return module

def measure_inference_time(module, input_name, input_data, dev, num_runs=1000):
    """
    Measure the mean inference time of a TVM module over a specified number of runs.

    Parameters:
        module (graph_executor.GraphModule): The TVM module to benchmark.
        input_name (str): The name of the input tensor.
        input_data (numpy.ndarray): The input data to feed into the model.
        dev (tvm.device): TVM device where the model runs.
        num_runs (int): Number of inference runs to perform.

    Returns:
        float: Mean inference time in milliseconds.
    """
    # Set the input
    module.set_input(input_name, tvm.nd.array(input_data, device=dev))

    # Warm up the model to ensure accurate timing
    for _ in range(10):
        module.run()

    # Start timing
    start_time = time.perf_counter()

    for _ in range(num_runs):
        module.run()

    end_time = time.perf_counter()

    # Calculate mean time in milliseconds
    total_time = end_time - start_time
    mean_time = (total_time / num_runs) * 1000  # Convert to ms

    return mean_time

def main():
    # Configuration
    input_name = "input"
    input_shape = (4, 3, 640, 640)  # Batch size of 4
    input_dtype = "uint8"

    # Generate random input data
    input_data = np.random.randint(
        low=0,
        high=256,
        size=input_shape,
        dtype=input_dtype
    )

    # Set the target device (CUDA)
    dev = tvm.device("cuda", 0)

    # Paths to the compiled models
    before_lib_path = "g2210_b_4_lib_before.so"
    before_graph_json_path = "g2210_b_4_graph_before.json"
    before_params_path = "g2210_b_4_param_before.params"

    after_lib_path = "g2210_b_4_lib_after.so"
    after_graph_json_path = "g2210_b_4_graph_after.json"
    after_params_path = "g2210_b_4_param_after.params"

    # Load models
    print("Loading models...")
    module_before = load_model(before_lib_path, before_graph_json_path, before_params_path, dev)
    module_after = load_model(after_lib_path, after_graph_json_path, after_params_path, dev)
    print("Models loaded successfully.\n")

    # Measure inference time before tuning
    print("Measuring performance before tuning...")
    mean_time_before = measure_inference_time(
        module_before,
        input_name,
        input_data,
        dev,
        num_runs=1000
    )
    print(f"Mean inference time (before tuning): {mean_time_before:.2f} ms\n")

    # Measure inference time after tuning
    print("Measuring performance after tuning...")
    mean_time_after = measure_inference_time(
        module_after,
        input_name,
        input_data,
        dev,
        num_runs=1000
    )
    print(f"Mean inference time (after tuning): {mean_time_after:.2f} ms\n")

    # Calculate performance improvement
    improvement = ((mean_time_before - mean_time_after) / mean_time_before) * 100
    print(f"Performance improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()

