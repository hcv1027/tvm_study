import onnxruntime as ort
import numpy as np
import time


def load_onnx_model(onnx_path):
    """
    Load an ONNX model using onnxruntime with GPU (CUDAExecutionProvider).
    
    Parameters:
        onnx_path (str): Path to the ONNX model file.
    
    Returns:
        onnxruntime.InferenceSession: The loaded ONNX runtime session.
    """
    # Use GPU
    providers = ["CUDAExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def measure_inference_time(session, input_name, input_data, num_runs=1000):
    """
    Measure the mean inference time of an ONNX model over a specified number of runs.

    Parameters:
        session (onnxruntime.InferenceSession): The ONNX runtime session.
        input_name (str): The name of the input tensor in the ONNX model.
        input_data (numpy.ndarray): The input data to feed into the model.
        num_runs (int): Number of inference runs to perform.

    Returns:
        float: Mean inference time in milliseconds.
    """
    # Warm up the model to ensure accurate timing (e.g., GPU context initialization)
    for _ in range(10):
        session.run(None, {input_name: input_data})

    # Start timing
    start_time = time.perf_counter()

    for _ in range(num_runs):
        session.run(None, {input_name: input_data})

    end_time = time.perf_counter()

    # Calculate mean time in milliseconds
    total_time = end_time - start_time
    mean_time = (total_time / num_runs) * 1000  # Convert to ms

    return mean_time


def main():
    # Configuration (matching your original shapes and dtype)
    onnx_model_path = "g2210_b_4.onnx"
    input_name = "input"  # Adjust if your ONNX input layer has a different name
    input_shape = (4, 3, 640, 640)  # Batch size of 4
    input_dtype = "uint8"
    num_runs = 1000

    # Generate random input data
    input_data = np.random.randint(
        low=0,
        high=256,
        size=input_shape,
        dtype=input_dtype
    )

    # Load the ONNX model
    print("Loading ONNX model...")
    session = load_onnx_model(onnx_model_path)
    print("ONNX model loaded successfully.\n")

    # Measure inference time
    print("Measuring performance with onnxruntime-gpu...")
    mean_time = measure_inference_time(session, input_name, input_data, num_runs=num_runs)
    print(f"Mean inference time: {mean_time:.2f} ms over {num_runs} runs\n")


if __name__ == "__main__":
    main()

