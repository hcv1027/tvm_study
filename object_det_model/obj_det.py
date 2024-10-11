import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import onnx

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
target_host = tvm.target.Target("llvm")

# Step 5: Build the optimized module
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(
        mod,
        target=target,
        target_host=target_host,
        params=params
    )

# Step 6: (Optional) Execute the model
dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, tvm.nd.array(input_data, device=dev))
module.run()
output = module.get_output(0).asnumpy()
print("Model output shape:", output.shape)

# Step 7: Export the compiled module
lib.export_library("deploy_lib.so")
with open("deploy_graph.json", "w") as f:
    f.write(lib.get_graph_json())
with open("deploy_param.params", "wb") as f:
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    f.write(param_bytes)

