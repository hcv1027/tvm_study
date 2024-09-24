import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm import te

# simple_model.onnx information:
# input name: "input"
# input shape: float32[batch_size, 4]
# output name: "output"
# output shape: float32[batch_size, 2]
onnx_model = onnx.load("simple_model.onnx")
input_name = "input"
shape_dict = {input_name: (1, 4)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print("mod:\n", mod)
print("params:\n", params)

# Set target1 to be "llvm" for CPU
target1 = tvm.target.Target("llvm", host="llvm")

# Set target2 to be "cuda" for GPU
target2 = tvm.target.Target("cuda", host="llvm")

# Set target_host to be "llvm" for CPU
target_host = tvm.target.Target("llvm")

# Set target_host to be "cuda" for GPU
# target_host = tvm.target.Target("cuda")

dev_cpu = tvm.cpu(0)
dev_cuda = tvm.cuda(0)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target2, target_host=target_host, params=params)
