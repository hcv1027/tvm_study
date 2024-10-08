# coding=utf-8
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

# Construct Batch Normalization (BN)
def batch_norm(data, gamma=None, beta=None, moving_mean=None, moving_var=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                               gamma=gamma,
                               beta=beta,
                               moving_mean=moving_mean,
                               moving_var=moving_var,
                               **kwargs)[0]

# Construct Convolution
def conv2d(data, weight=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)

# Construct Conv + BN + ReLU as simpleNet
def simplenet(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
              padding=(1, 1), epsilon=1e-5):
    conv = conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout='NCHW',
        name=name + '_conv'
    )
    bn = batch_norm(data=conv, epsilon=epsilon, name=name + '_bn')
    act = relay.nn.relu(data=bn)
    return act

# Define input data shape and kernel shape
data_shape = (1, 3, 224, 224)
kernel_shape = (32, 3, 3, 3)
dtype = "float32"
data = relay.var("data", shape=data_shape, dtype=dtype)

# Build the simpleNet
act = simplenet(data, "graph", 32, strides=(2, 2))
func = relay.Function(relay.analysis.free_vars(act), act)

# Print the function
print("SimpleNet function:")
print(func)

# Randomize input data
np_data = np.random.uniform(-1, 1, (1, 3, 224, 224))

# Define parameters
params = {
    "graph_conv_weight": tvm.nd.array(np.random.uniform(-1, 1, (32, 3, 3, 3)).astype(dtype)),
    "graph_bn_gamma": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_beta": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_moving_mean": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_moving_var": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
}

# Build with optimization level 3
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, "llvm", params=params)

# Set up the device and module for execution
dev = tvm.cpu(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))

# Set inputs
m.set_input("data", tvm.nd.array(np_data.astype(dtype)))

# Execute the model
m.run()

# Get outputs
tvm_output = m.get_output(0)

