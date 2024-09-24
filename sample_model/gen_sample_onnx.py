import torch
import torch.nn as nn
import torch.onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)  # First layer
        self.fc2 = nn.Linear(3, 2)  # Second layer

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # Apply ReLU activation
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleModel()

# Dummy input for tracing (batch size of 1)
dummy_input = torch.randn(1, 4)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "simple_model.onnx", 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

