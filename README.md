# OpenCL Conv1d PyTorch Extension

A custom 1D Convolution (`Conv1d`) operator for PyTorch, written from scratch using C++ and OpenCL.

## Features
*   **Custom OpenCL Kernels**: C++ and OpenCL implementations for both forward and backward passes.
*   **Full Autograd Support**: Integrates seamlessly with PyTorch's autograd engine (`loss.backward()`) for training.

## Prerequisites
*   Python 3.8+
*   PyTorch (`torch`)
*   OpenCL Headers and Library (`libOpenCL.so`)
*   C++17 compatible compiler (GCC/Clang)

## Installation

```bash
# Clone the repository
git clone https://github.com/tht2005/clAtenOps.git
cd clAtenOps

# Install the extension
pip install .
```

## Quick Start

```python
import torch
import clAtenOps

# 1. Initialize the custom OpenCL layer
layer = clAtenOps.Conv1d(
    in_ch=3, 
    out_ch=5, 
    k_size=16, 
    stride=1, 
    padding=0, 
    dilation=1
)

# 2. Create a dummy input with requires_grad=True
x = torch.randn(16, 3, 56, requires_grad=True)

# 3. Forward Pass
y = layer(x)
print(f"Output shape: {y.shape}")

# 4. Backward Pass
loss = y.sum()
loss.backward()

print(f"Weight gradient shape: {layer.weight.grad.shape}")
print(f"Input gradient shape: {x.grad.shape}")
```
