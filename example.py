import torch
import torch.nn as nn

import clAtenOps

x = torch.randn(16, 8, 52, requires_grad=True);
layer = clAtenOps.Conv1d(in_ch=8, out_ch=16, k_size=5, stride=1, padding=0, dilation=1)
y = layer(x)

loss = y.pow(2).sum()
loss.backward()

print(x.grad)
