import torch
from torch.nn import MaxPool2d
input = torch.rand([30,50], dtype=torch.float32)
print(input.shape)
input = torch.reshape(input, ( -1, 30, 50))
print(input.shape)
maxpool1 = MaxPool2d(kernel_size=5)
output = maxpool1(input)
print(output.shape)
