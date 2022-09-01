import torch
import torch.nn as nn

m = nn.Sigmoid()

loss = nn.BCELoss(size_average=False, reduce=False)
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
lossinput = m(input)
output = loss(lossinput, target)
input = input.detach().numpy()
print(output)