import torch
from torch import nn

x = torch.tensor([[
    [[1., 2., 0., 1.],
     [3., 1., 2., 2.],
     [0., 1., 3., 1.],
     [2., 2., 1., 0.]]
]])  # [1, 1, 4, 4]
w = torch.tensor([[[[1., 0.], [0., -1.]]]])

def aplicar(s, p=0):
    camada = nn.Conv2d(1, 1, 2, stride=s, padding=p, bias=False)
    with torch.no_grad():
        camada.weight.copy_(w)
        return camada(x)

y1, y2, y_pad = aplicar(1), aplicar(2), aplicar(1, 1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)(x)

assert y1.shape[-2:] == (3, 3) and y_pad.shape[-2:] == (5, 5)
assert y1[0, 0].tolist() == [[0., 0., -2.], [2., -2., 1.], [-2., 0., 3.]]
assert y2[0, 0].tolist() == [[0., -2.], [-2., 3.]]
assert pool[0, 0].tolist() == [[3., 2.], [2., 3.]]

print("Conv stride 2:", y2[0, 0])
print("MaxPool:", pool[0, 0])
