import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.relu = nn.ReLU()

    def disable_layer_grad(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def enable_layer_grad(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def forward(self, x, disable_grad_x2=True):

        # process x1 in normal mode
        x1 = x[:1]
        x1 = self.fc1(x1)

        # process x2 with with frozen layers
        if x.shape[0] > 1:
            x2 = x[1:]
            if disable_grad_x2:
                self.disable_layer_grad(self.fc1)
            x2 = self.fc1(x2)
            if disable_grad_x2:
                self.enable_layer_grad(self.fc1)

            # concat
            x = torch.cat((x1, x2), 0)
        else:
            x = x1

        x = self.relu(x)
        x = self.fc2(x)
        return x


net = SimpleNet()
z = torch.randn([2, 100])

net.zero_grad()
z_out_a = net(z, disable_grad_x2=True)
loss_a = z_out_a.sum()
loss_a.backward()
grad1_a = net.fc1.weight.grad
grad2_a = net.fc2.weight.grad

net.zero_grad()
z_out_b = net(z[:1], disable_grad_x2=True)
loss_b = z_out_b.sum()
loss_b.backward()
grad1_b = net.fc1.weight.grad
grad2_b = net.fc2.weight.grad

net.zero_grad()
z_out_c = net(z[1:], disable_grad_x2=True)
loss_c = z_out_c.sum()
loss_c.backward()
grad1_c = net.fc1.weight.grad
grad2_c = net.fc2.weight.grad

net.zero_grad()
z_out_d = net(z, disable_grad_x2=False)
loss_d = z_out_d.sum()
loss_d.backward()
grad1_d = net.fc1.weight.grad
grad2_d = net.fc2.weight.grad

import pdb
pdb.set_trace()
print((grad1_a - grad1_b).abs().max())
print((grad2_a - grad2_d).abs().max())
print(((grad2_b + grad2_c) - grad2_d).abs().max())
import pdb
pdb.set_trace()
