import torch
import torch.nn as nn


class SkipConn(nn.Module):
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, linmap=None):
        super(SkipConn, self).__init__()
        out_size = hidden_size

        self.inLayer = nn.Linear(init_size, out_size)
        self.relu = nn.LeakyReLU()
        hidden = []
        for i in range(num_hidden_layers):
            in_size = out_size*2 + init_size if i > 0 else out_size + init_size
            hidden.append(nn.Linear(in_size, out_size))
        self.hidden = nn.ModuleList(hidden)
        self.outLayer = nn.Linear(out_size*2+init_size, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self._linmap = linmap

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)
        cur = self.relu(self.inLayer(x))
        prev = torch.tensor([]).cuda()
        for layer in self.hidden:
            combined = torch.cat([cur, prev, x], 1)
            prev = cur
            cur = self.relu(layer(combined))
        y = self.outLayer(torch.cat([cur, prev, x], 1))
        return (self.tanh(y)+1)/2


class Fourier(nn.Module):
    def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
        super(Fourier, self).__init__()
        self.fourier_order = fourier_order
        self.inner_model = SkipConn(
            hidden_size, num_hidden_layers, fourier_order*4 + 2)
        self._linmap = linmap
        self.orders = torch.arange(1, fourier_order + 1).float().to('cuda')

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)
        x = x.unsqueeze(-1)
        fourier_features = torch.cat(
            [torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1)
        fourier_features = fourier_features.view(
            x.shape[0], -1)
        return self.inner_model(fourier_features)


class CenteredLinearMap():
    def __init__(self, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1, x_size=None, y_size=None):
        if x_size is not None:
            x_m = x_size/(xmax - xmin)
        else:
            x_m = 1.
        if y_size is not None:
            y_m = y_size/(ymax - ymin)
        else:
            y_m = 1.
        x_b = -(xmin + xmax)*x_m/2 - 1
        y_b = -(ymin + ymax)*y_m/2
        self.m = torch.tensor([x_m, y_m], dtype=torch.float)
        self.b = torch.tensor([x_b, y_b], dtype=torch.float)

    def map(self, x):
        m = self.m.cuda()
        b = self.b.cuda()
        return m*x + b
