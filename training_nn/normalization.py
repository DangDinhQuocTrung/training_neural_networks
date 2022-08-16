import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.register_buffer("running_mean", torch.ones(num_features))
        self.register_buffer("running_var", torch.zeros(num_features))

    def forward(self, inputs):
        if self.training:
            mean = inputs.mean([0, 2, 3])
            var = inputs.var([0, 2, 3], unbiased=True)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1.0 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inputs = (inputs - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        inputs = inputs * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return inputs


class InstanceNorm(nn.Module):
    def __init__(self, h, w, momentum=0.9, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.h = h
        self.w = w
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.ones(h, w))
        self.register_buffer("running_var", torch.zeros(h, w))

    def forward(self, inputs):
        if self.training:
            mean = inputs.mean([0, 1])
            var = inputs.var([0, 1], unbiased=True)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1.0 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inputs = (inputs - mean[None, None, :, :]) / (torch.sqrt(var[None, None, :, :] + self.eps))
        return inputs


class LayerNorm(nn.Module):
    def __init__(self, h, w, momentum=0.1, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.h = h
        self.w = w
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(h, w))
        self.bias = nn.Parameter(torch.zeros(h, w))

    def forward(self, inputs):
        mean = inputs.mean([0])
        var = inputs.var([0], unbiased=True)

        inputs = (inputs - mean[None, :, :, :]) / (torch.sqrt(var[None, :, :, :] + self.eps))
        inputs = inputs * self.weight[None, None, :, :] + self.bias[None, None, :, :]
        return inputs


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, momentum=0.1, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], self.num_groups, -1, inputs.shape[2], inputs.shape[3])
        mean = inputs.mean([0, 1])
        var = inputs.var([0, 1], unbiased=True)

        inputs = (inputs - mean[None, None, :, :, :]) / (torch.sqrt(var[None, None, :, :, :] + self.eps))
        inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[3], inputs.shape[4])
        inputs = inputs * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return inputs


def main():
    x = torch.rand(1, 64, 128, 128)
    layer = GroupNorm(2, 64)
    out = layer(x)


if __name__ == "__main__":
    main()
