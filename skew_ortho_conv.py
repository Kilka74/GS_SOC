from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import einops


def power_iteration(W, u=None, v=None, num_iters=1):
    if u is None:
        u = torch.randn(W.shape[0], W.shape[2], 1, device='cuda', requires_grad=False)
        u.data = F.normalize(u.data, dim=0)

        v = torch.randn(W.shape[0], W.shape[1], 1, device='cuda', requires_grad=False)
        v.data = F.normalize(v.data, dim=0)
    for i in range(num_iters):
        v.data = F.normalize(torch.matmul(W.data, u.data), dim=0)
        u.data = F.normalize(torch.matmul(torch.transpose(W.data, 1, 2), v.data), dim=0)
    return u, v


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 1, 2)
    conv_filter_T = torch.flip(conv_filter_T, [3, 4])
    return conv_filter_T


class ConvFilterNorm(nn.Module):
    def __init__(self, conv_filter):
        super(ConvFilterNorm, self).__init__()
        self.u1, self.u2, self.u3, self.u4 = None, None, None, None
        self.v1, self.v2, self.v3, self.v4 = None, None, None, None
        # self.name = name
        # self.num_iters = num_iters
        # self.init_filter = conv_filter.clone().detach()
        conv = conv_filter.clone().detach()

        with torch.no_grad():
            matrix1, matrix2, matrix3, matrix4 = self._conv_matrices(conv)

            u1, v1 = power_iteration(matrix1)
            self.u1 = u1
            self.v1 = v1

            u2, v2 = power_iteration(matrix2)
            self.u2 = u2
            self.v2 = v2

            u3, v3 = power_iteration(matrix3)
            self.u3 = u3
            self.v3 = v3

            u4, v4 = power_iteration(matrix4)
            self.u4 = u4
            self.v4 = v4

    def _conv_matrices(self, conv_filter):
        groups, out_ch, in_ch, h, w = conv_filter.shape
        
        transpose1 = torch.transpose(conv_filter, 2, 3)
        matrix1 = transpose1.reshape(groups, out_ch*h, in_ch*w)

        transpose2 = torch.transpose(conv_filter, 2, 4)
        matrix2 = transpose2.reshape(groups, out_ch*w, in_ch*h)

        matrix3 = conv_filter.view(groups, out_ch, in_ch*h*w)

        transpose4 = torch.transpose(conv_filter, 1, 2)
        matrix4 = transpose4.reshape(groups, in_ch, out_ch*h*w)

        return matrix1, matrix2, matrix3, matrix4

    @torch.no_grad()
    def forward(self, conv_filter, num_iters):
        # conv_filter = self.conv_filter
        _, _, _, h, w = conv_filter.shape
        
        matrix1, matrix2, matrix3, matrix4 = self._conv_matrices(conv_filter)
        
        with torch.no_grad():
            self.u1.data, self.v1.data = power_iteration(matrix1.data, self.u1, self.v1, num_iters)
            self.u2.data, self.v2.data = power_iteration(matrix2.data, self.u2, self.v2, num_iters)
            self.u3.data, self.v3.data = power_iteration(matrix3.data, self.u3, self.v3, num_iters)
            self.u4.data, self.v4.data = power_iteration(matrix4.data, self.u4, self.v4, num_iters)
    
        sigma1 = torch.matmul(self.v1.transpose(1, 2), torch.matmul(matrix1, self.u1))
        sigma2 = torch.matmul(self.v2.transpose(1, 2), torch.matmul(matrix2, self.u2))
        sigma3 = torch.matmul(self.v3.transpose(1, 2), torch.matmul(matrix3, self.u3)) 
        sigma4 = torch.matmul(self.v4.transpose(1, 2), torch.matmul(matrix4, self.u4))
        
        sigma = torch.min(torch.min(torch.min(sigma1, sigma2), sigma3), sigma4) #  removed multiplication by sqrt(h*w)
        return sigma.view(-1, 1, 1, 1, 1)


class SOC_Function(Function):
    @staticmethod
    def forward(ctx, curr_z, conv_filter):
        ctx.conv_filter = conv_filter
        n_groups = conv_filter.shape[0] // conv_filter.shape[1]
        ctx.n_groups = n_groups
        kernel_size = conv_filter.shape[2]
        z = curr_z
        for i in range(1, 14):
            curr_z = F.conv2d(
                curr_z, conv_filter, padding=(kernel_size // 2, kernel_size // 2), groups=n_groups
            ) / float(i)
            z = z + curr_z
        return z

    @staticmethod
    def backward(ctx, grad_output):
        conv_filter = ctx.conv_filter
        n_groups = ctx.n_groups
        kernel_size = conv_filter.shape[2]
        grad_input = grad_output
        for i in range(1, 14):
            grad_output = F.conv2d(
                grad_output, -conv_filter, padding=(kernel_size // 2, kernel_size // 2), groups=n_groups
            ) / float(i)
            grad_input = grad_input + grad_output

        return grad_input, None


class SOC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=True,
        groups=1,
        train_terms=5,
        eval_terms=12,
        init_iters=50,
        update_iters=1,
        update_freq=200,
        correction=0.7,
        device="cuda",
        testing=False
    ):
        super(SOC, self).__init__()
        assert (stride == 1) or (stride == 2)
        self.init_iters = init_iters
        self.out_channels = out_channels
        self.in_channels = in_channels * stride * stride
        self.groups = groups
        assert max(self.out_channels, self.in_channels) % self.groups == 0
        self.max_channels = max(self.out_channels, self.in_channels) // self.groups
        self.device = device
        self.stride = stride
        self.kernel_size = kernel_size
        self.update_iters = update_iters
        self.update_freq = update_freq
        self.total_iters = 0
        self.train_terms = train_terms
        self.eval_terms = eval_terms
        self.testing = testing

        if kernel_size == 1:
            correction = 1.0

        self.random_conv_filter = nn.Parameter(
            torch.randn(
                self.groups,
                self.max_channels,
                self.max_channels,
                self.kernel_size,
                self.kernel_size,
                device=self.device
            ),
            requires_grad=True,
        )
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5 * (self.random_conv_filter - random_conv_filter_T)
        self.conv_filter_norm = ConvFilterNorm(conv_filter)

        self.correction = nn.Parameter(
            torch.tensor([correction], device=self.device), requires_grad=False
        )

        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(
                torch.randn(self.out_channels, device=self.device), requires_grad=True
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.max_channels)
        nn.init.normal_(self.random_conv_filter, std=stdv)

        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def update_sigma(self):
        if self.training:
            if self.total_iters % self.update_freq == 0:
                update_iters = self.init_iters
            else:
                update_iters = self.update_iters
            self.total_iters = self.total_iters + 1
        else:
            update_iters = 0
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5 * (self.random_conv_filter - random_conv_filter_T)
        # pad_size = conv_filter.shape[2] // 2
        with torch.no_grad():
            return self.conv_filter_norm(conv_filter, update_iters)

    def forward(self, x):
        random_conv_filter_T = transpose_filter(self.random_conv_filter).contiguous()
        conv_filter_skew = 0.5 * (self.random_conv_filter - random_conv_filter_T)
        sigma = self.update_sigma()
        # sigma = 1
        conv_filter_n = ((self.correction * conv_filter_skew) / sigma).view(
            self.groups * self.max_channels,
            self.max_channels,
            self.kernel_size,
            self.kernel_size,
        ).contiguous()
        if self.training:
            num_terms = self.train_terms
        else:
            num_terms = self.eval_terms

        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x

        z = curr_z
        for i in range(1, num_terms + 1):
            curr_z = F.conv2d(
                curr_z,
                conv_filter_n,
                padding=(self.kernel_size // 2, self.kernel_size // 2),
                groups=self.groups,
            ) / float(i)
            z = z + curr_z

        if self.out_channels < self.in_channels:
            z = z[:, :self.out_channels, :, :]

        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        if not self.testing:
            return z
        return z, conv_filter_n


# https://github.com/jaxony/ShuffleNet/blob/e9bf42f0cda8dda518cafffd515654cc04584e7a/model.py#L36C1-L53C13
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width).contiguous()

    return x


class MonarchSOC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=True,
        groups=1,
        train_terms=5,
        eval_terms=12,
        init_iters=50,
        update_iters=1,
        update_freq=200,
        correction=0.7,
        device="cuda",
        testing=False
    ):
        super(MonarchSOC, self).__init__()

        self.soc1 = SOC(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
            train_terms=train_terms,
            eval_terms=eval_terms,
            init_iters=init_iters,
            update_iters=update_iters,
            update_freq=update_freq,
            correction=correction,
            device=device,
            testing=testing
        )

        self.soc2 = SOC(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            groups=out_channels//groups, # fix for correct intuition in number of blocks
            train_terms=train_terms,
            eval_terms=eval_terms,
            init_iters=init_iters,
            update_iters=update_iters,
            update_freq=update_freq,
            correction=correction,
            device=device,
            testing=testing
        )
        self.groups = groups
        self.testing = testing
        self.out_channels = out_channels

    def forward(self, x):
        if self.testing:
            x, filter_1 = self.soc1(x)
        else:
            x = self.soc1(x)
        x = channel_shuffle(x, self.groups)
        if not self.testing:
            return channel_shuffle(self.soc2(x), self.out_channels // self.groups)
        result, filter_2 = self.soc2(x)  # for testing
        return channel_shuffle(result, self.out_channels // self.groups), filter_1, filter_2
