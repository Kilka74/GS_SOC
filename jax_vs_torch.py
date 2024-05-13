import jax
import jax.numpy as jnp
from jax import random
from jax import jit
import hydra
import wandb
import timeit
import torch
from torch import nn as nn
from torch.utils import benchmark


key = random.PRNGKey(0)

# Define the convolution function
def conv2d(inputs, kernel, strides=1, groups=1):
    """2D convolution with non-padded, 'valid' mode."""
    outputs = jax.lax.conv_general_dilated(
        inputs, kernel, window_strides=(strides, strides), padding='VALID',
        dimension_numbers=('NCHW', 'HWIO', 'NCHW'),
        feature_group_count=groups)
    return outputs


# Define the convolutional layer
class Conv2DLayer:
    def __init__(self, in_channels, out_channels, filter_shape, strides=1, padding='VALID', groups=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding = padding
        self.groups = groups
        fan_in = self.filter_shape[0] * self.filter_shape[1]
        self.w_init = random.normal(key, (self.filter_shape[0], self.filter_shape[1], self.in_channels//self.groups, self.out_channels)) * (1. / jnp.sqrt(fan_in))

    def __call__(self, inputs):
        return conv2d(inputs, self.w_init, self.strides, self.groups)

def warmup():
    linear = nn.Linear(10000, 512, device="cuda")
    x = torch.randn(64, 10000, device="cuda")
    t0 = benchmark.Timer(stmt="linear(x)", globals={"x": x, "linear": linear})
    res = t0.timeit(1000).mean
    del x
    del linear
    torch.cuda.empty_cache()

def channel_shuffle_jax(x, groups):
    batchsize, num_channels, height, width = x.shape

    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = jnp.transpose(x, (0, 2, 1, 3, 4))

    # flatten
    x = x.reshape(batchsize, -1, height, width)

    return x


def jax_conv_compiled(batch_size, in_channels, out_channels, h, w, groups=1):
    key = random.PRNGKey(0)
    layer = Conv2DLayer(in_channels, out_channels, filter_shape=(3, 3), groups=groups)

    jit_layer = jit(layer)
    inputs = random.normal(key, (batch_size, in_channels, h, w))
    time = timeit.timeit(
        stmt="jit_layer(inputs)",
        globals={
            "jit_layer": jit_layer,
            "inputs": inputs
        },
        number=1000
    )
    del inputs
    del jit_layer
    del layer
    return time


def torch_conv(batch_size, in_channels, out_channels, h, w, groups=1):
    inputs = torch.randn(batch_size, in_channels, h, w).cuda()
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=groups).cuda()
    time = timeit.timeit(
        stmt="conv(inputs)",
        globals={
            "conv": conv,
            "inputs": inputs
        },
        number=1000
    )
    del conv
    del inputs
    return time

@hydra.main(config_path="conf", config_name="compare_config", version_base=None)
def main(args):
    num_channels = list(map(int, args.channels.split()))
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(
        entity="kilka74",
        project="MonarchSOC",
        tags=['jax_vs_torch'],
        name=f"jax vs torch, hidden_size=({args.hidden_size}, {args.hidden_size}), groups={args.groups}",
        config={
            "hidden_size": args.hidden_size,
            "kernel_size": args.kernel_size,
            "batch_size": args.batch_size,
            "groups": args.groups,
        }
    )
    for i in range(5):
        warmup()

    torch.cuda.empty_cache()

    for channels in num_channels:
        jax_compiled_time = jax_conv_compiled(
            args.batch_size,
            channels,
            channels,
            args.hidden_size,
            args.hidden_size,
            args.groups
        )
        torch_time = torch_conv(
            args.batch_size,
            channels,
            channels,
            args.hidden_size,
            args.hidden_size,
            args.groups
        )
        wandb.log({
            "channels": channels,
            "torch_time": torch_time,
            "jax_time": jax_compiled_time
        })
        torch.cuda.empty_cache()

    wandb.finish()

if __name__ == "__main__":
    main()