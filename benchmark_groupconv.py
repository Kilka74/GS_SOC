import torch
import torch.utils.benchmark as benchmark
from torch import nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


def warmup():
    linear = nn.Linear(10000, 512, device="cuda")
    x = torch.randn(64, 10000, device="cuda")
    t0 = benchmark.Timer(stmt="linear(x)", globals={"x": x, "linear": linear})
    res = t0.timeit(1000).mean
    del x
    del linear
    torch.cuda.empty_cache()

def run_in_streams(kernel, x):
    out = []
    xs = torch.split(x, kernel.shape[1], dim=1)
    for ind, stream in enumerate([torch.cuda.Stream("cuda") for _ in range(kernel.shape[0])]):
        with torch.cuda.stream(stream):
            out.append(F.conv2d(
                xs[ind],
                kernel[ind, ...],
                padding=(kernel.shape[-1] // 2, kernel.shape[-1] // 2)))
    torch.cuda.synchronize()
    return torch.cat(out, dim=1)

for i in range(5):
    warmup()

groups = [1, 2, 4, 8, 16, 32]
in_channels, out_channels = 512, 512
for group in groups:
    print(f"Groups: {group}")
    kernel = torch.randn(group, in_channels//group, out_channels//group, 3, 3, device="cuda")
    x = torch.randn(128, 512, 32, 32, device="cuda")
    torch.cuda.empty_cache()
    for _ in range(2):  # warm-up CUDA
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("model_inference"):
                res_manual = run_in_streams(kernel, x)
    print('Manually put each convolution in a CUDA stream:')
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    fused_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        groups=group,
        bias=False,
        padding=(1, 1)
    ).cuda()
    fused_conv.weight = nn.Parameter(kernel.view(in_channels, out_channels//group, 3, 3))

    for _ in range(2):  # warm-up CUDA
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("model_inference"):
                out_fused = fused_conv(x)
    print('Use a single convolution with groups')
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("--------------------------------------")
    assert out_fused.shape == res_manual.shape
    assert torch.allclose(out_fused, res_manual), torch.norm(out_fused - res_manual)
    del x
    del kernel
    del out_fused
    del res_manual
    torch.cuda.empty_cache()
