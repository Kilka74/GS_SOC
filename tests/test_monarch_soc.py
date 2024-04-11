import pytest
import torch
import torch.nn.functional as F
from skew_ortho_conv import MonarchSOC, channel_shuffle


@torch.no_grad()
@pytest.mark.parametrize(
    ["device", "groups"],
    [
        ["cpu", 1],
        ["cpu", 4]
    ]
)
def test_monarch_soc(device, groups):
    x = torch.randn(4, 64, 64, 64)
    x /= torch.norm(x)
    soc = MonarchSOC(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        groups=groups,
        device=device,
        testing=True,
        bias=False
    )
    res, filter_1, filter_2 = soc(x)
    assert not torch.any(res.isnan())
    curr_x = x
    for i in range(1, 21):
        curr_x = F.conv2d(
            curr_x,
            filter_1,
            padding=(1, 1),
            groups=groups
        ) / float(i)
        x = x + curr_x
    assert not torch.any(x.isnan())
    x = channel_shuffle(x, groups=groups)
    curr_x = x
    for i in range(1, 21):
        curr_x = F.conv2d(
            curr_x,
            filter_2,
            padding=(1, 1),
            groups=groups
        ) / float(i)
        x = x + curr_x
    assert res.shape == x.shape
    assert torch.allclose(x, res), torch.norm(res - x)
