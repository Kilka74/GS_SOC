import pytest
import torch
from skew_ortho_conv import MonarchSOC, channel_shuffle


@torch.no_grad()
@pytest.mark.parametrize(
    ["device", "groups"],
    [
        ["cpu", 1],
        # ["cpu", 4]
    ]
)
def test_monarch_soc(device, groups):
    x = torch.randn(4, 64, 64, 64)
    soc = MonarchSOC(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        groups=groups,
        device=device,
        testing=True
    )
    y = channel_shuffle(soc(x), groups=groups)
    assert y.shape == (4, 64, 64, 64), y.shape
    assert torch.allclose(y, x), torch.norm(y - x)
