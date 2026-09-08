import torch

from strokeai.losses import bce_dice_loss, soft_dice_loss
from strokeai.models import UNet2D


def test_unet_shapes_and_params():
    m = UNet2D(in_ch=2, base=16, depth=4)
    x = torch.randn(2, 2, 128, 128)
    y = m(x)
    assert y.shape == (2, 1, 128, 128)
    n = sum(p.numel() for p in m.parameters())
    assert 1e6 < n < 3e6


def test_dice_loss_bounds_and_empty_target():
    logits = torch.randn(4, 1, 16, 16)
    t = torch.zeros(4, 1, 16, 16)
    d = soft_dice_loss(logits, t)
    assert torch.isfinite(d) and 0 <= d <= 1
    t[0, 0, :4, :4] = 1
    loss, parts = bce_dice_loss(logits, t)
    assert torch.isfinite(loss) and parts["bce"] > 0


def test_overfit_two_batches():
    """A4 sanity check: the model must be able to memorise 2 slices."""
    torch.manual_seed(0)
    m = UNet2D(in_ch=2, base=8, depth=3)
    x = torch.randn(2, 2, 64, 64)
    t = torch.zeros(2, 1, 64, 64); t[:, :, 20:40, 20:40] = 1
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, _ = bce_dice_loss(m(x), t)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first and loss.item() < 0.5, (first, loss.item())
