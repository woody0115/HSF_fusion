"""Fusion modules for the dual-stream model.

Interface (one module per scale P3/P4/P5):
    forward(evt, rgb, h) -> (evt_out, rgb_out, neck_in, h_new)
        evt, rgb  : backbone features of the two streams at this scale (B, C, H, W)
        h         : hidden state from the previous time step (temporal fusion only), else None
        evt_out   : feature that continues in the event backbone
        rgb_out   : feature that continues in the RGB backbone
        neck_in   : fused feature for the shared neck (C channels)
        h_new     : new hidden state (or None)

'add'  - HSF ablation baseline (Tab. II, first row): element-wise addition, streams unchanged.
'af'   - Adaptive Fusion (Sec. III-D, Fig. 4)
'tfs'  - Temporal Feature Selection (Sec. III-C, Fig. 3) on the event stream + addition
'hsf'  - full HSF: TFS then AF
Stateful modules (tfs, hsf) carry a hidden state h per scale across consecutive frames.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class AddFusion(nn.Module):
    def __init__(self, c):
        super().__init__()

    def forward(self, evt, rgb, h=None):
        return evt, rgb, evt + rgb, None


class AdaptivePool(nn.Module):
    """Eq. 10: F_o = lambda_a * AvgPool(F) + lambda_m * MaxPool(F), lambdas learnable in (0, 1)."""

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.lambda_a = nn.Parameter(torch.zeros(1))     # sigmoid(0) = 0.5
        self.lambda_m = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (torch.sigmoid(self.lambda_a) * F.adaptive_avg_pool2d(x, self.grid)
                + torch.sigmoid(self.lambda_m) * F.adaptive_max_pool2d(x, self.grid))


class AFBlock(nn.Module):
    """Transformer block with adaptive residual weights (Eq. 11-12)."""

    def __init__(self, c, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, c * mlp_ratio), nn.GELU(), nn.Linear(c * mlp_ratio, c))
        # learnable residual weights; 1.0 everywhere = a standard pre-norm transformer block at init
        self.lambda_sa1, self.lambda_sa2 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        self.lambda_mlp1, self.lambda_mlp2 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))

    def forward(self, f):
        y = self.ln1(f)
        f = self.lambda_sa1 * self.attn(y, y, y, need_weights=False)[0] + self.lambda_sa2 * f      # Eq. 11
        f = self.lambda_mlp1 * self.mlp(self.ln2(f)) + self.lambda_mlp2 * f                     # Eq. 12
        return f


class AFFusion(nn.Module):
    """Adaptive Fusion module (HSF Sec. III-D, Fig. 4), a lighter Cross-Modality Fusion Transformer.

    1. each modality is reduced to a grid x grid token map by adaptive mixed pooling,
    2. event and RGB tokens are concatenated, a learnable position embedding is added, and one
       transformer block with adaptive residual weights mixes them (cross-modal self-attention),
    3. the token maps are upsampled back and added to the corresponding stream ("⊕" in Fig. 2),
    4. the two updated streams are concatenated and a 1x1 conv gives the neck input ("©" + Conv).
    Not specified in the paper (our choice): grid 8x8 (as in CFT), one block, 8 heads, MLP ratio 4,
    and zero-initialised learnable gates on the three places where new features enter:
        evt' = evt + lambda_e * up(fe),  rgb' = rgb + lambda_v * up(fv),
        neck = evt' + rgb' + lambda_out * Conv([evt', rgb'])
    so at initialisation AF equals the 'add' baseline and the pretrained single-stream backbones,
    neck and head stay valid (with random gates the neck receives random features and the model
    has to relearn from near zero mAP). Training then decides how much of AF to use.
    """

    def __init__(self, c, grid=8, heads=8, blocks=1):
        super().__init__()
        self.grid = grid
        self.pool_e, self.pool_v = AdaptivePool(grid), AdaptivePool(grid)
        self.pos = nn.Parameter(torch.zeros(1, 2 * grid * grid, c))
        self.blocks = nn.Sequential(*[AFBlock(c, heads) for _ in range(blocks)])
        self.ln = nn.LayerNorm(c)
        self.out = Conv(2 * c, c, 1)                          # Conv + BN + SiLU (Ultralytics)
        self.lambda_e = nn.Parameter(torch.zeros(1))
        self.lambda_v = nn.Parameter(torch.zeros(1))
        self.lambda_out = nn.Parameter(torch.zeros(1))

    def forward(self, evt, rgb, h=None):
        b, c, H, W = evt.shape
        n = self.grid * self.grid
        te = self.pool_e(evt).flatten(2).transpose(1, 2)      # (B, n, C)
        tv = self.pool_v(rgb).flatten(2).transpose(1, 2)
        t = self.ln(self.blocks(torch.cat([te, tv], 1) + self.pos))
        fe = t[:, :n].transpose(1, 2).reshape(b, c, self.grid, self.grid)
        fv = t[:, n:].transpose(1, 2).reshape(b, c, self.grid, self.grid)
        evt = evt + self.lambda_e * F.interpolate(fe, size=(H, W), mode='bilinear', align_corners=False)
        rgb = rgb + self.lambda_v * F.interpolate(fv, size=(H, W), mode='bilinear', align_corners=False)
        neck = evt + rgb + self.lambda_out * self.out(torch.cat([evt, rgb], 1))
        return evt, rgb, neck, None


class TFSCell(nn.Module):
    """Temporal Feature Selection (HSF Sec. III-C, Fig. 3, Eq. 6-9): a convolutional GRU over the
    event features whose gates also look at the RGB features.
        z_t  = sigmoid(W_z [h_{t-1}, x_evt, x_rgb])          update gate        (Eq. 6)
        r_t  = sigmoid(W_r [h_{t-1}, x_evt, x_rgb])          reset gate         (Eq. 7)
        h~_t = tanh(W [r_t * h_{t-1}, x_evt])                 candidate          (Eq. 8)
        h_t  = (1 - z_t) * h_{t-1} + z_t * h~_t               new state / output (Eq. 9)
    W_z, W_r, W are 1x1 convolutions. Our addition (not in the paper): the event stream continues
    with x_evt + lambda_t * h_t where lambda_t starts at 0, so the pretrained event backbone is
    undisturbed at initialisation (same reasoning as the gates in AFFusion).
    """

    def __init__(self, c):
        super().__init__()
        self.w_z = nn.Conv2d(3 * c, c, 1)
        self.w_r = nn.Conv2d(3 * c, c, 1)
        self.w_h = nn.Conv2d(2 * c, c, 1)
        self.lambda_t = nn.Parameter(torch.zeros(1))

    def forward(self, evt, rgb, h):
        if h is None:
            h = torch.zeros_like(evt)
        zr_in = torch.cat([h, evt, rgb], 1)
        z = torch.sigmoid(self.w_z(zr_in))
        r = torch.sigmoid(self.w_r(zr_in))
        h_cand = torch.tanh(self.w_h(torch.cat([r * h, evt], 1)))
        h_new = (1 - z) * h + z * h_cand
        return evt + self.lambda_t * h_new, h_new


class TFSFusion(nn.Module):
    """TFS on the event stream, then element-wise addition (ablation row "TFS only")."""
    stateful = True

    def __init__(self, c):
        super().__init__()
        self.tfs = TFSCell(c)

    def forward(self, evt, rgb, h=None):
        evt, h = self.tfs(evt, rgb, h)
        return evt, rgb, evt + rgb, h


class HSFFusion(nn.Module):
    """Full HSF fusion at one scale: TFS on the event stream, then AF (Fig. 2)."""
    stateful = True

    def __init__(self, c):
        super().__init__()
        self.tfs = TFSCell(c)
        self.af = AFFusion(c)

    def forward(self, evt, rgb, h=None):
        evt, h = self.tfs(evt, rgb, h)
        evt, rgb, neck, _ = self.af(evt, rgb)
        return evt, rgb, neck, h


ILLUM_DIM = 16 + 48          # luminance histogram (16 bins) + 4x4 mean colour


def illum_descriptor(rgb):
    """Global lighting descriptor of the RGB network input (B, 3, H, W) in [0, 1] -> (B, 64):
    a 16-bin luminance histogram and the 4x4 average colour. Not learnable; the letterbox padding
    is constant grey and simply occupies a fixed share of one bin."""
    rgb = rgb.float()
    lum = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    small = F.avg_pool2d(lum, 4).flatten(1)                                   # (B, N)
    idx = (small.clamp(0, 0.9999) * 16).long()
    hist = torch.zeros(rgb.shape[0], 16, device=rgb.device).scatter_add_(1, idx, torch.ones_like(small))
    hist = hist / small.shape[1]
    col = F.adaptive_avg_pool2d(rgb, 4).flatten(1)                            # (B, 48)
    return torch.cat([hist, col], 1)


class IllumGate(nn.Module):
    """Illumination-aware fusion (our extension, not in HSF): wraps a fusion module and decides per
    image and per channel how much of the fused contribution reaches the neck and the RGB stream:
        g        = 2 * sigmoid(MLP([illum descriptor, GAP(evt), GAP(rgb)]))   in (0, 2), (B, C)
        neck     = rgb + g * (neck_base - rgb)       (add: rgb + g * evt)
        rgb_out  = rgb + g * (rgb_base - rgb)        (AF's injection into the RGB stream)
    The MLP's last layer starts at zero, so g = 1 and the model equals the wrapped fusion at
    initialisation. keep (B,) from the model zeroes g for modality-dropout samples (pure RGB path),
    which teaches the neck to also work at g = 0 so the gate can really switch the event stream off
    in good light. last_gate keeps the per-image mean of g for analysis."""

    def __init__(self, base, c, hidden=64):
        super().__init__()
        self.base = base
        self.stateful = bool(getattr(base, 'stateful', False))
        self.mlp = nn.Sequential(nn.LayerNorm(ILLUM_DIM + 2 * c), nn.Linear(ILLUM_DIM + 2 * c, hidden),
                                 nn.SiLU(), nn.Linear(hidden, c))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_gate = None

    def forward(self, evt, rgb, h=None, illum=None, keep=None):
        evt2, rgb2, neck, h = self.base(evt, rgb, h)
        z = torch.cat([illum, evt.float().mean((2, 3)), rgb.float().mean((2, 3))], 1)
        g = 2 * torch.sigmoid(self.mlp(z))
        if keep is not None:
            g = g * keep[:, None]
        self.last_gate = g.detach().mean(1)
        g = g[:, :, None, None].to(neck.dtype)
        return evt2, rgb + g * (rgb2 - rgb), rgb + g * (neck - rgb), h


FUSIONS = {'add': AddFusion, 'af': AFFusion, 'tfs': TFSFusion, 'hsf': HSFFusion}


def build_fusion(name, c, scale_idx=0):
    """'iaf_<base>' = <base> fusion wrapped in the illumination-aware gate (e.g. iaf_add, iaf_af)."""
    base = name[4:] if name.startswith('iaf_') else name
    if base not in FUSIONS:
        raise ValueError(f'unknown fusion "{name}" (available: {list(FUSIONS)} and iaf_<name>)')
    m = FUSIONS[base](c)
    return IllumGate(m, c) if name.startswith('iaf_') else m
