import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

class VBConv2d(nn.Module):
    """
    Bayesian Conv2d with learnable posterior q(w|mu, rho) and Gaussian prior p(w|0,1).
    Reparameterization trick; returns KL term via .kl_loss().
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prior_sigma=1.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight_mu = nn.Parameter(torch.Tensor(out_ch, in_ch // groups, *self.kernel_size).normal_(0, 0.02))
        self.weight_rho = nn.Parameter(torch.Tensor(out_ch, in_ch // groups, *self.kernel_size).fill_(-3.0))
        self.register_buffer('prior_mu', torch.zeros_like(self.weight_mu))
        self.register_buffer('prior_sigma', torch.ones_like(self.weight_mu) * prior_sigma)
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_ch))
            self.bias_rho = nn.Parameter(torch.ones(out_ch)*-3.0)
            self.register_buffer('bias_prior_mu', torch.zeros_like(self.bias_mu))
            self.register_buffer('bias_prior_sigma', torch.ones_like(self.bias_mu)*prior_sigma)
        else:
            self.bias_mu = None
            self.bias_rho = None
            self.bias_prior_mu = None
            self.bias_prior_sigma = None
        self._kl = 0.0

    def _sample(self, mu, rho):
        sigma = torch.log1p(torch.exp(rho))  
        eps = torch.randn_like(mu)
        return mu + sigma * eps, sigma

    def forward(self, x):
        w, w_sigma = self._sample(self.weight_mu, self.weight_rho) if self.training else (self.weight_mu, torch.log1p(torch.exp(self.weight_rho)))
        b = None
        if self.bias_mu is not None:
            b, b_sigma = self._sample(self.bias_mu, self.bias_rho) if self.training else (self.bias_mu, torch.log1p(torch.exp(self.bias_rho)))

        out = F.conv2d(x, w, b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        w_kl = self._kl_gaussian(w, w_sigma, self.prior_mu, self.prior_sigma)
        if self.bias_mu is not None:
            b_kl = self._kl_gaussian(b, b_sigma, self.bias_prior_mu, self.bias_prior_sigma)
        else:
            b_kl = torch.tensor(0.0, device=x.device)
        self._kl = w_kl + b_kl
        return out

    @staticmethod
    def _kl_gaussian(q_mu, q_sigma, p_mu, p_sigma, eps=1e-8):
        term = torch.log((p_sigma + eps)/(q_sigma + eps)) + (q_sigma**2 + (q_mu - p_mu)**2)/(2*(p_sigma**2 + eps)) - 0.5
        return term.sum()

    def kl_loss(self):
        return self._kl

def sinusoidal_position_encoding_2d(h, w, c, device):
    """
    Classic sine-cos 2D positions (split channels for x/y).
    c must be even.
    """
    if c % 2 != 0:
        c = c + 1  # pad to even
    y_pos = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
    x_pos = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)
    div = torch.exp(torch.arange(0, c//2, device=device) * (-math.log(10000.0) / (c//2)))
    pe_x = torch.zeros(h, w, c//2, device=device)
    pe_y = torch.zeros(h, w, c//2, device=device)
    pe_x[..., 0::2] = torch.sin(x_pos.unsqueeze(-1) * div[0::2])
    pe_x[..., 1::2] = torch.cos(x_pos.unsqueeze(-1) * div[1::2])
    pe_y[..., 0::2] = torch.sin(y_pos.unsqueeze(-1) * div[0::2])
    pe_y[..., 1::2] = torch.cos(y_pos.unsqueeze(-1) * div[1::2])
    pe = torch.cat([pe_x, pe_y], dim=-1) 
    return pe.permute(2,0,1).unsqueeze(0) 

class SpatialMHAttention(nn.Module):
    """
    Multi-head attention over spatial tokens (H*W). Includes uncertainty-gated residual.
    """
    def __init__(self, in_ch, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch//4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//4, in_ch, 1), nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pe = sinusoidal_position_encoding_2d(H, W, C, x.device)
        x_pe = x + pe

        q = self.q(x_pe).view(B, self.heads, C//self.heads, H*W) 
        k = self.k(x_pe).view(B, self.heads, C//self.heads, H*W)
        v = self.v(x).view(B, self.heads, C//self.heads, H*W)
        attn = torch.einsum('bhdk,bhdn->bhkn', q, k) / math.sqrt(C//self.heads)  
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhkn,bhdn->bhdk', attn, v).contiguous().view(B, C, H, W)
        out = self.proj(out)
        gate = self.gate(x)
        out = self.norm(x + gate * out)
        return out

class CrossScaleFusion(nn.Module):
    def __init__(self, c2, c3, c4, fused_ch):
        super().__init__()
        self.proj2 = nn.Conv2d(c2, fused_ch, 1)
        self.proj3 = nn.Conv2d(c3, fused_ch, 1)
        self.proj4 = nn.Conv2d(c4, fused_ch, 1)
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta  = nn.Parameter(torch.tensor([1.0]))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.sa = SpatialMHAttention(fused_ch, heads=4)

    def forward(self, f2, f3, f4):
        # Upsample to f2 size
        H, W = f2.shape[-2:]
        p2 = self.proj2(f2)
        p3 = F.interpolate(self.proj3(f3), size=(H,W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=(H,W), mode='bilinear', align_corners=False)
        weights = torch.softmax(torch.stack([self.alpha, self.beta, self.gamma]), dim=0)
        fused = weights[0]*p2 + weights[1]*p3 + weights[2]*p4
        fused = self.sa(fused)
        return fused

class TumorNet34Bayes(nn.Module):
    """
    ResNet34 backbone with:
      - VBConv at stem, last conv of layer3 and layer4
      - Spatial MH attention after layer2/3/4
      - Cross-scale fusion (l2,l3,l4)
      - Dual pooling + classifier with MC Dropout
      - Uncertainty (aleatoric head)
    """
    def __init__(self, dropout_p=0.4, num_classes=1):
        super().__init__()
        base = tv.resnet34(weights=tv.ResNet34_Weights.IMAGENET1K_V1)

        # Replace stem conv with VBConv
        self.stem_bayes = VBConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.stem_bn = base.bn1
        self.stem_relu = base.relu
        self.stem_pool = base.maxpool

        # Copy layers
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        # Replace last conv in layer3 & layer4 basic blocks with VBConv
        self.layer3 = self._replace_last_conv_with_vb(base.layer3)
        self.layer4 = self._replace_last_conv_with_vb(base.layer4)

        c2 = 128
        c3 = 256
        c4 = 512

        # Attention after each
        self.att2 = SpatialMHAttention(c2, heads=4)
        self.att3 = SpatialMHAttention(c3, heads=4)
        self.att4 = SpatialMHAttention(c4, heads=4)

        # Cross-scale fusion
        self.fuse = CrossScaleFusion(c2, c3, c4, fused_ch=256)

        # Dual pooling + 1x1 conv fusion
        self.mix = nn.Conv2d(256, 256, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Classifier with MC Dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

        # Aleatoric uncertainty head (predicts log-variance over logits)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()  
        )

    @staticmethod
    def _replace_last_conv_with_vb(layer):
        # layer is Sequential of BasicBlocks; replace last block's conv2 with VBConv
        blocks = []
        for i, b in enumerate(layer):
            if i == len(layer)-1:
                vb = VBConv2d(b.conv2.in_channels, b.conv2.out_channels, kernel_size=3, stride=b.conv2.stride,
                              padding=b.conv2.padding, bias=(b.conv2.bias is not None))
                b.conv2 = vb
            blocks.append(b)
        return nn.Sequential(*blocks)

    def forward(self, x):
        kl_total = 0.0

        # Stem
        x = self.stem_bayes(x); kl_total += self.stem_bayes.kl_loss()
        x = self.stem_bn(x); x = self.stem_relu(x); x = self.stem_pool(x)

        # Residual body
        x1 = self.layer1(x)       
        x2 = self.layer2(x1)      
        x3 = self.layer3(x2)      
        # accumulate KL if last block has VBConv
        for m in self.layer3.modules():
            if isinstance(m, VBConv2d): kl_total += m.kl_loss()
        x4 = self.layer4(x3)      
        for m in self.layer4.modules():
            if isinstance(m, VBConv2d): kl_total += m.kl_loss()

        x2 = self.att2(x2)
        x3 = self.att3(x3)
        x4 = self.att4(x4)

        fused = self.fuse(x2, x3, x4)
        fused = self.mix(fused)

        gap = self.gap(fused).flatten(1)
        gmp = self.gmp(fused).flatten(1)
        feat = torch.cat([gap, gmp], dim=1)  

        logits = self.classifier(feat)
        logvar_raw = self.aleatoric_head(feat)  
        logvar = torch.log1p(torch.exp(logvar_raw * 3.0)) - 1e-6  

        return logits, logvar, kl_total