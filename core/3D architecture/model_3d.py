"""
3D U-Net Architecture for Brain Tumor Segmentation

A flexible, production-ready 3D U-Net implementation optimized for medical image segmentation.
Supports variable depth, configurable filters, and mixed precision training on RTX 4060.

Key Features:
- 4-channel input (T1, T1CE, T2, FLAIR)
- Configurable depth and base filters
- Instance normalization for stability
- Dropout for regularization
- Skip connections with optional attention
- Memory-efficient design for 128³ volumes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import warnings


class ConvBlock3D(nn.Module):
    """
    3D Convolutional Block with InstanceNorm and LeakyReLU
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        padding: Padding size
        dropout_rate: Dropout probability
        use_attention: Whether to use squeeze-excite attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Optional squeeze-excite attention
        self.attention = SqueezeExcite3D(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        
        # Optional attention
        out = self.attention(out)
        
        return out


class SqueezeExcite3D(nn.Module):
    """
    3D Squeeze-and-Excite attention module
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class DownBlock3D(nn.Module):
    """
    Downsampling block with MaxPool and ConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv_block = ConvBlock3D(
            in_channels, out_channels, dropout_rate=dropout_rate, use_attention=use_attention
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Save skip connection before pooling
        skip = x
        
        # Downsample and process
        x = self.pool(x)
        x = self.conv_block(x)
        
        return x, skip


class UpBlock3D(nn.Module):
    """
    Upsampling block with transpose conv and skip connections
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(
            (in_channels // 2) + skip_channels, 
            out_channels, 
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch (crop or pad skip connection)
        if x.shape != skip.shape:
            x = self._match_size(x, skip)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process concatenated features
        x = self.conv_block(x)
        
        return x
    
    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match tensor sizes by cropping or padding"""
        target_size = target.shape[2:]  # D, H, W
        x_size = x.shape[2:]
        
        # Calculate padding/cropping for each dimension
        pad_crop = []
        for i in range(3):
            diff = target_size[i] - x_size[i]
            if diff > 0:  # Need padding
                pad_crop.extend([diff // 2, diff - diff // 2])
            else:  # Need cropping
                start = (-diff) // 2
                end = start - diff
                x = x[:, :, 
                     start:end if i == 0 else slice(None),
                     start:end if i == 1 else slice(None),
                     start:end if i == 2 else slice(None)]
        
        # Apply padding if needed
        if any(p > 0 for p in pad_crop):
            x = F.pad(x, pad_crop[::-1])  # PyTorch pad expects reverse order
        
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for brain tumor segmentation
    
    Args:
        in_channels: Number of input modalities (4 for T1, T1CE, T2, FLAIR)
        out_channels: Number of output classes (1 for binary segmentation)
        base_filters: Base number of filters in first layer
        depth: Number of down/up sampling levels
        dropout_rate: Dropout probability
        use_attention: Whether to use squeeze-excite attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        dropout_rate: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        
        if depth < 2 or depth > 6:
            warnings.warn(f"Depth {depth} may not be optimal. Recommended: 3-5")
        
        self.depth = depth
        self.base_filters = base_filters
        
        # Calculate filter sizes for each level
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution block
        self.input_conv = ConvBlock3D(
            in_channels, filters[0], dropout_rate=dropout_rate, use_attention=use_attention
        )
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                DownBlock3D(
                    filters[i], filters[i + 1], 
                    dropout_rate=dropout_rate, 
                    use_attention=use_attention
                )
            )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(
                UpBlock3D(
                    filters[depth - i], filters[depth - i - 1], filters[depth - i - 1],
                    dropout_rate=dropout_rate,
                    use_attention=use_attention
                )
            )
        
        # Final classification layer
        self.output_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 4, D, H, W)
            
        Returns:
            Output tensor of shape (B, 1, D, H, W)
        """
        # Validate input
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got {x.dim()}D")
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Encoder path - store skip connections
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
        
        # Decoder path - use skip connections in reverse order
        skip_connections.reverse()
        for up_block, skip in zip(self.up_blocks, skip_connections):
            x = up_block(x, skip)
        
        # Final classification
        x = self.output_conv(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_size(self) -> Tuple[int, float]:
        """
        Get model size information
        
        Returns:
            (num_parameters, size_in_mb)
        """
        num_params = sum(p.numel() for p in self.parameters())
        size_mb = num_params * 4 / (1024 ** 2)  # Assuming float32
        return num_params, size_mb
    
    def estimate_memory_usage(self, batch_size: int = 1, input_size: Tuple[int, int, int] = (128, 128, 128)) -> float:
        """
        Estimate GPU memory usage in MB
        
        Args:
            batch_size: Batch size
            input_size: Input volume size (D, H, W)
            
        Returns:
            Estimated memory usage in MB
        """
        d, h, w = input_size
        
        # Input memory
        input_mem = batch_size * 4 * d * h * w * 4 / (1024 ** 2)  # 4 channels, float32
        
        # Feature map memory (rough estimate)
        feature_mem = 0
        current_size = [d, h, w]
        current_filters = self.base_filters
        
        # Encoder
        for i in range(self.depth + 1):
            vol_size = current_size[0] * current_size[1] * current_size[2]
            feature_mem += batch_size * current_filters * vol_size * 4 / (1024 ** 2)
            
            if i < self.depth:
                current_size = [s // 2 for s in current_size]
                current_filters *= 2
        
        # Model parameters
        _, param_mem = self.get_model_size()
        
        # Add some overhead for gradients and optimizer states
        total_mem = (input_mem + feature_mem + param_mem) * 2.5
        
        return total_mem


def create_unet3d(config: str = "standard") -> UNet3D:
    """
    Factory function to create pre-configured U-Net models
    
    Args:
        config: Configuration preset ("light", "standard", "heavy", "attention")
        
    Returns:
        Configured UNet3D model
    """
    configs = {
        "light": {
            "base_filters": 16,
            "depth": 3,
            "dropout_rate": 0.1,
            "use_attention": False
        },
        "standard": {
            "base_filters": 32,
            "depth": 4,
            "dropout_rate": 0.1,
            "use_attention": False
        },
        "heavy": {
            "base_filters": 64,
            "depth": 5,
            "dropout_rate": 0.15,
            "use_attention": False
        },
        "attention": {
            "base_filters": 32,
            "depth": 4,
            "dropout_rate": 0.1,
            "use_attention": True
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")
    
    return UNet3D(**configs[config])


def test_model():
    """Test function to verify model works correctly"""
    print("Testing 3D U-Net model...")
    
    # Test with different configurations
    configs = ["light", "standard", "heavy", "attention"]
    
    for config_name in configs:
        print(f"\nTesting {config_name} configuration:")
        
        model = create_unet3d(config_name)
        model.eval()
        
        # Test input (batch_size=1, channels=4, depth=128, height=128, width=128)
        test_input = torch.randn(1, 4, 128, 128, 128)
        
        try:
            with torch.no_grad():
                output = model(test_input)
            
            num_params, size_mb = model.get_model_size()
            memory_est = model.estimate_memory_usage(batch_size=2)
            
            print(f"  ✓ Input shape: {test_input.shape}")
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Parameters: {num_params:,} ({size_mb:.1f} MB)")
            print(f"  ✓ Estimated GPU memory (batch=2): {memory_est:.1f} MB")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    test_model()