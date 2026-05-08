import torch
import torch.nn as nn
import torch.nn.functional as F
from arch.model_ablation import Light_SplitNet, Light_SplitNet_Pooling
import torchsummary


#################################################################################    
# Upsample-Denoising-Downsample--------------------------------------------------
#################################################################################

class HighResWrapper(nn.Module):
    def __init__(self, base_channels: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (scale ** 2)   # 3 * 16 = 48

        self.proj_in = nn.Conv2d(in_ch, base_channels, kernel_size=1, bias=True)

        self.denoiser = Light_SplitNet(
            input_channels=base_channels,
            base_channels=base_channels,
        )

        # 채널 복원: base_channels → 48
        self.proj_out = nn.Conv2d(base_channels, in_ch, kernel_size=1, bias=True)

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.pixel_shuffle   = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual (원본 해상도 유지)
        residual = x                              # [B, 3, H, W]

        # Downsampling (space → channel)
        x = self.pixel_unshuffle(x)              # [B, 32, H/4, W/4]

        # 채널 정렬
        x = self.proj_in(x)                      # [B, base_ch, H/4, W/4]

        # Denoising (Light_SplitNet의 padder_size=16 기준 내부 패딩 처리됨)
        x = self.denoiser(x)                     # [B, base_ch, H/4, W/4]

        # 채널 복원
        x = self.proj_out(x)                     # [B, 48, H/4, W/4]

        # Upsampling (channel → space)
        x = self.pixel_shuffle(x)               # [B, 3, H, W]

        return x + residual

#################################################################################    
# Upsample-Denoising-Downsample_Pooling------------------------------------------
#################################################################################

    
class HighResWrapper_Pooling(nn.Module):

    def __init__(self, base_channels: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (scale ** 2)   # 3 * 16 = 48

        # 채널 축소: 48 → base_channels
        self.proj_in = nn.Conv2d(in_ch, base_channels, kernel_size=1, bias=True)

        # 내부 denoiser (입출력 채널을 base_channels로 맞춤)
        self.denoiser = Light_SplitNet_Pooling(
            input_channels=base_channels,
            base_channels=base_channels,
        )

        # 채널 복원: base_channels → 48
        self.proj_out = nn.Conv2d(base_channels, in_ch, kernel_size=1, bias=True)

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.pixel_shuffle   = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual (원본 해상도 유지)
        residual = x                              # [B, 3, H, W]

        # Downsampling (space → channel)
        x = self.pixel_unshuffle(x)              # [B, 32, H/4, W/4]

        # 채널 정렬
        x = self.proj_in(x)                      # [B, base_ch, H/4, W/4]

        # Denoising (Light_SplitNet의 padder_size=16 기준 내부 패딩 처리됨)
        x = self.denoiser(x)                     # [B, base_ch, H/4, W/4]

        # 채널 복원
        x = self.proj_out(x)                     # [B, 48, H/4, W/4]

        # Upsampling (channel → space)
        x = self.pixel_shuffle(x)               # [B, 3, H, W]

        return x + residual
    
if __name__ == '__main__':
    img_channel = 3
    width = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom = HighResWrapper()
    custom.to(device,dtype = torch.float32)
    
    
    #Model Summary
    
    torchsummary.summary(custom,(3,1024,1024))
    # Model Complexity
    from ptflops import get_model_complexity_info
    
    macs, params = get_model_complexity_info(custom, (3,1024, 1024), verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(f"Custom MACS: {macs}, PARAMS:{params}")
