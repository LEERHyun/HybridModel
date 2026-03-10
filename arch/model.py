import torch
import torch.nn as nn
import torch.nn.functional as F
from arch.arch_util import LayerNorm2d
import numbers
from einops import rearrange
import torchsummary

###########################################################################
#Upsample, Downsample, Concatenation---------------------------------------
###########################################################################

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
def concat_tensor(x1, x2):
    return torch.cat((x1,x2),dim=1)
    
####################################################################################################################################################
#Transformer Module---------------------------------------------------------------------------------------------------------------------------------
####################################################################################################################################################

#Normalization Module
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
   
#Layer normalization Main
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

#FeedForward Newtork    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.sg = SimpleGate()
        self.project_out = nn.Conv2d(hidden_features//2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1,x2 = self.project_in(x).chunk(2, dim=1)
        x = self.sg(x1)*self.sg(x2)
        x = self.project_out(x)
        return x

#Transposed Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


####################################################################################################################################################
#CNN Module-----------------------------------------------------------------------------------------------------------------------------------------
####################################################################################################################################################


#CNN Block Architecture(Simplified Channel Attention x)
class CNNBlock(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, padding=1, stride=1, groups=base_channels,
                               bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(base_channels)
        self.beta = nn.Parameter(torch.zeros((1, base_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):  
        x = inp
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        y = inp + x * self.beta
        return y

#CNN Block Architecture(Simplified Channel Attention o)
class SCA_CNNBlock(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, padding=1, stride=1, groups=base_channels,
                               bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(base_channels)
        self.beta = nn.Parameter(torch.zeros((1, base_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        
        x = inp
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        
        x = x * self.sca(x)
        y = inp + x * self.beta


        return y

#SimpleGate    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

#################################################################################    
# Hybrid Network-----------------------------------------------------------------
#################################################################################
class HybridNet(nn.Module):

    def __init__(self, input_channels=3, base_channels=32, num_heads=[1,1,1,1], 
                 ffn_expansion_factor = 2, bias=False, LayerNorm_type='WithBias'):
        super(HybridNet, self).__init__()
        
        # ============ Initial Convolution ============
        self.intro = nn.Conv2d(input_channels, base_channels, 3, 1, 1)
        
        # ============ Encoder ============
        # Stage 1: 32 channels
        self.enc1_transformer = TransformerBlock(base_channels, num_heads[0], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.enc1_cnn = CNNBlock(base_channels)
        self.down1 = Downsample(base_channels)
        
        # Stage 2: 64 channels
        self.enc2_transformer = TransformerBlock(base_channels * 2, num_heads[1], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.enc2_cnn = CNNBlock(base_channels * 2)
        self.down2 = Downsample(base_channels * 2)
        
        # Stage 3: 128 channels
        self.enc3_transformer = TransformerBlock(base_channels * 4, num_heads[2], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.enc3_cnn = CNNBlock(base_channels * 4)
        self.down3 = Downsample(base_channels * 4)
        
        # ============ Bottleneck: 256 channels ============
        self.bottleneck_transformer = TransformerBlock(base_channels * 8, num_heads[3], 
                                                      ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn = CNNBlock(base_channels * 8)
        
        # ============ Decoder ============
        # Stage 3: 256 → 128
        self.up3 = Upsample(base_channels * 8)
        self.dec3_transformer = TransformerBlock(base_channels * 4, num_heads[2], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.dec3_cnn = CNNBlock(base_channels * 4)
        
        # Stage 2: 128 → 64
        self.up2 = Upsample(base_channels * 4)
        self.dec2_transformer = TransformerBlock(base_channels * 2, num_heads[1], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.dec2_cnn = CNNBlock(base_channels * 2)
        
        # Stage 1: 64 → 32
        self.up1 = Upsample(base_channels*2)
        self.dec1_transformer = TransformerBlock(base_channels, num_heads[0], 
                                                ffn_expansion_factor, bias, LayerNorm_type)
        self.dec1_cnn = CNNBlock(base_channels)
        
        # ============ Output ============
        self.output = nn.Conv2d(base_channels, input_channels, 3, 1, 1)
        
        self.padder_size = 2**4
        
    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
        
        # ============ Initial ============
        x = self.intro(inp)  # [B, 32, H, W]
        
        # ============ Encoder ============
        # Stage 1: [B, 32, H, W]
        x1 = self.enc1_transformer(x)
        x1 = self.enc1_cnn(x1)
        skip1 = x1  # Skip connection
        x = self.down1(x1)  # [B, 64, H/2, W/2]

        
        # Stage 2: [B, 64, H/2, W/2]
        x2 = self.enc2_transformer(x)
        x2 = self.enc2_cnn(x2)
        skip2 = x2  # Skip connection
        x = self.down2(x2)  # [B, 128, H/4, W/4]
        
        # Stage 3: [B, 128, H/4, W/4]
        x3 = self.enc3_transformer(x)
        x3 = self.enc3_cnn(x3)
        skip3 = x3  # Skip connection
        x = self.down3(x3)  # [B, 256, H/8, W/8]


        # ============ Bottleneck: [B, 256, H/8, W/8] ============
        x = self.bottleneck_transformer(x)
        x = self.bottleneck_cnn(x)
        # ============ Decoder ============
        # Stage 3: [B, 256, H/8, W/8] → [B, 128, H/4, W/4]
        x = self.up3(x)
        x = x + skip3  # Skip connection
        x = self.dec3_transformer(x)
        x = self.dec3_cnn(x)
                
        # Stage 2: [B, 128, H/4, W/4] → [B, 64, H/2, W/2]
        x = self.up2(x)
        x = x + skip2  # Skip connection
        x = self.dec2_transformer(x)
        x = self.dec2_cnn(x)
        # Stage 1: [B, 64, H/2, W/2] → [B, 32, H, W]
        x = self.up1(x)
        x = x + skip1  # Skip connection
        x = self.dec1_transformer(x)
        x = self.dec1_cnn(x)
        # ============ Output ============
        output = self.output(x)  # [B, 3, H, W]
        
        return output[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
      
#################################################################################    
# Hybrid Split Network-----------------------------------------------------------
#################################################################################
class HybridSplitNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_heads=[1,1,1,1], ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(HybridSplitNet, self).__init__()
        
        # ============ Initial Convolution ============
        self.intro = nn.Conv2d(input_channels, base_channels, 3, 1, 1)
        
        # ============ Stage 1: 1개씩 → 2 outputs ============
        self.stage1_transformer1 = TransformerBlock(base_channels, num_heads[0], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage1_cnn1 = CNNBlock(base_channels)
        
        self.downsample1_t1 = Downsample(base_channels)
        self.downsample1_c1 = Downsample(base_channels)
        
        # ============ Stage 2: 2개씩 → 4 outputs ============
        self.stage2_transformer1 = TransformerBlock(base_channels, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage2_cnn1 = CNNBlock(base_channels)
        self.stage2_transformer2 = TransformerBlock(base_channels, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage2_cnn2 = CNNBlock(base_channels)
        
        self.downsample2_t1 = Downsample(base_channels)
        self.downsample2_c1 = Downsample(base_channels)
        self.downsample2_t2 = Downsample(base_channels)
        self.downsample2_c2 = Downsample(base_channels)
        
        # ============ Stage 3: 4개씩 → 8 outputs ============
        self.stage3_transformer1 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage3_cnn1 = CNNBlock(base_channels)
        self.stage3_transformer2 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage3_cnn2 = CNNBlock(base_channels)
        self.stage3_transformer3 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage3_cnn3 = CNNBlock(base_channels)
        self.stage3_transformer4 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.stage3_cnn4 = CNNBlock(base_channels)
        
        self.downsample3_t1 = Downsample(base_channels)
        self.downsample3_c1 = Downsample(base_channels)
        self.downsample3_t2 = Downsample(base_channels)
        self.downsample3_c2 = Downsample(base_channels)
        self.downsample3_t3 = Downsample(base_channels)
        self.downsample3_c3 = Downsample(base_channels)
        self.downsample3_t4 = Downsample(base_channels)
        self.downsample3_c4 = Downsample(base_channels)
        
        # ============ Bottleneck: 8개씩 → 16 outputs  ============
        self.bottleneck_transformer1 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn1 = CNNBlock(base_channels)
        self.bottleneck_transformer2 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn2 = CNNBlock(base_channels)
        self.bottleneck_transformer3 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn3 = CNNBlock(base_channels)
        self.bottleneck_transformer4 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn4 = CNNBlock(base_channels)
        self.bottleneck_transformer5 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn5 = CNNBlock(base_channels)
        self.bottleneck_transformer6 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn6 = CNNBlock(base_channels)
        self.bottleneck_transformer7 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn7 = CNNBlock(base_channels)
        self.bottleneck_transformer8 = TransformerBlock(base_channels, num_heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn8 = CNNBlock(base_channels)
        
        # ============ Decoder Stage 3: 8개씩 ============
        self.upsample3_t1 = Upsample(base_channels*2)
        self.upsample3_c1 = Upsample(base_channels*2)
        self.upsample3_t2 = Upsample(base_channels*2)
        self.upsample3_c2 = Upsample(base_channels*2)
        self.upsample3_t3 = Upsample(base_channels*2)
        self.upsample3_c3 = Upsample(base_channels*2)
        self.upsample3_t4 = Upsample(base_channels*2)
        self.upsample3_c4 = Upsample(base_channels*2)
        
        self.dec3_transformer1 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec3_cnn1 = CNNBlock(base_channels)
        self.dec3_transformer2 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec3_cnn2 = CNNBlock(base_channels)
        self.dec3_transformer3 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec3_cnn3 = CNNBlock(base_channels)
        self.dec3_transformer4 = TransformerBlock(base_channels, num_heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec3_cnn4 = CNNBlock(base_channels)
        
        # ============ Decoder Stage 2: 4개씩 ============
        self.upsample2_t1 = Upsample(base_channels*2)
        self.upsample2_c1 = Upsample(base_channels*2)
        self.upsample2_t2 = Upsample(base_channels*2)
        self.upsample2_c2 = Upsample(base_channels*2)
        
        self.dec2_transformer1 = TransformerBlock(base_channels, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec2_cnn1 = CNNBlock(base_channels)
        self.dec2_transformer2 = TransformerBlock(base_channels, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec2_cnn2 = CNNBlock(base_channels)
        
        # ============ Decoder Stage 1: 2개씩 ============
        self.upsample1_t1 = Upsample(base_channels*2)
        self.upsample1_c1 = Upsample(base_channels*2)
        
        self.dec1_transformer1 = TransformerBlock(base_channels, num_heads[0], ffn_expansion_factor, bias, LayerNorm_type)
        self.dec1_cnn1 = CNNBlock(base_channels)
        
        # ============ Output ============
        self.output = nn.Conv2d(base_channels * 2, input_channels, 3, 1, 1)
        
        self.padder_size = 2**4
        
    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
        
        # ============ Initial ============
        x = self.intro(inp)  # [B, 32, H, W]
   
        # ============ Stage 1 ============
        t1 = self.stage1_transformer1(x)  # [32]
        c1 = self.stage1_cnn1(x)          # [32]
        
        skip1_t, skip1_c = t1, c1
        
        t1 = self.downsample1_t1(t1)  # [64, H/2, W/2]
        c1 = self.downsample1_c1(c1)  # [64, H/2, W/2]
        
        
        t1_1,t1_2 = t1.chunk(2, dim=1) # [32, H/2, W/2]
        c1_1,c1_2= c1.chunk(2,dim=1)   # [32, H/2, W/2]
        
        # ============ Stage 2 ============
        t2_1 = self.stage2_transformer1(t1_1)  # [32]
        t2_2 = self.stage2_transformer2(t1_2)  # [32]
        c2_1 = self.stage2_cnn1(c1_1)          # [32]
        c2_2 = self.stage2_cnn2(c1_2)          # [32]
        
        skip2_t1, skip2_t2 = t2_1, t2_2
        skip2_c1, skip2_c2 = c2_1, c2_2
        
        t2_1 = self.downsample2_t1(t2_1)  # [64, H/4, W/4]
        c2_1 = self.downsample2_c1(c2_1)
        t2_2 = self.downsample2_t2(t2_2)
        c2_2 = self.downsample2_c2(c2_2)
           
        t3_1,t3_2 = t2_1.chunk(2, dim=1) # [32, H/4, W/4]
        t3_3,t3_4 = t2_2.chunk(2, dim=1)
        c3_1,c3_2 = c2_1.chunk(2, dim=1)
        c3_3,c3_4 = c2_2.chunk(2, dim=1)
        # ============ Stage 3 ============
        t3_1 = self.stage3_transformer1(t3_1)  # [B, 32, H/4, W/4]
        t3_2 = self.stage3_transformer2(t3_2) 
        t3_3 = self.stage3_transformer3(t3_3)  
        t3_4 = self.stage3_transformer4(t3_4)  
        
        c3_1 = self.stage3_cnn1(c3_1) 
        c3_2 = self.stage3_cnn2(c3_2)  
        c3_3 = self.stage3_cnn3(c3_3) 
        c3_4 = self.stage3_cnn4(c3_4)  
        
        skip3_t1 = t3_1
        skip3_t2 = t3_2
        skip3_t3 = t3_3
        skip3_t4 = t3_4
        skip3_c1 = c3_1
        skip3_c2 = c3_2
        skip3_c3 = c3_3
        skip3_c4 = c3_4
        
        t3_1 = self.downsample3_t1(t3_1)  # [B, 64, H/8, W/8]
        t3_2 = self.downsample3_t2(t3_2) 
        t3_3 = self.downsample3_t3(t3_3)  
        t3_4 = self.downsample3_t4(t3_4)  
        c3_1 = self.downsample3_c1(c3_1)  
        c3_2 = self.downsample3_c2(c3_2)  
        c3_3 = self.downsample3_c3(c3_3)  
        c3_4 = self.downsample3_c4(c3_4)  
        
        # Split for Bottleneck (8개)
        t4_1, t4_2 = t3_1.chunk(2, dim=1)  # [B, 32, H/8, W/8] each
        t4_3, t4_4 = t3_2.chunk(2, dim=1)  
        t4_5, t4_6 = t3_3.chunk(2, dim=1)  
        t4_7, t4_8 = t3_4.chunk(2, dim=1)  
        
        c4_1, c4_2 = c3_1.chunk(2, dim=1)  
        c4_3, c4_4 = c3_2.chunk(2, dim=1) 
        c4_5, c4_6 = c3_3.chunk(2, dim=1)  
        c4_7, c4_8 = c3_4.chunk(2, dim=1)  
        
        
        # ============ Bottleneck ============
        bt1 = self.bottleneck_transformer1(t4_1) # [B, 32, H/8, W/8]
        bt2 = self.bottleneck_transformer2(t4_2)
        bt3 = self.bottleneck_transformer3(t4_3)
        bt4 = self.bottleneck_transformer4(t4_4)
        bt5 = self.bottleneck_transformer5(t4_5)
        bt6 = self.bottleneck_transformer6(t4_6)
        bt7 = self.bottleneck_transformer7(t4_7)
        bt8 = self.bottleneck_transformer8(t4_8)
        
        bc1 = self.bottleneck_cnn1(c4_1)
        bc2 = self.bottleneck_cnn2(c4_2)
        bc3 = self.bottleneck_cnn3(c4_3)
        bc4 = self.bottleneck_cnn4(c4_4)
        bc5 = self.bottleneck_cnn5(c4_5)
        bc6 = self.bottleneck_cnn6(c4_6)
        bc7 = self.bottleneck_cnn7(c4_7)
        bc8 = self.bottleneck_cnn8(c4_8)
        
        # ============ Decoder Stage 3 ============
        
        # Concat pairs (8개 → 4개)
        dt3_1 = concat_tensor(bt1, bt2)  # [B, 64, H/8, W/8]
        dt3_2 = concat_tensor(bt3, bt4)
        dt3_3 = concat_tensor(bt5, bt6)
        dt3_4 = concat_tensor(bt7, bt8)
        
        dc3_1 = concat_tensor(bc1, bc2)
        dc3_2 = concat_tensor(bc3, bc4)
        dc3_3 = concat_tensor(bc5, bc6)
        dc3_4 = concat_tensor(bc7, bc8)
        # Upsample

        dt3_1 = self.upsample3_t1(dt3_1)  # [B, 32, H/4, W/4]
        dt3_2 = self.upsample3_t2(dt3_2)
        dt3_3 = self.upsample3_t3(dt3_3)
        dt3_4 = self.upsample3_t4(dt3_4)

        dc3_1 = self.upsample3_c1(dc3_1)
        dc3_2 = self.upsample3_c2(dc3_2)
        dc3_3 = self.upsample3_c3(dc3_3)
        dc3_4 = self.upsample3_c4(dc3_4)
        
        
        # Skip connection + process
        dt3_1 = dt3_1 + skip3_t1 # [B, 32, H/4, W/4]
        dt3_2 = dt3_2 + skip3_t2
        dt3_3 = dt3_3 + skip3_t3
        dt3_4 = dt3_4 + skip3_t4
        
        dc3_1 = dc3_1 + skip3_c1
        dc3_2 = dc3_2 + skip3_c2
        dc3_3 = dc3_3 + skip3_c3
        dc3_4 = dc3_4 + skip3_c4
        
        dt3_1 = self.dec3_transformer1(dt3_1)
        dt3_2 = self.dec3_transformer2(dt3_2)
        dt3_3 = self.dec3_transformer3(dt3_3)
        dt3_4 = self.dec3_transformer4(dt3_4)
        
        dc3_1 = self.dec3_cnn1(dc3_1)
        dc3_2 = self.dec3_cnn2(dc3_2)
        dc3_3 = self.dec3_cnn3(dc3_3)
        dc3_4 = self.dec3_cnn4(dc3_4)
        
        # ============ Decoder Stage 2 ============
        
        # Concat pairs (4개 → 2개)
        dt2_1 = concat_tensor(dt3_1, dt3_2)  # [B, 64, H/4, W/4]
        dt2_2 = concat_tensor(dt3_3, dt3_4)
        
        dc2_1 = concat_tensor(dc3_1, dc3_2)
        dc2_2 = concat_tensor(dc3_3, dc3_4)
        # Upsample
        dt2_1 = self.upsample2_t1(dt2_1)  # [B, 32, H/2, W/2]
        dt2_2 = self.upsample2_t2(dt2_2)
        dc2_1 = self.upsample2_c1(dc2_1)
        dc2_2 = self.upsample2_c2(dc2_2)

        
        # Skip connection + process
        dt2_1 = dt2_1 + skip2_t1
        dt2_2 = dt2_2 + skip2_t2
        dc2_1 = dc2_1 + skip2_c1
        dc2_2 = dc2_2 + skip2_c2
        
        
        dt2_1 = self.dec2_transformer1(dt2_1)
        dt2_2 = self.dec2_transformer2(dt2_2)
        
        dc2_1 = self.dec2_cnn1(dc2_1)
        dc2_2 = self.dec2_cnn2(dc2_2)
        
        # ============ Decoder Stage 1: 2개 → concat → 1개 ============
        
        # Concat pairs (2개 → 1개)
        dt1 = concat_tensor(dt2_1, dt2_2)  # [B, 64, H/2, W/2]
        dc1 = concat_tensor(dc2_1, dc2_2)
        
        # Upsample
        dt1 = self.upsample1_t1(dt1)  #[B, 32, H, W]
        
        dc1 = self.upsample1_c1(dc1)
        
        # Skip connection + process
        dt1 = dt1 + skip1_t
        dc1 = dc1 + skip1_c
        
        dt1 = self.dec1_transformer1(dt1)
        dc1 = self.dec1_cnn1(dc1)
        
        # ============ Output ============
        out = torch.cat([dt1, dc1], dim=1)  # [B, 64, H, W]
        output = self.output(out)  # [B, 3, H, W]
        
        return output[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class Light_HybridSplitNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Light_HybridSplitNet, self).__init__()
        
        # ============ Initial Convolution ============
        self.intro = nn.Conv2d(input_channels, base_channels, 3, 1, 1)
        
        # ============ Stage 1: 1개씩 → 2 outputs ============
        self.stage1_transformer1 = SCA_CNNBlock(base_channels)
        self.stage1_cnn1 = SCA_CNNBlock(base_channels)
        
        self.downsample1_t1 = Downsample(base_channels)
        self.downsample1_c1 = Downsample(base_channels)
        
        # ============ Stage 2: 2개씩 → 4 outputs ============
        self.stage2_transformer1 = SCA_CNNBlock(base_channels)
        self.stage2_cnn1 = SCA_CNNBlock(base_channels)
        self.stage2_transformer2 = SCA_CNNBlock(base_channels)
        self.stage2_cnn2 = SCA_CNNBlock(base_channels)
        
        self.downsample2_t1 = Downsample(base_channels)
        self.downsample2_c1 = Downsample(base_channels)
        self.downsample2_t2 = Downsample(base_channels)
        self.downsample2_c2 = Downsample(base_channels)
        
        # ============ Stage 3: 4개씩 → 8 outputs ============
        self.stage3_transformer1 = SCA_CNNBlock(base_channels)
        self.stage3_cnn1 = SCA_CNNBlock(base_channels)
        self.stage3_transformer2 = SCA_CNNBlock(base_channels)
        self.stage3_cnn2 = SCA_CNNBlock(base_channels)
        self.stage3_transformer3 = SCA_CNNBlock(base_channels)
        self.stage3_cnn3 = SCA_CNNBlock(base_channels)
        self.stage3_transformer4 = SCA_CNNBlock(base_channels)
        self.stage3_cnn4 = SCA_CNNBlock(base_channels)
        
        self.downsample3_t1 = Downsample(base_channels)
        self.downsample3_c1 = Downsample(base_channels)
        self.downsample3_t2 = Downsample(base_channels)
        self.downsample3_c2 = Downsample(base_channels)
        self.downsample3_t3 = Downsample(base_channels)
        self.downsample3_c3 = Downsample(base_channels)
        self.downsample3_t4 = Downsample(base_channels)
        self.downsample3_c4 = Downsample(base_channels)
        
        # ============ Bottleneck: 8개씩 → 16 outputs  ============
        self.bottleneck_transformer1 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn1 = CNNBlock(base_channels)
        self.bottleneck_transformer2 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn2 = CNNBlock(base_channels)
        self.bottleneck_transformer3 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn3 = CNNBlock(base_channels)
        self.bottleneck_transformer4 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn4 = CNNBlock(base_channels)
        self.bottleneck_transformer5 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn5 = CNNBlock(base_channels)
        self.bottleneck_transformer6 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn6 = CNNBlock(base_channels)
        self.bottleneck_transformer7 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn7 = CNNBlock(base_channels)
        self.bottleneck_transformer8 = TransformerBlock(base_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck_cnn8 = CNNBlock(base_channels)
        
        # ============ Decoder Stage 3: 8개씩 ============
        self.upsample3_t1 = Upsample(base_channels*2)
        self.upsample3_c1 = Upsample(base_channels*2)
        self.upsample3_t2 = Upsample(base_channels*2)
        self.upsample3_c2 = Upsample(base_channels*2)
        self.upsample3_t3 = Upsample(base_channels*2)
        self.upsample3_c3 = Upsample(base_channels*2)
        self.upsample3_t4 = Upsample(base_channels*2)
        self.upsample3_c4 = Upsample(base_channels*2)
        
        self.dec3_transformer1 = SCA_CNNBlock(base_channels)
        self.dec3_cnn1 = SCA_CNNBlock(base_channels)
        self.dec3_transformer2 = SCA_CNNBlock(base_channels)
        self.dec3_cnn2 = SCA_CNNBlock(base_channels)
        self.dec3_transformer3 = SCA_CNNBlock(base_channels)
        self.dec3_cnn3 = SCA_CNNBlock(base_channels)
        self.dec3_transformer4 = SCA_CNNBlock(base_channels)
        self.dec3_cnn4 = SCA_CNNBlock(base_channels)
        
        # ============ Decoder Stage 2: 4개씩 ============
        self.upsample2_t1 = Upsample(base_channels*2)
        self.upsample2_c1 = Upsample(base_channels*2)
        self.upsample2_t2 = Upsample(base_channels*2)
        self.upsample2_c2 = Upsample(base_channels*2)
        
        self.dec2_transformer1 = SCA_CNNBlock(base_channels)
        self.dec2_cnn1 = SCA_CNNBlock(base_channels)
        self.dec2_transformer2 = SCA_CNNBlock(base_channels)
        self.dec2_cnn2 = SCA_CNNBlock(base_channels)
        
        # ============ Decoder Stage 1: 2개씩 ============
        self.upsample1_t1 = Upsample(base_channels*2)
        self.upsample1_c1 = Upsample(base_channels*2)
        
        self.dec1_transformer1 = SCA_CNNBlock(base_channels)
        self.dec1_cnn1 = SCA_CNNBlock(base_channels)
        
        # ============ Output ============
        self.output = nn.Conv2d(base_channels * 2, input_channels, 3, 1, 1)
        
        self.padder_size = 2**4
        
    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
        
        # ============ Initial ============
        x = self.intro(inp)  # [B, 32, H, W]
   
        # ============ Stage 1 ============
        t1 = self.stage1_transformer1(x)  # [32]
        c1 = self.stage1_cnn1(x)          # [32]
        
        skip1_t, skip1_c = t1, c1
        
        t1 = self.downsample1_t1(t1)  # [32, H/2, W/2]
        c1 = self.downsample1_c1(c1)  # [32, H/2, W/2]
        
        
        t1_1,t1_2 = t1.chunk(2, dim=1)
        c1_1,c1_2= c1.chunk(2,dim=1)   
        
        # ============ Stage 2 ============
        t2_1 = self.stage2_transformer1(t1_1)  # [32]
        t2_2 = self.stage2_transformer2(t1_2)  # [32]
        c2_1 = self.stage2_cnn1(c1_1)          # [32]
        c2_2 = self.stage2_cnn2(c1_2)          # [32]
        
        skip2_t1, skip2_t2 = t2_1, t2_2
        skip2_c1, skip2_c2 = c2_1, c2_2
        
        t2_1 = self.downsample2_t1(t2_1)  # [32, H/4, W/4]
        c2_1 = self.downsample2_c1(c2_1)
        t2_2 = self.downsample2_t2(t2_2)
        c2_2 = self.downsample2_c2(c2_2)
           
        t3_1,t3_2 = t2_1.chunk(2, dim=1)
        t3_3,t3_4 = t2_2.chunk(2, dim=1)
        c3_1,c3_2 = c2_1.chunk(2, dim=1)
        c3_3,c3_4 = c2_2.chunk(2, dim=1)
        # ============ Stage 3 ============
        t3_1 = self.stage3_transformer1(t3_1)  # [B, 32, H/4, W/8]
        t3_2 = self.stage3_transformer2(t3_2)  # [B, 32, H/4, W/8]
        t3_3 = self.stage3_transformer3(t3_3)  # [B, 32, H/4, W/8]
        t3_4 = self.stage3_transformer4(t3_4)  # [B, 32, H/4, W/8]
        
        c3_1 = self.stage3_cnn1(c3_1)  # [B, 32, H/4, W/8]
        c3_2 = self.stage3_cnn2(c3_2)  # [B, 32, H/4, W/8]
        c3_3 = self.stage3_cnn3(c3_3)  # [B, 32, H/4, W/8]
        c3_4 = self.stage3_cnn4(c3_4)  # [B, 32, H/4, W/8]
        
        skip3_t1 = t3_1
        skip3_t2 = t3_2
        skip3_t3 = t3_3
        skip3_t4 = t3_4
        skip3_c1 = c3_1
        skip3_c2 = c3_2
        skip3_c3 = c3_3
        skip3_c4 = c3_4
        
        t3_1 = self.downsample3_t1(t3_1)  # [B, 32, H/8, W/8]
        t3_2 = self.downsample3_t2(t3_2)  # [B, 32, H/8, W/8]
        t3_3 = self.downsample3_t3(t3_3)  # [B, 32, H/8, W/8]
        t3_4 = self.downsample3_t4(t3_4)  # [B, 32, H/8, W/8]
        c3_1 = self.downsample3_c1(c3_1)  # [B, 32, H/8, W/8]
        c3_2 = self.downsample3_c2(c3_2)  # [B, 32, H/8, W/8]
        c3_3 = self.downsample3_c3(c3_3)  # [B, 32, H/8, W/8]
        c3_4 = self.downsample3_c4(c3_4)  # [B, 32, H/8, W/8]
        
        # Split for Bottleneck (8개)
        t4_1, t4_2 = t3_1.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_3, t4_4 = t3_2.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_5, t4_6 = t3_3.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_7, t4_8 = t3_4.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        
        c4_1, c4_2 = c3_1.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_3, c4_4 = c3_2.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_5, c4_6 = c3_3.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_7, c4_8 = c3_4.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        
        
        # ============ Bottleneck ============
        bt1 = self.bottleneck_transformer1(t4_1)
        bt2 = self.bottleneck_transformer2(t4_2)
        bt3 = self.bottleneck_transformer3(t4_3)
        bt4 = self.bottleneck_transformer4(t4_4)
        bt5 = self.bottleneck_transformer5(t4_5)
        bt6 = self.bottleneck_transformer6(t4_6)
        bt7 = self.bottleneck_transformer7(t4_7)
        bt8 = self.bottleneck_transformer8(t4_8)
        
        bc1 = self.bottleneck_cnn1(c4_1)
        bc2 = self.bottleneck_cnn2(c4_2)
        bc3 = self.bottleneck_cnn3(c4_3)
        bc4 = self.bottleneck_cnn4(c4_4)
        bc5 = self.bottleneck_cnn5(c4_5)
        bc6 = self.bottleneck_cnn6(c4_6)
        bc7 = self.bottleneck_cnn7(c4_7)
        bc8 = self.bottleneck_cnn8(c4_8)
        
        # ============ Decoder Stage 3 ============
        
        # Concat pairs (8개 → 4개)
        dt3_1 = concat_tensor(bt1, bt2)  # [B, 32, H/4, W/8]
        dt3_2 = concat_tensor(bt3, bt4)
        dt3_3 = concat_tensor(bt5, bt6)
        dt3_4 = concat_tensor(bt7, bt8)
        
        dc3_1 = concat_tensor(bc1, bc2)
        dc3_2 = concat_tensor(bc3, bc4)
        dc3_3 = concat_tensor(bc5, bc6)
        dc3_4 = concat_tensor(bc7, bc8)
        # Upsample

        dt3_1 = self.upsample3_t1(dt3_1)  # [B, 32, H/4, W/16]
        dt3_2 = self.upsample3_t2(dt3_2)
        dt3_3 = self.upsample3_t3(dt3_3)
        dt3_4 = self.upsample3_t4(dt3_4)

        dc3_1 = self.upsample3_c1(dc3_1)
        dc3_2 = self.upsample3_c2(dc3_2)
        dc3_3 = self.upsample3_c3(dc3_3)
        dc3_4 = self.upsample3_c4(dc3_4)
        
        
        # Skip connection + process
        dt3_1 = dt3_1 + skip3_t1
        dt3_2 = dt3_2 + skip3_t2
        dt3_3 = dt3_3 + skip3_t3
        dt3_4 = dt3_4 + skip3_t4
        
        dc3_1 = dc3_1 + skip3_c1
        dc3_2 = dc3_2 + skip3_c2
        dc3_3 = dc3_3 + skip3_c3
        dc3_4 = dc3_4 + skip3_c4
        
        dt3_1 = self.dec3_transformer1(dt3_1)
        dt3_2 = self.dec3_transformer2(dt3_2)
        dt3_3 = self.dec3_transformer3(dt3_3)
        dt3_4 = self.dec3_transformer4(dt3_4)
        
        dc3_1 = self.dec3_cnn1(dc3_1)
        dc3_2 = self.dec3_cnn2(dc3_2)
        dc3_3 = self.dec3_cnn3(dc3_3)
        dc3_4 = self.dec3_cnn4(dc3_4)
        
        # ============ Decoder Stage 2 ============
        
        # Concat pairs (4개 → 2개)
        dt2_1 = concat_tensor(dt3_1, dt3_2)  # [B, 32, H/2, W/4]
        dt2_2 = concat_tensor(dt3_3, dt3_4)
        
        dc2_1 = concat_tensor(dc3_1, dc3_2)
        dc2_2 = concat_tensor(dc3_3, dc3_4)
        # Upsample
        dt2_1 = self.upsample2_t1(dt2_1)  # [B, 32, H/2, W/8]
        dt2_2 = self.upsample2_t2(dt2_2)
        dc2_1 = self.upsample2_c1(dc2_1)
        dc2_2 = self.upsample2_c2(dc2_2)

        
        # Skip connection + process
        dt2_1 = dt2_1 + skip2_t1
        dt2_2 = dt2_2 + skip2_t2
        dc2_1 = dc2_1 + skip2_c1
        dc2_2 = dc2_2 + skip2_c2
        
        
        dt2_1 = self.dec2_transformer1(dt2_1)
        dt2_2 = self.dec2_transformer2(dt2_2)
        
        dc2_1 = self.dec2_cnn1(dc2_1)
        dc2_2 = self.dec2_cnn2(dc2_2)
        
        # ============ Decoder Stage 1: 2개 → concat → 1개 ============
        
        # Concat pairs (2개 → 1개)
        dt1 = concat_tensor(dt2_1, dt2_2)  # [B, 32, H, W/2]
        dc1 = concat_tensor(dc2_1, dc2_2)
        
        # Upsample
        dt1 = self.upsample1_t1(dt1)  # [B, 32, H, W/4]
        
        dc1 = self.upsample1_c1(dc1)
        
        # Skip connection + process
        dt1 = dt1 + skip1_t
        dc1 = dc1 + skip1_c
        
        dt1 = self.dec1_transformer1(dt1)
        dc1 = self.dec1_cnn1(dc1)
        
        # ============ Output ============
        out = torch.cat([dt1, dc1], dim=1)  # [B, 64, H, W]
        output = self.output(out)  # [B, 3, H, W]
        
        return output[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

##########################################################################################
# Only CNN--------------------------------------------------------------------------------
##########################################################################################
# CNNBlock -> SCA_CNNBlock
class CNNSplitNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super(CNNSplitNet, self).__init__()
        
        # ============ Initial Convolution ============
        self.intro = nn.Conv2d(input_channels, base_channels, 3, 1, 1)
        
        # ============ Stage 1: 1개씩 → 2 outputs (num_heads[0]) ============
        self.stage1_transformer1 = SCA_CNNBlock(base_channels)
        self.stage1_cnn1 = SCA_CNNBlock(base_channels)
        
        self.downsample1_t1 = Downsample(base_channels)
        self.downsample1_c1 = Downsample(base_channels)
        
        # ============ Stage 2: 2개씩 → 4 outputs (num_heads[1]) ============
        self.stage2_transformer1 = SCA_CNNBlock(base_channels)
        self.stage2_cnn1 = SCA_CNNBlock(base_channels)
        self.stage2_transformer2 = SCA_CNNBlock(base_channels)
        self.stage2_cnn2 = SCA_CNNBlock(base_channels)
        
        self.downsample2_t1 = Downsample(base_channels)
        self.downsample2_c1 = Downsample(base_channels)
        self.downsample2_t2 = Downsample(base_channels)
        self.downsample2_c2 = Downsample(base_channels)
        
        # ============ Stage 3: 4개씩 → 8 outputs (num_heads[2]) ============
        self.stage3_transformer1 = SCA_CNNBlock(base_channels)
        self.stage3_cnn1 = SCA_CNNBlock(base_channels)
        self.stage3_transformer2 = SCA_CNNBlock(base_channels)
        self.stage3_cnn2 = SCA_CNNBlock(base_channels)
        self.stage3_transformer3 = SCA_CNNBlock(base_channels)
        self.stage3_cnn3 = SCA_CNNBlock(base_channels)
        self.stage3_transformer4 = SCA_CNNBlock(base_channels)
        self.stage3_cnn4 = SCA_CNNBlock(base_channels)
        
        self.downsample3_t1 = Downsample(base_channels)
        self.downsample3_c1 = Downsample(base_channels)
        self.downsample3_t2 = Downsample(base_channels)
        self.downsample3_c2 = Downsample(base_channels)
        self.downsample3_t3 = Downsample(base_channels)
        self.downsample3_c3 = Downsample(base_channels)
        self.downsample3_t4 = Downsample(base_channels)
        self.downsample3_c4 = Downsample(base_channels)
        
        # ============ Bottleneck: 8개씩 → 16 outputs (num_heads[3]) ============
        self.bottleneck_transformer1 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn1 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer2 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn2 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer3 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn3 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer4 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn4 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer5 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn5 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer6 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn6 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer7 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn7 = SCA_CNNBlock(base_channels)
        self.bottleneck_transformer8 = SCA_CNNBlock(base_channels)
        self.bottleneck_cnn8 = SCA_CNNBlock(base_channels)
        
        # ============ Decoder Stage 3: 8개씩 (num_heads[2]) ============
        self.upsample3_t1 = Upsample(base_channels*2)
        self.upsample3_c1 = Upsample(base_channels*2)
        self.upsample3_t2 = Upsample(base_channels*2)
        self.upsample3_c2 = Upsample(base_channels*2)
        self.upsample3_t3 = Upsample(base_channels*2)
        self.upsample3_c3 = Upsample(base_channels*2)
        self.upsample3_t4 = Upsample(base_channels*2)
        self.upsample3_c4 = Upsample(base_channels*2)
        
        self.dec3_transformer1 = SCA_CNNBlock(base_channels)
        self.dec3_cnn1 = SCA_CNNBlock(base_channels)
        self.dec3_transformer2 = SCA_CNNBlock(base_channels)
        self.dec3_cnn2 = SCA_CNNBlock(base_channels)
        self.dec3_transformer3 = SCA_CNNBlock(base_channels)
        self.dec3_cnn3 = SCA_CNNBlock(base_channels)
        self.dec3_transformer4 = SCA_CNNBlock(base_channels)
        self.dec3_cnn4 = SCA_CNNBlock(base_channels)
        
        # ============ Decoder Stage 2: 4개씩 (num_heads[1]) ============
        self.upsample2_t1 = Upsample(base_channels*2)
        self.upsample2_c1 = Upsample(base_channels*2)
        self.upsample2_t2 = Upsample(base_channels*2)
        self.upsample2_c2 = Upsample(base_channels*2)
        
        self.dec2_transformer1 = SCA_CNNBlock(base_channels)
        self.dec2_cnn1 = SCA_CNNBlock(base_channels)
        self.dec2_transformer2 = SCA_CNNBlock(base_channels)
        self.dec2_cnn2 = SCA_CNNBlock(base_channels)
        
        # ============ Decoder Stage 1: 2개씩 (num_heads[0]) ============
        self.upsample1_t1 = Upsample(base_channels*2)
        self.upsample1_c1 = Upsample(base_channels*2)
        
        self.dec1_transformer1 = SCA_CNNBlock(base_channels)
        self.dec1_cnn1 = SCA_CNNBlock(base_channels)
        
        # ============ Output ============
        self.output = nn.Conv2d(base_channels * 2, input_channels, 3, 1, 1)
        
        self.padder_size = 2**4
        
    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
        
        # ============ Initial ============
        x = self.intro(inp)  # [B, 32, H, W]
   
        # ============ Stage 1 ============
        t1 = self.stage1_transformer1(x)  # [32]
        c1 = self.stage1_cnn1(x)          # [32]
        
        skip1_t, skip1_c = t1, c1
        
        t1 = self.downsample1_t1(t1)  # [32, H/2, W/2]
        c1 = self.downsample1_c1(c1)  # [32, H/2, W/2]
        
        
        t1_1,t1_2 = t1.chunk(2, dim=1)
        c1_1,c1_2= c1.chunk(2,dim=1)   
        
        # ============ Stage 2 ============
        t2_1 = self.stage2_transformer1(t1_1)  # [32]
        t2_2 = self.stage2_transformer2(t1_2)  # [32]
        c2_1 = self.stage2_cnn1(c1_1)          # [32]
        c2_2 = self.stage2_cnn2(c1_2)          # [32]
        
        skip2_t1, skip2_t2 = t2_1, t2_2
        skip2_c1, skip2_c2 = c2_1, c2_2
        
        t2_1 = self.downsample2_t1(t2_1)  # [32, H/4, W/4]
        c2_1 = self.downsample2_c1(c2_1)
        t2_2 = self.downsample2_t2(t2_2)
        c2_2 = self.downsample2_c2(c2_2)
           
        t3_1,t3_2 = t2_1.chunk(2, dim=1)
        t3_3,t3_4 = t2_2.chunk(2, dim=1)
        c3_1,c3_2 = c2_1.chunk(2, dim=1)
        c3_3,c3_4 = c2_2.chunk(2, dim=1)
        # ============ Stage 3 ============
        t3_1 = self.stage3_transformer1(t3_1)  # [B, 32, H/4, W/8]
        t3_2 = self.stage3_transformer2(t3_2)  # [B, 32, H/4, W/8]
        t3_3 = self.stage3_transformer3(t3_3)  # [B, 32, H/4, W/8]
        t3_4 = self.stage3_transformer4(t3_4)  # [B, 32, H/4, W/8]
        
        c3_1 = self.stage3_cnn1(c3_1)  # [B, 32, H/4, W/8]
        c3_2 = self.stage3_cnn2(c3_2)  # [B, 32, H/4, W/8]
        c3_3 = self.stage3_cnn3(c3_3)  # [B, 32, H/4, W/8]
        c3_4 = self.stage3_cnn4(c3_4)  # [B, 32, H/4, W/8]
        
        skip3_t1 = t3_1
        skip3_t2 = t3_2
        skip3_t3 = t3_3
        skip3_t4 = t3_4
        skip3_c1 = c3_1
        skip3_c2 = c3_2
        skip3_c3 = c3_3
        skip3_c4 = c3_4
        
        t3_1 = self.downsample3_t1(t3_1)  # [B, 32, H/8, W/8]
        t3_2 = self.downsample3_t2(t3_2)  # [B, 32, H/8, W/8]
        t3_3 = self.downsample3_t3(t3_3)  # [B, 32, H/8, W/8]
        t3_4 = self.downsample3_t4(t3_4)  # [B, 32, H/8, W/8]
        c3_1 = self.downsample3_c1(c3_1)  # [B, 32, H/8, W/8]
        c3_2 = self.downsample3_c2(c3_2)  # [B, 32, H/8, W/8]
        c3_3 = self.downsample3_c3(c3_3)  # [B, 32, H/8, W/8]
        c3_4 = self.downsample3_c4(c3_4)  # [B, 32, H/8, W/8]
        
        # Split for Bottleneck (8개)
        t4_1, t4_2 = t3_1.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_3, t4_4 = t3_2.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_5, t4_6 = t3_3.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        t4_7, t4_8 = t3_4.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        
        c4_1, c4_2 = c3_1.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_3, c4_4 = c3_2.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_5, c4_6 = c3_3.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        c4_7, c4_8 = c3_4.chunk(2, dim=1)  # [B, 32, H/8, W/16] each
        
        
        # ============ Bottleneck ============
        bt1 = self.bottleneck_transformer1(t4_1)
        bt2 = self.bottleneck_transformer2(t4_2)
        bt3 = self.bottleneck_transformer3(t4_3)
        bt4 = self.bottleneck_transformer4(t4_4)
        bt5 = self.bottleneck_transformer5(t4_5)
        bt6 = self.bottleneck_transformer6(t4_6)
        bt7 = self.bottleneck_transformer7(t4_7)
        bt8 = self.bottleneck_transformer8(t4_8)
        
        bc1 = self.bottleneck_cnn1(c4_1)
        bc2 = self.bottleneck_cnn2(c4_2)
        bc3 = self.bottleneck_cnn3(c4_3)
        bc4 = self.bottleneck_cnn4(c4_4)
        bc5 = self.bottleneck_cnn5(c4_5)
        bc6 = self.bottleneck_cnn6(c4_6)
        bc7 = self.bottleneck_cnn7(c4_7)
        bc8 = self.bottleneck_cnn8(c4_8)
        
        # ============ Decoder Stage 3 ============
        
        # Concat pairs (8개 → 4개)
        dt3_1 = concat_tensor(bt1, bt2)  # [B, 32, H/4, W/8]
        dt3_2 = concat_tensor(bt3, bt4)
        dt3_3 = concat_tensor(bt5, bt6)
        dt3_4 = concat_tensor(bt7, bt8)
        
        dc3_1 = concat_tensor(bc1, bc2)
        dc3_2 = concat_tensor(bc3, bc4)
        dc3_3 = concat_tensor(bc5, bc6)
        dc3_4 = concat_tensor(bc7, bc8)
        # Upsample

        dt3_1 = self.upsample3_t1(dt3_1)  # [B, 32, H/4, W/16]
        dt3_2 = self.upsample3_t2(dt3_2)
        dt3_3 = self.upsample3_t3(dt3_3)
        dt3_4 = self.upsample3_t4(dt3_4)

        dc3_1 = self.upsample3_c1(dc3_1)
        dc3_2 = self.upsample3_c2(dc3_2)
        dc3_3 = self.upsample3_c3(dc3_3)
        dc3_4 = self.upsample3_c4(dc3_4)
        
        
        # Skip connection + process
        dt3_1 = dt3_1 + skip3_t1
        dt3_2 = dt3_2 + skip3_t2
        dt3_3 = dt3_3 + skip3_t3
        dt3_4 = dt3_4 + skip3_t4
        
        dc3_1 = dc3_1 + skip3_c1
        dc3_2 = dc3_2 + skip3_c2
        dc3_3 = dc3_3 + skip3_c3
        dc3_4 = dc3_4 + skip3_c4
        
        dt3_1 = self.dec3_transformer1(dt3_1)
        dt3_2 = self.dec3_transformer2(dt3_2)
        dt3_3 = self.dec3_transformer3(dt3_3)
        dt3_4 = self.dec3_transformer4(dt3_4)
        
        dc3_1 = self.dec3_cnn1(dc3_1)
        dc3_2 = self.dec3_cnn2(dc3_2)
        dc3_3 = self.dec3_cnn3(dc3_3)
        dc3_4 = self.dec3_cnn4(dc3_4)
        
        # ============ Decoder Stage 2 ============
        
        # Concat pairs (4개 → 2개)
        dt2_1 = concat_tensor(dt3_1, dt3_2)  # [B, 32, H/2, W/4]
        dt2_2 = concat_tensor(dt3_3, dt3_4)
        
        dc2_1 = concat_tensor(dc3_1, dc3_2)
        dc2_2 = concat_tensor(dc3_3, dc3_4)
        # Upsample
        dt2_1 = self.upsample2_t1(dt2_1)  # [B, 32, H/2, W/8]
        dt2_2 = self.upsample2_t2(dt2_2)
        dc2_1 = self.upsample2_c1(dc2_1)
        dc2_2 = self.upsample2_c2(dc2_2)

        
        # Skip connection + process
        dt2_1 = dt2_1 + skip2_t1
        dt2_2 = dt2_2 + skip2_t2
        dc2_1 = dc2_1 + skip2_c1
        dc2_2 = dc2_2 + skip2_c2
        
        
        dt2_1 = self.dec2_transformer1(dt2_1)
        dt2_2 = self.dec2_transformer2(dt2_2)
        
        dc2_1 = self.dec2_cnn1(dc2_1)
        dc2_2 = self.dec2_cnn2(dc2_2)
        
        # ============ Decoder Stage 1: 2개 → concat → 1개 ============
        
        # Concat pairs (2개 → 1개)
        dt1 = concat_tensor(dt2_1, dt2_2)  # [B, 32, H, W/2]
        dc1 = concat_tensor(dc2_1, dc2_2)
        
        # Upsample
        dt1 = self.upsample1_t1(dt1)  # [B, 32, H, W/4]
        
        dc1 = self.upsample1_c1(dc1)
        
        # Skip connection + process
        dt1 = dt1 + skip1_t
        dc1 = dc1 + skip1_c
        
        dt1 = self.dec1_transformer1(dt1)
        dc1 = self.dec1_cnn1(dc1)
        
        # ============ Output ============
        out = torch.cat([dt1, dc1], dim=1)  # [B, 64, H, W]
        output = self.output(out)  # [B, 3, H, W]
        
        return output[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x    



#Test model
if __name__ == '__main__':
    img_channel = 3
    width = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom = Light_HybridSplitNet()
    custom.to(device,dtype = torch.float32)
    
    
    #Model Summary
    
    #torchsummary.summary(custom,(3,256,256))
    
    #Inference Time
    # GPU 측정
    input_tensor = torch.randn(1, 3, 1000, 1000).to(device, dtype = torch.float32)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = custom(input_tensor)

    # CUDA를 사용한 inference time 측정
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    num_iterations = 100
    timings = []
    with torch.no_grad():
        starter.record()
        output = custom(input_tensor)
        ender.record()
        # GPU 동기화
        
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)  # milliseconds


    print(f"Inference time (GPU): {curr_time:.2f} ms")
    

    # Model Complexity
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(custom, (3,1000,1000), verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(f"Custom MACS: {macs}, PARAMS:{params}")


    
