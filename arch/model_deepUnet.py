import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensor):
        avg_pool = torch.mean(input_tensor, dim=1, keepdim=True)
        max_pool, _ = torch.max(input_tensor, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        concat = F.pad(concat, (1, 1, 1, 1), mode='reflect')
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        return input_tensor * attention

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super(SimplifiedChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, inputs):
        feature_descriptor = self.global_avg_pool(inputs)
        features = self.conv(feature_descriptor)
        return inputs * features

class MidBlock(nn.Module):
    def __init__(self, filters):
        super(MidBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.chan_att = SimplifiedChannelAttention(filters)
        
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.spatial_att = SpatialAttention()
    
    def forward(self, input_tensor):
        net = F.pad(input_tensor, (1, 1, 1, 1), mode='reflect')
        net = self.conv1(net)
        net = self.leaky_relu1(net)
        net = self.chan_att(net)
        net2 = net + input_tensor
        
        net = F.pad(net2, (1, 1, 1, 1), mode='reflect')
        net = self.conv2(net)
        net = self.leaky_relu2(net)
        net = self.spatial_att(net)
        return net + net2

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        return x + skip

class SimpleUNet(nn.Module):
    def __init__(self, input_channels=3, num_filters=32):
        super(SimpleUNet, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        
        self.padder_size = 2**4
        
        
        # Encoder
        self.enc1 = EncoderBlock(num_filters, num_filters*2)      # 256 -> 128
        self.enc2 = EncoderBlock(num_filters*2, num_filters*4)      # 128 -> 64
        self.enc3 = EncoderBlock(num_filters*4, num_filters*8)      # 64 -> 32
        self.enc4 = EncoderBlock(num_filters*8, num_filters*16)      # 32 -> 16
        
        # Middle (bottleneck)
        self.mid_block = MidBlock(num_filters*16)
        # Decoder
        self.dec4 = DecoderBlock(num_filters*16, num_filters*8)      # 16 -> 32
        self.dec3 = DecoderBlock(num_filters*8, num_filters*4)      # 32 -> 64
        self.dec2 = DecoderBlock(num_filters*4, num_filters*2)      # 64 -> 128
        self.dec1 = DecoderBlock(num_filters*2, num_filters)      # 128 -> 256
        
        # Final convolution
        self.final_conv = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)
    
    def forward(self, input_tensor):
        
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
        x = self.initial_conv(inp)
        
        
        # Initial
        x0 = self.initial_conv(inp)
        
        # Encoder
        x1 = self.enc1(x0)   # skip1
        x2 = self.enc2(x1)   # skip2
        x3 = self.enc3(x2)   # skip3
        x4 = self.enc4(x3)   # skip4
        
        # Middle
        x = self.mid_block(x4)
        
        # Decoder with skip connections
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        
        # Final output
        x = self.final_conv(x)
        
        output = inp+x
        
        return output[:,:,:H,:W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# DeepUNet - 수정된 버전
class DeepUNet(nn.Module):
    def __init__(self, input_channels=3, base_filters=32):
        super(DeepUNet, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1)
        
        # Encoder - 4 stages
        # Stage 1: 256x256 -> 128x128 (2 branches)
        self.enc1_1 = EncoderBlock(base_filters, base_filters)
        self.enc1_2 = EncoderBlock(base_filters, base_filters)
        
        # Stage 2: 128x128 -> 64x64 (4 branches)
        self.enc2_1 = EncoderBlock(base_filters, base_filters)
        self.enc2_2 = EncoderBlock(base_filters, base_filters)
        self.enc2_3 = EncoderBlock(base_filters, base_filters)
        self.enc2_4 = EncoderBlock(base_filters, base_filters)
        
        # Stage 3: 64x64 -> 32x32 (8 branches)
        self.enc3_blocks = nn.ModuleList([
            EncoderBlock(base_filters, base_filters) for _ in range(8)
        ])
        
        # Stage 4: 32x32 -> 16x16 (16 branches)
        self.enc4_blocks = nn.ModuleList([
            EncoderBlock(base_filters, base_filters) for _ in range(16)
        ])
        
        # Middle blocks (16개)
        self.mid_blocks = nn.ModuleList([
            MidBlock(base_filters) for _ in range(16)
        ])
        
        # Decoder - 4 stages
        # Stage 4: 16x16 -> 32x32 (8 decoders)
        self.dec4_blocks = nn.ModuleList([
            DecoderBlock(base_filters, base_filters) for _ in range(8)
        ])
        
        # Stage 3: 32x32 -> 64x64 (4 decoders)
        self.dec3_blocks = nn.ModuleList([
            DecoderBlock(base_filters, base_filters) for _ in range(4)
        ])
        
        # Stage 2: 64x64 -> 128x128 (2 decoders)
        self.dec2_1 = DecoderBlock(base_filters, base_filters)
        self.dec2_2 = DecoderBlock(base_filters, base_filters)
        
        # Stage 1: 128x128 -> 256x256 (1 decoder)
        self.dec1 = DecoderBlock(base_filters, base_filters)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_filters, 3, kernel_size=3, padding=1)
    
    def forward(self, input_tensor):
        # Initial
        B, C, H, W = input_tensor.shape
        inp = self.check_image_size(input_tensor)
                
        x0 = self.initial_conv(inp)
        
        # Encoder Stage 1: 256 -> 128
        e1_1 = self.enc1_1(x0)
        e1_2 = self.enc1_2(x0)
        
        # Encoder Stage 2: 128 -> 64
        e2_1 = self.enc2_1(e1_1)
        e2_2 = self.enc2_2(e1_1)
        e2_3 = self.enc2_3(e1_2)
        e2_4 = self.enc2_4(e1_2)
        
        # Encoder Stage 3: 64 -> 32
        e2_list = [e2_1, e2_2, e2_3, e2_4]
        e3_list = []
        for i, e2 in enumerate(e2_list):
            e3_1 = self.enc3_blocks[i*2](e2)
            e3_2 = self.enc3_blocks[i*2+1](e2)
            e3_list.extend([e3_1, e3_2])
        
        # Encoder Stage 4: 32 -> 16
        e4_list = []
        for i, e3 in enumerate(e3_list):
            e4_1 = self.enc4_blocks[i*2](e3)
            e4_2 = self.enc4_blocks[i*2+1](e3)
            e4_list.extend([e4_1, e4_2])
        
        # Middle blocks (16개 각각 적용)
        for i in range(16):
            e4_list[i] = self.mid_blocks[i](e4_list[i])
        
        # Decoder Stage 4: 16 -> 32 (16개를 8개로)
        d4_list = []
        for i in range(8):
            # 2개씩 묶어서 더한 후 디코딩
            merged = e4_list[i*2] + e4_list[i*2+1]
            d4 = self.dec4_blocks[i](merged, e3_list[i])
            d4_list.append(d4)
        
        # Decoder Stage 3: 32 -> 64 (8개를 4개로)
        d3_list = []
        for i in range(4):
            merged = d4_list[i*2] + d4_list[i*2+1]
            d3 = self.dec3_blocks[i](merged, e2_list[i])
            d3_list.append(d3)
        
        # Decoder Stage 2: 64 -> 128 (4개를 2개로)
        d2_1 = self.dec2_1(d3_list[0] + d3_list[1], e1_1)
        d2_2 = self.dec2_2(d3_list[2] + d3_list[3], e1_2)
        
        # Decoder Stage 1: 128 -> 256 (2개를 1개로)
        d1 = self.dec1(d2_1 + d2_2, x0)
        
        # Final output
        output = self.final_conv(d1)
        
        return output[:,:,:H,:W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# 테스트 코드
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing SimpleUNet...")
    model_1 = SimpleUNet(input_channels=3, num_filters=32).to(device)
    x = torch.randn(1, 3, 1920, 1080).to(device)
    
    model = model_1.to(device)
    input_tensor = torch.randn(1, 3, 1920, 1080).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # CUDA Event를 사용한 Inference Time 측정
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    num_iterations = 100
    timings = []

    with torch.no_grad():
        for _ in range(num_iterations):
            starter.record()
            output = model(input_tensor)
            ender.record()
            
            # GPU 동기화
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # milliseconds
            timings.append(curr_time)

    mean_time = sum(timings) / len(timings)
    std_time = (sum([(t - mean_time)**2 for t in timings]) / len(timings)) ** 0.5
    
    import torchsummary
    
    torchsummary.summary(model,(3,720,480))
    
    print(f"Average inference time (GPU): {mean_time:.2f} ms")
    print(f"Std: {std_time:.2f} ms")
    print(f"FPS: {1000/mean_time:.2f}")
    
    #파라미터 계산
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3,720,480), verbose=False, print_per_layer_stat=False)
    params = float(params[:-3])
    macs = float(macs[:-4])
    print(f"Custom MACS: {macs}, PARAMS:{params}")  