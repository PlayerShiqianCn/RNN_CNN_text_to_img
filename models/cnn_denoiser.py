import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDenoiser(nn.Module):
    """
    CNN去噪网络
    将带噪声的图像处理成清晰的图像
    输出层使用Sigmoid激活函数
    """
    def __init__(self, input_channels=3, output_channels=3, base_channels=64, num_layers=4):
        """
        初始化CNN去噪网络
        
        参数:
            input_channels (int): 输入图像的通道数
            output_channels (int): 输出图像的通道数
            base_channels (int): 基础通道数
            num_layers (int): U-Net结构的层数
        """
        super(CNNDenoiser, self).__init__()
        
        # 编码器部分
        self.encoders = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        # 第一层编码器
        self.encoders.append(self._make_conv_block(input_channels, base_channels))
        
        # 后续编码器层
        for i in range(1, num_layers):
            in_channels = base_channels * (2 ** (i-1))
            out_channels = base_channels * (2 ** i)
            self.encoders.append(self._make_conv_block(in_channels, out_channels))
            self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 中间层
        middle_channels = base_channels * (2 ** num_layers)
        self.middle_conv = self._make_conv_block(base_channels * (2 ** (num_layers-1)), middle_channels)
        
        # 解码器部分
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # 上采样和解码器层
        for i in range(num_layers-1, 0, -1):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i-1))
            self.upsamples.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(self._make_conv_block(in_channels, out_channels))
        
        # 输出层，使用Sigmoid激活函数
        self.output_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_conv_block(self, in_channels, out_channels):
        """
        创建卷积块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            
        返回:
            nn.Sequential: 卷积块
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入的带噪声图像 [batch_size, input_channels, height, width]
            
        返回:
            torch.Tensor: 去噪后的图像 [batch_size, output_channels, height, width]
        """
        # 保存编码器的输出，用于跳跃连接
        encoder_outputs = []
        
        # 编码器前向传播
        out = self.encoders[0](x)
        encoder_outputs.append(out)
        
        for i in range(len(self.encoder_pools)):
            out = self.encoder_pools[i](out)
            out = self.encoders[i+1](out)
            encoder_outputs.append(out)
        
        # 中间层
        out = self.middle_conv(out)
        
        # 解码器前向传播
        for i in range(len(self.decoders)):
            out = self.upsamples[i](out)
            # 跳跃连接，连接对应的编码器输出
            out = torch.cat([out, encoder_outputs[-(i+2)]], dim=1)
            out = self.decoders[i](out)
        
        # 输出层
        out = self.output_conv(out)
        out = self.sigmoid(out)  # 使用Sigmoid确保输出在[0,1]范围内
        
        return out
    
    def denoise_image(self, noisy_image):
        """
        对带噪声的图像进行去噪
        
        参数:
            noisy_image (torch.Tensor): 带噪声的图像 [batch_size, channels, height, width]
            
        返回:
            torch.Tensor: 去噪后的图像 [batch_size, channels, height, width]
        """
        return self.forward(noisy_image)