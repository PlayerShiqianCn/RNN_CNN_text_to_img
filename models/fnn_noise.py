import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNNoiseProcessor(nn.Module):
    """
    FNN噪声处理模块
    处理输入的one-hot文本与噪声图相加后压缩的特征
    """
    def __init__(self, input_dim, hidden_dims=[512, 1024, 2048], output_dim=None, dropout=0.2):
        """
        初始化FNN噪声处理模块
        
        参数:
            input_dim (int): 输入维度 (压缩后的特征维度)
            hidden_dims (list): 隐藏层维度列表
            output_dim (int): 输出维度，如果为None则使用input_dim
            dropout (float): Dropout比率
        """
        super(FNNNoiseProcessor, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())  # 使用ELU激活函数
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, input_dim]
            
        返回:
            torch.Tensor: 处理后的噪声 [batch_size, output_dim]
        """
        return self.model(x)
    
    def process_compressed_features(self, text_features, noise_image, compression_factor=12):
        """
        处理文本特征与噪声图相加后压缩的特征
        
        参数:
            text_features (torch.Tensor): 文本特征 [batch_size, feature_dim]
            noise_image (torch.Tensor): 噪声图 [batch_size, channels, height, width]
            compression_factor (int): 压缩因子，将图像压缩到原来的1/compression_factor
            
        返回:
            torch.Tensor: 处理后的噪声，与输入noise_image形状相同
        """
        batch_size, channels, height, width = noise_image.shape
        
        # 将文本特征转换为与噪声图相同的形状
        text_features_reshaped = text_features.view(batch_size, -1, 1, 1)
        text_features_expanded = text_features_reshaped.expand(-1, -1, height, width)
        
        # 如果通道数不匹配，调整文本特征的通道数
        if text_features_expanded.shape[1] != channels:
            text_features_expanded = F.conv2d(
                text_features_expanded, 
                torch.ones(channels, text_features_expanded.shape[1], 1, 1, device=text_features.device) / text_features_expanded.shape[1]
            )
        
        # 文本特征与噪声图相加
        combined = text_features_expanded + noise_image
        
        # 压缩到k/compression_factor x k/compression_factor大小
        compressed_height = height // compression_factor
        compressed_width = width // compression_factor
        compressed = F.adaptive_avg_pool2d(combined, (compressed_height, compressed_width))
        
        # 展平为一维向量
        flattened = compressed.view(batch_size, -1)
        
        # 通过FNN处理
        processed = self.forward(flattened)
        
        # 重塑为原始噪声图的形状
        processed_reshaped = processed.view(batch_size, channels, compressed_height, compressed_width)
        
        # 上采样回原始大小
        processed_upsampled = F.interpolate(processed_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        
        return processed_upsampled