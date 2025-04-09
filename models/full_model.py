import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.lstm_encoder import LSTMEncoder
from models.fnn_noise import FNNNoiseProcessor
from models.cnn_denoiser import CNNDenoiser

class TextToImageModel(nn.Module):
    """
    完整的文本到图像生成模型
    结合LSTM文本编码器、FNN噪声处理模块和CNN去噪网络
    """
    def __init__(self, vocab_size, image_size=64, channels=3, text_embedding_dim=300, 
                 text_hidden_dim=512, text_output_dim=1024, noise_hidden_dims=[512, 1024, 2048],
                 cnn_base_channels=64, num_denoising_steps=3):
        """
        初始化文本到图像生成模型
        
        参数:
            vocab_size (int): 词汇表大小
            image_size (int): 生成图像的大小 (k x k)
            channels (int): 图像通道数
            text_embedding_dim (int): 文本嵌入维度
            text_hidden_dim (int): LSTM隐藏层维度
            text_output_dim (int): 文本特征输出维度
            noise_hidden_dims (list): FNN噪声处理模块的隐藏层维度
            cnn_base_channels (int): CNN去噪网络的基础通道数
            num_denoising_steps (int): 去噪步骤的重复次数
        """
        super(TextToImageModel, self).__init__()
        
        self.image_size = image_size
        self.channels = channels
        self.num_denoising_steps = num_denoising_steps
        
        # 压缩因子
        self.compression_factor = 12
        compressed_size = image_size // self.compression_factor
        compressed_dim = compressed_size * compressed_size * channels
        
        # 初始化LSTM文本编码器
        self.text_encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            output_dim=text_output_dim
        )
        
        # 初始化FNN噪声处理模块
        self.noise_processor = FNNNoiseProcessor(
            input_dim=compressed_dim,
            hidden_dims=noise_hidden_dims,
            output_dim=compressed_dim
        )
        
        # 初始化CNN去噪网络
        self.denoiser = CNNDenoiser(
            input_channels=channels,
            output_channels=channels,
            base_channels=cnn_base_channels
        )
        
    def forward(self, text, noise=None, return_intermediates=False):
        """
        前向传播
        
        参数:
            text (torch.Tensor): 输入文本的one-hot编码 [batch_size, seq_len]
            noise (torch.Tensor, optional): 初始噪声图像，如果为None则随机生成
            return_intermediates (bool): 是否返回中间结果
            
        返回:
            torch.Tensor: 生成的图像 [batch_size, channels, height, width]
            list (optional): 中间生成的图像列表
        """
        batch_size = text.shape[0]
        device = text.device
        
        # 获取文本特征
        text_features = self.text_encoder(text)  # [batch_size, text_output_dim]
        
        # 如果没有提供噪声，则生成随机噪声
        if noise is None:
            noise = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        
        # 初始图像为噪声
        current_image = noise
        
        # 保存中间结果
        intermediate_images = [current_image] if return_intermediates else None
        
        # 迭代去噪和添加噪声的步骤
        for i in range(self.num_denoising_steps):
            # 去噪
            denoised_image = self.denoiser(current_image)
            
            if return_intermediates:
                intermediate_images.append(denoised_image)
            
            # 如果是最后一步，直接返回去噪后的图像
            if i == self.num_denoising_steps - 1:
                break
            
            # 添加噪声
            noise = self.noise_processor.process_compressed_features(
                text_features, denoised_image, self.compression_factor
            )
            
            # 更新当前图像
            current_image = denoised_image + 0.1 * noise  # 添加一定比例的噪声
            
            if return_intermediates:
                intermediate_images.append(current_image)
        
        if return_intermediates:
            return denoised_image, intermediate_images
        else:
            return denoised_image
    
    def generate_image(self, text, num_variations=1, noise_scale=1.0):
        """
        生成图像
        
        参数:
            text (torch.Tensor): 输入文本的one-hot编码 [batch_size, seq_len]
            num_variations (int): 生成变体的数量
            noise_scale (float): 噪声比例，控制多样性
            
        返回:
            torch.Tensor: 生成的图像 [batch_size * num_variations, channels, height, width]
        """
        batch_size = text.shape[0]
        device = text.device
        
        # 复制文本特征以生成多个变体
        if num_variations > 1:
            text = text.repeat(num_variations, 1)
        
        # 生成不同的随机噪声
        noise = torch.randn(
            batch_size * num_variations, 
            self.channels, 
            self.image_size, 
            self.image_size, 
            device=device
        ) * noise_scale
        
        # 生成图像
        with torch.no_grad():
            images = self.forward(text, noise)
        
        return images