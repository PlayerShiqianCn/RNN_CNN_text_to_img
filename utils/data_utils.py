import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd

class TextImageDataset(Dataset):
    """
    文本-图像数据集
    """
    def __init__(self, data_dir, transform=None, max_text_length=100, vocab_size=10000):
        """
        初始化文本-图像数据集
        
        参数:
            data_dir (str): 数据目录
            transform (torchvision.transforms): 图像变换
            max_text_length (int): 最大文本长度
            vocab_size (int): 词汇表大小
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        
        # 加载数据
        self.data = self._load_data()
        
    def _load_data(self):
        """
        加载数据
        
        返回:
            list: 数据列表，每个元素为(text_path, image_path)元组
        """
        # 这里应该根据实际数据集格式进行修改
        # 示例实现：假设data_dir下有一个metadata.csv文件，包含text_path和image_path列
        metadata_path = os.path.join(self.data_dir, 'metadata.csv')
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            data = list(zip(df['text_path'], df['image_path']))
        else:
            # 如果没有metadata.csv，则假设数据目录下有text和images两个子目录
            text_dir = os.path.join(self.data_dir, 'text')
            image_dir = os.path.join(self.data_dir, 'images')
            
            if not (os.path.exists(text_dir) and os.path.exists(image_dir)):
                raise FileNotFoundError(f"找不到数据目录: {text_dir} 或 {image_dir}")
            
            text_files = sorted(os.listdir(text_dir))
            image_files = sorted(os.listdir(image_dir))
            
            # 确保文本和图像文件数量相同
            assert len(text_files) == len(image_files), "文本和图像文件数量不匹配"
            
            data = [
                (os.path.join(text_dir, text_file), os.path.join(image_dir, image_file))
                for text_file, image_file in zip(text_files, image_files)
            ]
        
        return data
    
    def _tokenize_text(self, text):
        """
        将文本转换为one-hot编码
        
        参数:
            text (str): 输入文本
            
        返回:
            torch.Tensor: one-hot编码的文本 [max_text_length]
        """
        # 简单的字符级tokenization，实际应用中应该使用更复杂的分词器
        tokens = [ord(c) % self.vocab_size for c in text[:self.max_text_length]]
        
        # 填充到最大长度
        if len(tokens) < self.max_text_length:
            tokens += [0] * (self.max_text_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_path, image_path = self.data[idx]
        
        # 加载文本
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 文本转换为one-hot编码
        text_tensor = self._tokenize_text(text)
        
        return text_tensor, image

def tokenize_text(text, max_length, vocab_size):
    """
    将文本转换为one-hot编码
    
    参数:
        text (str): 输入文本
        max_length (int): 最大文本长度
        vocab_size (int): 词汇表大小
        
    返回:
        torch.Tensor: one-hot编码的文本 [1, max_length]
    """
    # 简单的字符级tokenization，实际应用中应该使用更复杂的分词器
    tokens = [ord(c) % vocab_size for c in text[:max_length]]
    
    # 填充到最大长度
    if len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # 添加批次维度

def save_image_grid(images, output_path, title=None, nrow=5):
    """
    保存图像网格
    
    参数:
        images (torch.Tensor): 图像张量 [batch_size, channels, height, width]
        output_path (str): 输出路径
        title (str, optional): 图像标题
        nrow (int): 每行的图像数量
    """
    from torchvision.utils import make_grid
    
    # 创建图像网格
    grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    grid_np = np.clip(grid_np, 0, 1)
    
    # 保存图像
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return grid_np

def calculate_inception_score(images):
    """
    计算Inception Score
    注意：这是一个简化版本，实际应用中应该使用预训练的Inception模型
    
    参数:
        images (torch.Tensor): 图像张量 [batch_size, channels, height, width]
        
    返回:
        float: Inception Score
    """
    # 这里应该实现Inception Score的计算
    # 简化版本，返回随机值
    return np.random.uniform(1, 10)

def calculate_fid(real_images, generated_images):
    """
    计算Fréchet Inception Distance (FID)
    注意：这是一个简化版本，实际应用中应该使用预训练的Inception模型
    
    参数:
        real_images (torch.Tensor): 真实图像张量 [batch_size, channels, height, width]
        generated_images (torch.Tensor): 生成图像张量 [batch_size, channels, height, width]
        
    返回:
        float: FID分数
    """
    # 这里应该实现FID的计算
    # 简化版本，返回随机值
    return np.random.uniform(0, 100)