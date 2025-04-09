import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from models.full_model import TextToImageModel
from config import Config

# 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 文本-图像数据集
class TextImageDataset(Dataset):
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

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    训练模型
    
    参数:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        optimizer (torch.optim): 优化器
        scheduler (torch.optim.lr_scheduler): 学习率调度器
        config (Config): 配置
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 创建日志目录
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 训练日志
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (text, target_images) in enumerate(progress_bar):
            text, target_images = text.to(device), target_images.to(device)
            
            # 前向传播
            generated_images = model(text)
            
            # 计算损失
            loss = criterion(generated_images, target_images)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            
            optimizer.step()
            
            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})
            
            # 记录日志
            if (batch_idx + 1) % config.log_interval == 0:
                step = epoch * len(train_loader) + batch_idx
                train_losses.append((step, loss.item()))
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for text, target_images in val_loader:
                text, target_images = text.to(device), target_images.to(device)
                
                # 前向传播
                generated_images = model(text)
                
                # 计算损失
                loss = criterion(generated_images, target_images)
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append((epoch, avg_val_loss))
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 保存训练日志
    train_log = pd.DataFrame(train_losses, columns=['step', 'loss'])
    val_log = pd.DataFrame(val_losses, columns=['epoch', 'loss'])
    
    train_log.to_csv(os.path.join(config.log_dir, 'train_loss.csv'), index=False)
    val_log.to_csv(os.path.join(config.log_dir, 'val_loss.csv'), index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses])
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses])
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, 'loss_curves.png'))

# 主函数
def main():
    parser = argparse.ArgumentParser(description='训练文本到图像生成模型')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
    # 加载数据集
    dataset = TextImageDataset(
        data_dir=config.data_dir,
        transform=transform,
        max_text_length=config.max_text_length,
        vocab_size=config.vocab_size
    )
    
    # 划分数据集
    dataset_size = len(dataset)
    train_size = int(dataset_size * config.train_ratio)
    val_size = int(dataset_size * config.val_ratio)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = TextToImageModel(
        vocab_size=config.vocab_size,
        image_size=config.image_size,
        channels=config.channels,
        text_embedding_dim=config.text_embedding_dim,
        text_hidden_dim=config.text_hidden_dim,
        text_output_dim=config.text_output_dim,
        noise_hidden_dims=config.noise_hidden_dims,
        cnn_base_channels=config.cnn_base_channels,
        num_denoising_steps=config.num_denoising_steps
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")
    
    # 学习率调度器
    if config.lr_scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_scheduler_step_size,
            gamma=config.lr_scheduler_gamma
        )
    elif config.lr_scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )
    else:
        scheduler = None
    
    # 加载检查点
    if config.load_checkpoint:
        checkpoint = torch.load(config.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"加载检查点: {config.load_checkpoint}, 从epoch {start_epoch}开始训练")
    
    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
    
    print("训练完成!")

if __name__ == "__main__":
    main()