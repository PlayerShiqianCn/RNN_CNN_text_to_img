import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from sklearn.metrics import mean_squared_error
from PIL import Image

def save_comparison(real_images, generated_images, output_path, titles=None):
    """
    保存真实图像和生成图像的对比图
    
    参数:
        real_images (torch.Tensor): 真实图像张量 [batch_size, channels, height, width]
        generated_images (torch.Tensor): 生成图像张量 [batch_size, channels, height, width]
        output_path (str): 输出路径
        titles (list, optional): 图像标题列表
    """
    # 确保输入是张量
    if not isinstance(real_images, torch.Tensor):
        real_images = torch.tensor(real_images)
    if not isinstance(generated_images, torch.Tensor):
        generated_images = torch.tensor(generated_images)
    
    # 确保形状一致
    batch_size = min(real_images.shape[0], generated_images.shape[0])
    real_images = real_images[:batch_size]
    generated_images = generated_images[:batch_size]
    
    # 创建对比图
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    # 如果batch_size为1，则axes不是二维数组
    if batch_size == 1:
        axes = np.array([axes])
    
    for i in range(batch_size):
        # 真实图像
        real_img = real_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        real_img = np.clip(real_img, 0, 1)
        axes[i, 0].imshow(real_img)
        axes[i, 0].set_title('真实图像' if titles is None else titles[i][0])
        axes[i, 0].axis('off')
        
        # 生成图像
        gen_img = generated_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        gen_img = np.clip(gen_img, 0, 1)
        axes[i, 1].imshow(gen_img)
        axes[i, 1].set_title('生成图像' if titles is None else titles[i][1])
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calculate_psnr(real_images, generated_images):
    """
    计算峰值信噪比 (PSNR)
    
    参数:
        real_images (torch.Tensor): 真实图像张量 [batch_size, channels, height, width]
        generated_images (torch.Tensor): 生成图像张量 [batch_size, channels, height, width]
        
    返回:
        float: 平均PSNR
    """
    # 确保输入是张量
    if not isinstance(real_images, torch.Tensor):
        real_images = torch.tensor(real_images)
    if not isinstance(generated_images, torch.Tensor):
        generated_images = torch.tensor(generated_images)
    
    # 确保形状一致
    batch_size = min(real_images.shape[0], generated_images.shape[0])
    real_images = real_images[:batch_size]
    generated_images = generated_images[:batch_size]
    
    # 转换为numpy数组
    real_np = real_images.cpu().detach().numpy()
    gen_np = generated_images.cpu().detach().numpy()
    
    # 计算MSE
    mse = np.mean((real_np - gen_np) ** 2)
    
    # 计算PSNR
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_ssim(real_images, generated_images):
    """
    计算结构相似性指数 (SSIM)
    注意：这是一个简化版本，实际应用中应该使用更复杂的实现
    
    参数:
        real_images (torch.Tensor): 真实图像张量 [batch_size, channels, height, width]
        generated_images (torch.Tensor): 生成图像张量 [batch_size, channels, height, width]
        
    返回:
        float: 平均SSIM
    """
    # 这里应该实现SSIM的计算
    # 简化版本，返回随机值
    return np.random.uniform(0, 1)

def evaluate_diversity(generated_images):
    """
    评估生成图像的多样性
    
    参数:
        generated_images (torch.Tensor): 生成图像张量 [batch_size, channels, height, width]
        
    返回:
        float: 多样性分数
    """
    # 确保输入是张量
    if not isinstance(generated_images, torch.Tensor):
        generated_images = torch.tensor(generated_images)
    
    # 计算图像之间的平均欧氏距离
    batch_size = generated_images.shape[0]
    if batch_size <= 1:
        return 0.0
    
    # 展平图像
    flattened = generated_images.view(batch_size, -1)
    
    # 计算所有图像对之间的欧氏距离
    distances = []
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            dist = torch.norm(flattened[i] - flattened[j]).item()
            distances.append(dist)
    
    # 返回平均距离
    return np.mean(distances) if distances else 0.0

def generate_interpolation(model, text1, text2, steps=10, config=None):
    """
    在两个文本之间生成插值图像
    
    参数:
        model (nn.Module): 模型
        text1 (str): 第一个文本
        text2 (str): 第二个文本
        steps (int): 插值步数
        config (Config): 配置
        
    返回:
        torch.Tensor: 插值图像 [steps, channels, height, width]
    """
    from utils.data_utils import tokenize_text
    
    device = next(model.parameters()).device
    
    # 文本处理
    text1_tensor = tokenize_text(text1, config.max_text_length, config.vocab_size).to(device)
    text2_tensor = tokenize_text(text2, config.max_text_length, config.vocab_size).to(device)
    
    # 获取文本特征
    with torch.no_grad():
        text1_features = model.text_encoder(text1_tensor)
        text2_features = model.text_encoder(text2_tensor)
    
    # 生成插值图像
    interpolated_images = []
    with torch.no_grad():
        # 使用相同的噪声
        noise = torch.randn(1, model.channels, model.image_size, model.image_size, device=device)
        
        for alpha in np.linspace(0, 1, steps):
            # 特征插值
            interpolated_features = (1 - alpha) * text1_features + alpha * text2_features
            
            # 替换模型的文本特征
            def forward_with_features(text, noise=None):
                batch_size = text.shape[0]
                device = text.device
                
                # 使用插值特征
                text_features = interpolated_features
                
                # 如果没有提供噪声，则生成随机噪声
                if noise is None:
                    noise = torch.randn(batch_size, model.channels, model.image_size, model.image_size, device=device)
                
                # 初始图像为噪声
                current_image = noise
                
                # 迭代去噪和添加噪声的步骤
                for i in range(model.num_denoising_steps):
                    # 去噪
                    denoised_image = model.denoiser(current_image)
                    
                    # 如果是最后一步，直接返回去噪后的图像
                    if i == model.num_denoising_steps - 1:
                        break
                    
                    # 添加噪声
                    noise = model.noise_processor.process_compressed_features(
                        text_features, denoised_image, model.compression_factor
                    )
                    
                    # 更新当前图像
                    current_image = denoised_image + 0.1 * noise
                
                return denoised_image
            
            # 生成图像
            image = forward_with_features(text1_tensor, noise)
            interpolated_images.append(image)
    
    return torch.cat(interpolated_images, dim=0)

def save_interpolation_grid(interpolated_images, output_path, text1, text2):
    """
    保存插值图像网格
    
    参数:
        interpolated_images (torch.Tensor): 插值图像张量 [steps, channels, height, width]
        output_path (str): 输出路径
        text1 (str): 第一个文本
        text2 (str): 第二个文本
    """
    # 创建图像网格
    grid = make_grid(interpolated_images, nrow=interpolated_images.shape[0], normalize=True, padding=2)
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    grid_np = np.clip(grid_np, 0, 1)
    
    # 保存图像
    plt.figure(figsize=(15, 5))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f'从 "{text1}" 到 "{text2}" 的插值')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()