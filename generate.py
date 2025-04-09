import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from models.full_model import TextToImageModel
from config import Config

def set_seed(seed):
    """
    设置随机种子，确保结果可复现
    
    参数:
        seed (int): 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def generate_images(model, text, config, num_variations=None, noise_scale=None, output_path=None, show=True):
    """
    生成图像
    
    参数:
        model (nn.Module): 模型
        text (str): 输入文本
        config (Config): 配置
        num_variations (int, optional): 生成变体的数量
        noise_scale (float, optional): 噪声比例，控制多样性
        output_path (str, optional): 输出图像路径
        show (bool): 是否显示图像
        
    返回:
        PIL.Image.Image: 生成的图像
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 设置生成参数
    if num_variations is None:
        num_variations = config.num_variations
    if noise_scale is None:
        noise_scale = config.noise_scale
    
    # 文本处理
    text_tensor = tokenize_text(text, config.max_text_length, config.vocab_size).to(device)
    
    # 生成图像
    with torch.no_grad():
        images = model.generate_image(text_tensor, num_variations, noise_scale)
    
    # 转换为PIL图像
    if num_variations > 1:
        # 创建图像网格
        grid = make_grid(images, nrow=min(num_variations, 5), normalize=True, padding=2)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        grid_np = np.clip(grid_np, 0, 1)
        
        # 保存图像
        if output_path:
            plt.figure(figsize=(12, 12))
            plt.imshow(grid_np)
            plt.axis('off')
            plt.title(f'生成的图像: "{text}"')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"图像已保存到: {output_path}")
            
            # 同时保存单独的图像
            output_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            for i in range(num_variations):
                img_path = os.path.join(output_dir, f"{base_name}_{i+1}.png")
                save_image(images[i], img_path)
        
        # 显示图像
        if show:
            plt.figure(figsize=(12, 12))
            plt.imshow(grid_np)
            plt.axis('off')
            plt.title(f'生成的图像: "{text}"')
            plt.show()
        
        return Image.fromarray((grid_np * 255).astype(np.uint8))
    else:
        # 单个图像
        img = images[0].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = np.clip(img, 0, 1)
        
        # 保存图像
        if output_path:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'生成的图像: "{text}"')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"图像已保存到: {output_path}")
        
        # 显示图像
        if show:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'生成的图像: "{text}"')
            plt.show()
        
        return Image.fromarray((img * 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description='使用文本生成图像')
    parser.add_argument('--text', type=str, required=True, help='输入文本')
    parser.add_argument('--output', type=str, default='output.png', help='输出图像路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--variations', type=int, default=None, help='生成变体的数量')
    parser.add_argument('--noise_scale', type=float, default=None, help='噪声比例，控制多样性')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--no_show', action='store_true', help='不显示图像')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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
    
    # 加载检查点
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 如果未指定检查点，则使用最佳模型
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            # 如果最佳模型不存在，则查找最新的检查点
            checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[0])
    
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"警告: 找不到检查点 {checkpoint_path}，使用未训练的模型")
    
    # 生成图像
    generate_images(
        model=model,
        text=args.text,
        config=config,
        num_variations=args.variations,
        noise_scale=args.noise_scale,
        output_path=args.output,
        show=not args.no_show
    )

if __name__ == "__main__":
    main()