import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.full_model import TextToImageModel
from utils.data_utils import tokenize_text
from utils.evaluation import generate_interpolation, save_interpolation_grid
from config import Config

def set_seed(seed):
    """
    设置随机种子
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model(checkpoint_path=None):
    """
    加载模型
    
    参数:
        checkpoint_path (str, optional): 检查点路径
        
    返回:
        nn.Module: 加载的模型
    """
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
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
    ).to(device)
    
    # 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("使用未训练的模型")
    
    return model, config, device

def generate_single_image(text, model, config, device, output_path=None, show=True):
    """
    生成单个图像
    
    参数:
        text (str): 输入文本
        model (nn.Module): 模型
        config (Config): 配置
        device (torch.device): 设备
        output_path (str, optional): 输出路径
        show (bool): 是否显示图像
        
    返回:
        PIL.Image.Image: 生成的图像
    """
    model.eval()
    
    # 文本处理
    text_tensor = tokenize_text(text, config.max_text_length, config.vocab_size).to(device)
    
    # 生成图像
    with torch.no_grad():
        image = model(text_tensor)
    
    # 转换为PIL图像
    img = image[0].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
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

def generate_variations(text, model, config, device, num_variations=5, output_path=None, show=True):
    """
    生成多个变体图像
    
    参数:
        text (str): 输入文本
        model (nn.Module): 模型
        config (Config): 配置
        device (torch.device): 设备
        num_variations (int): 变体数量
        output_path (str, optional): 输出路径
        show (bool): 是否显示图像
        
    返回:
        PIL.Image.Image: 生成的图像网格
    """
    from torchvision.utils import make_grid
    import numpy as np
    
    model.eval()
    
    # 文本处理
    text_tensor = tokenize_text(text, config.max_text_length, config.vocab_size).to(device)
    text_tensor = text_tensor.repeat(num_variations, 1)  # 复制文本
    
    # 生成图像
    with torch.no_grad():
        images = model.generate_image(text_tensor, num_variations=1, noise_scale=1.0)
    
    # 创建图像网格
    grid = make_grid(images, nrow=min(num_variations, 5), normalize=True, padding=2)
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    grid_np = np.clip(grid_np, 0, 1)
    
    # 保存图像
    if output_path:
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(f'生成的图像变体: "{text}"')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"图像已保存到: {output_path}")
    
    # 显示图像
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(f'生成的图像变体: "{text}"')
        plt.show()
    
    return Image.fromarray((grid_np * 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description='生成示例图像')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--output_dir', type=str, default='examples/outputs', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_show', action='store_true', help='不显示图像')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, config, device = load_model(args.checkpoint)
    
    # 示例文本
    example_texts = [
        "一只猫坐在窗台上",
        "夕阳下的海滩",
        "繁星点点的夜空",
        "雪山下的森林",
        "城市的夜景"
    ]
    
    # 生成单个图像
    print("\n生成单个图像...")
    for i, text in enumerate(example_texts):
        output_path = os.path.join(args.output_dir, f"single_{i+1}.png")
        generate_single_image(text, model, config, device, output_path, not args.no_show)
    
    # 生成变体图像
    print("\n生成变体图像...")
    for i, text in enumerate(example_texts[:2]):  # 只使用前两个文本示例
        output_path = os.path.join(args.output_dir, f"variations_{i+1}.png")
        generate_variations(text, model, config, device, 5, output_path, not args.no_show)
    
    # 生成插值图像
    print("\n生成插值图像...")
    text1 = "一只猫坐在窗台上"
    text2 = "夕阳下的海滩"
    output_path = os.path.join(args.output_dir, "interpolation.png")
    
    interpolated_images = generate_interpolation(model, text1, text2, steps=8, config=config)
    save_interpolation_grid(interpolated_images, output_path, text1, text2)
    print(f"插值图像已保存到: {output_path}")

if __name__ == "__main__":
    main()