# 模型配置文件

class Config:
    # 数据集配置
    data_dir = "data"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    max_text_length = 100
    
    # 模型配置
    vocab_size = 10000  # 词汇表大小
    image_size = 64     # 生成图像的大小 (k x k)
    channels = 3        # 图像通道数
    
    # 文本编码器配置
    text_embedding_dim = 300
    text_hidden_dim = 512
    text_num_layers = 2
    text_output_dim = 1024
    text_dropout = 0.2
    
    # 噪声处理模块配置
    noise_hidden_dims = [512, 1024, 2048]
    compression_factor = 12
    
    # CNN去噪网络配置
    cnn_base_channels = 64
    cnn_num_layers = 4
    
    # 生成配置
    num_denoising_steps = 3
    noise_scale = 1.0
    num_variations = 5
    
    # 训练配置
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-5
    clip_grad_norm = 5.0
    
    # 优化器配置
    optimizer = "adam"
    beta1 = 0.9
    beta2 = 0.999
    
    # 学习率调度器配置
    lr_scheduler = "cosine"
    lr_scheduler_step_size = 10
    lr_scheduler_gamma = 0.5
    
    # 保存和加载配置
    checkpoint_dir = "checkpoints"
    save_interval = 5
    load_checkpoint = None
    
    # 日志配置
    log_dir = "logs"
    log_interval = 100
    
    # 设备配置
    device = "cpu# "cuda" 或 "cpu"
    
    # 随机种子
    seed = 42