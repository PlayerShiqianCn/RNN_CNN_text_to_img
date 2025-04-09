# RNN_CNN_text_to_img 文本生成图像模型
# 目前AI生成
## 项目概述

这个项目实现了一个基于LSTM、CNN和FNN的文本到图像生成模型，不使用Transformer架构。该模型通过以下步骤将文本转换为图像：

1. 使用LSTM处理One-hot编码的文本输入，提取文本特征
2. 将文本特征转换为图像特征，并插入到随机噪声中
3. 使用CNN进行去噪处理
4. 添加噪声（使用FNN处理输入的one-hot文本与噪声图相加后压缩的特征）
5. 重复CNN去噪和添加噪声的步骤
6. 输出最终生成的图像

## 模型架构

### 文本编码器 (LSTM)
- 输入：One-hot编码的文本
- 隐藏层激活函数：ELU
- 输出：文本特征表示

### 噪声处理模块 (FNN)
- 输入：文本特征与噪声图相加后压缩到k/12 x k/12大小
- 输出：处理后的噪声

### 图像生成器 (CNN)
- 输入：带噪声的图像
- 输出层激活函数：Sigmoid
- 输出：去噪后的图像，大小为k x k

## 与Stable Diffusion的比较

### 优点
1. 计算资源需求较低，不需要大型Transformer模型
2. 实现相对简单，更易于理解和调试
3. 训练数据需求可能更少
4. 推理速度可能更快
5. 可以在资源受限的环境中运行

### 缺点
1. 生成图像的质量和细节可能不如Stable Diffusion
2. 对长文本的理解能力有限（LSTM相比Transformer在处理长序列时效果较差）
3. 缺乏Stable Diffusion中的注意力机制，可能导致文本-图像对齐性较弱
4. 多样性可能需要额外的机制来保证
5. 缺乏预训练的大规模模型的知识

## 实现多样性的方法

为了增强生成图像的多样性，本项目将采用以下策略：

1. 在噪声添加过程中引入随机性
2. 在训练过程中使用数据增强
3. 实现条件变分自编码器(CVAE)的思想
4. 在推理时使用不同的随机种子
5. 添加风格控制参数

## 项目结构

```
├── data/                  # 数据集
├── models/                # 模型定义
│   ├── lstm_encoder.py    # LSTM文本编码器
│   ├── fnn_noise.py       # FNN噪声处理模块
│   ├── cnn_denoiser.py    # CNN去噪网络
│   └── full_model.py      # 完整模型
├── utils/                 # 工具函数
├── train.py               # 训练脚本
├── generate.py            # 图像生成脚本
└── config.py              # 配置文件
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python train.py
```

### 生成图像
```bash
python generate.py --text "描述文本" --output output.png
```
### 对于爬取图像
修改crawler.py的main
```python
    crawler.download_images("你想要的", num_images=数量)
```
