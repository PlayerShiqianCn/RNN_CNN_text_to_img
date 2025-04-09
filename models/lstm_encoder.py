import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    """
    LSTM文本编码器，将one-hot编码的文本转换为特征表示
    隐藏层使用ELU激活函数
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, num_layers=2, output_dim=1024, dropout=0.2):
        """
        初始化LSTM编码器
        
        参数:
            vocab_size (int): 词汇表大小
            embedding_dim (int): 词嵌入维度
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            output_dim (int): 输出特征维度
            dropout (float): Dropout比率
        """
        super(LSTMEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 使用ELU作为隐藏层激活函数
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x, hidden=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入文本的one-hot编码 [batch_size, seq_len]
            hidden (tuple, optional): 初始隐藏状态
            
        返回:
            torch.Tensor: 文本特征表示 [batch_size, output_dim]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM处理
        if hidden is None:
            output, (hidden, cell) = self.lstm(embedded)
        else:
            output, (hidden, cell) = self.lstm(embedded, hidden)
        
        # 使用最后一个时间步的输出
        hidden_state = hidden[-1]  # [batch_size, hidden_dim]
        
        # 通过全连接层和ELU激活函数
        x = self.fc1(hidden_state)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        
        参数:
            batch_size (int): 批次大小
            device (torch.device): 设备
            
        返回:
            tuple: (hidden, cell) LSTM的隐藏状态和细胞状态
        """
        hidden = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        cell = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (hidden, cell)

    def get_text_features(self, x):
        """
        获取文本特征，用于生成图像
        
        参数:
            x (torch.Tensor): 输入文本的one-hot编码 [batch_size, seq_len]
            
        返回:
            torch.Tensor: 文本特征表示 [batch_size, output_dim]
        """
        return self.forward(x)