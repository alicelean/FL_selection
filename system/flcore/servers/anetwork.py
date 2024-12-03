import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size=600, num_heads=4):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, x):
        # 假设输入 x 形状为 [sequence_length, batch_size, embed_size]
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class FeedAtteniion(nn.Module):
    def __init__(self, num_clients, num_features, output_dim):
        super(FeedAtteniion, self).__init__()
        # 将输入矩阵展平成一维向量
        self.flatten_size = num_clients * num_features
        # 两个隐藏层，每个包含256个神经元
        # 定义 Attention 层
        self.attention = SelfAttention()

        self.fc1 = nn.Linear(self.flatten_size, 256)

        self.fc2 = nn.Linear(256, 256)
        # 输出层，输出维度为给定的output_dim
        self.fc3 = nn.Linear(256, output_dim)

    def forward_1(self, obs):
        #如果输入是 numpy 数组，先转换为 PyTorch 张量
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float)
            # 检查输入有效性
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            raise ValueError("obs contains NaN or infinity values")
        #print("obs is :obs",obs)

        # 展平输入矩阵

        # 归一化输入数据（标准化）
        # 如果数据是多维的，可以使用 torch.mean 和 torch.std 对每一维进行标准化
        # 这里假设输入是2D的：batch_size x flatten_size
        obs_mean = torch.mean(obs, dim=0, keepdim=True)
        obs_std = torch.std(obs, dim=0, keepdim=True)

        # 标准化（去均值，除以标准差）
        obs = (obs - obs_mean) / (obs_std + 1e-8)  # 添加一个小的epsilon防止除以0

        # 展平输入矩阵
        obs = obs.view(-1, self.flatten_size)
        obs = obs.float()
        #print(f"Input shape after flattening: {obs.shape}")
        x = torch.relu(self.fc1(obs))  # 第一个隐藏层 + ReLU
        x = torch.relu(self.fc2(x))  # 第二个隐藏层 + ReLU
        output = self.fc3(x)
        #print("x",x)# 输出层（线性）
        return output

    def forward(self, obs):
        # 如果输入是 numpy 数组，先转换为 PyTorch 张量
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float)

        # 检查输入有效性
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            raise ValueError("obs contains NaN or infinity values")

        # 归一化输入数据（标准化）
       # print("obs is" ,obs)
        obs= obs.to(torch.float32)
        obs_mean = torch.mean(obs, dim=0, keepdim=True)
        obs_std = torch.std(obs, dim=0, keepdim=True)
        obs = (obs - obs_mean) / (obs_std + 1e-8)  # 添加一个小的epsilon防止除以0

        # 展平输入矩阵
        obs = obs.view(-1, self.flatten_size)

        # 使用自注意力机制
        # 这里的输入需要是 [sequence_length, batch_size, embed_size] 格式
        obs = obs.unsqueeze(0)  # 将批次维度添加到第一维，使得输入形状为 [1, batch_size, flatten_size]
        attention_output = self.attention(obs)  # 使用自注意力机制

        # 展开输出为 batch_size, flatten_size 形状
        attention_output = attention_output.squeeze(0)  # 将维度恢复为 [batch_size, flatten_size]

        # 全连接层处理
        x = torch.relu(self.fc1(attention_output))  # 第一个隐藏层 + ReLU
        x = torch.relu(self.fc2(x))  # 第二个隐藏层 + ReLU
        output = self.fc3(x)  # 输出层（线性）

        return output

