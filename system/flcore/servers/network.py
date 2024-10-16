import torch
from torch import nn
import numpy as np
class FeedForwardNN(nn.Module):
    def __init__(self, num_clients, num_features, output_dim):
        super(FeedForwardNN, self).__init__()
        # 将输入矩阵展平成一维向量
        self.flatten_size = num_clients * num_features
        # 两个隐藏层，每个包含256个神经元
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 256)
        # 输出层，输出维度为给定的output_dim
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, obs):
        #如果输入是 numpy 数组，先转换为 PyTorch 张量
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # 展平输入矩阵
        obs = obs.view(-1, self.flatten_size)
        #print(f"Input shape after flattening: {obs.shape}")
        x = torch.relu(self.fc1(obs))  # 第一个隐藏层 + ReLU
        x = torch.relu(self.fc2(x))  # 第二个隐藏层 + ReLU
        output = self.fc3(x)         # 输出层（线性）
        return output

# class FeedForwardNN(nn.Module):
#     def __init__(self, num_clients, num_features, out_dim):
#         super(FeedForwardNN, self).__init__()
#
#         # 将输入矩阵展平成一维向量
#         self.flatten_size = num_clients * num_features
#
#         # 定义全连接层
#         print(f"network input is :{self.flatten_size},out put is {out_dim}")
#         self.layer1 = nn.Linear(self.flatten_size, 256)
#         self.layer2 = nn.Linear(256, 256)
#         self.layer3 = nn.Linear(256, 256)
#         self.layer4 = nn.Linear(256, out_dim)
#
#     def forward(self, obs):
#         # 如果输入是 numpy 数组，先转换为 PyTorch 张量
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
#
#         # 展平输入矩阵
#         obs = obs.view(-1, self.flatten_size)
#         print(f"Input shape after flattening: {obs.shape}")
#
#         # 前向传播
#         activation1 = torch.relu(self.layer1(obs))
#         print(f"Shape after layer1: {activation1.shape}")
#         activation2 = torch.relu(self.layer2(activation1))
#         print(f"Shape after layer2: {activation2.shape}")
#         activation3 = torch.relu(self.layer3(activation2))
#         print(f"Shape after layer3: {activation3.shape}")
#         output = self.layer4(activation3)
#         print(f"Output shape: {output.shape}")
#
#         return output

# class CNNPolicyNetwork(nn.Module):
#     def __init__(self, num_clients, num_features, out_dim):
#         super(CNNPolicyNetwork, self).__init__()
#
#         # 定义卷积层，输入维度是 1 个通道
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#
#         # 全连接层，将卷积结果展平后输入
#         self.fc1 = nn.Linear(32 * num_clients * num_features, 128)
#         self.fc2 = nn.Linear(128, out_dim)
#
#     def forward(self, obs):
#         # 如果输入是 numpy 数组，先转换为 PyTorch 张量
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
#
#         # 将输入矩阵添加一个通道维度，适应卷积层输入 (batch_size, 1, num_clients, num_features)
#         obs = obs.unsqueeze(1)
#
#         # 卷积层 + 激活函数
#         x = torch.relu(self.conv1(obs))
#         x = torch.relu(self.conv2(x))
#
#         # 展平卷积输出
#         x = x.view(x.size(0), -1)
#
#         # 全连接层
#         x = torch.relu(self.fc1(x))
#         output = self.fc2(x)
#         return output
