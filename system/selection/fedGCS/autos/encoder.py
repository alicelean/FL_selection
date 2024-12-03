from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        '''- 参数：
        - layers：网络层数（在子类中会使用）
        - vocab_size：词汇大小（即输入的特征维度，通常是词嵌入的大小）
        - hidden_size：隐藏层的大小（每个层的输出维度）
        - 作用：
        - 初始化编码器的基本结构，包括词嵌入层（nn.Embedding）和其他结构化属性。'''
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def infer(self, x, predict_lambda, direction='-'):
        '''参数：
            x：输入数据
            predict_lambda：一个预测值，用于在梯度上进行调整
            direction：梯度调整的方向，+ 或 -
            作用：
            根据输入数据 x 生成编码器输出，然后计算基于预测值的梯度。
            根据 direction 的值（+ 或 -），将梯度加或减到编码器输出。
            对新的输出进行归一化，并生成新的序列嵌入（new_seq_emb）。
            返回：
            encoder_outputs：原始编码器输出
            encoder_hidden：编码器的隐藏状态
            seq_emb：序列的嵌入表示
            predict_value：预测值
            new_encoder_outputs：根据梯度调整后的编码器输出
            new_seq_emb：调整后的序列嵌入'''


        encoder_outputs, encoder_hidden, seq_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass


class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout
                 ):
        '''参数：
        layers：LSTM 网络的层数
        vocab_size：词汇大小
        hidden_size：隐藏层的大小
        dropout：LSTM 中的 dropout 参数
        mlp_layers：MLP 中的层数
        mlp_hidden_size：MLP 中每层的隐藏单元数量
        mlp_dropout：MLP 中的 dropout 参数
        作用：
        初始化 LSTM 网络（self.rnn）并构建 MLP 网络（self.mlp）用于后续的回归任务。
        self.rnn 是一个多层的 LSTM 网络，用于处理时间序列数据或序列数据。
        self.mlp 是一个多层感知机，用于从 LSTM 的输出中进一步提取特征。
        self.regressor 是一个线性回归层，用于根据 LSTM 和 MLP 的输出进行最终预测。'''
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        '''x：输入的序列数据
        作用：

        对输入 x 进行词嵌入（self.embedding(x)），得到嵌入向量。
        通过 LSTM 对嵌入向量进行处理，获得编码器输出（encoder_outputs）和隐藏状态（encoder_hidden）。
        对 LSTM 输出进行归一化（F.normalize(out, 2, dim=-1)），并通过 MLP 进行处理，最后用回归层预测一个值。
        predict_value 是经过回归层预测后的结果，经过 Sigmoid 激活函数，输出一个介于 [0, 1] 之间的值。
        返回：

        encoder_outputs：LSTM 网络的输出（每个时间步的隐状态）
        encoder_hidden：LSTM 网络的隐藏状态
        seq_emb：序列的嵌入表示，取 LSTM 输出的均值
        predict_value：模型的最终预测结果'''
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)

        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value


def construct_encoder() -> Encoder:
    name = "rnn"
    size = 100
    info(f'Construct Encoder with method {name}...')
    if name == 'rnn':
        return RNNEncoder(
            layers=1,
            vocab_size=size + 1,
            hidden_size=64,
            dropout=0.0,
            mlp_layers=2,
            mlp_hidden_size=200,
            mlp_dropout=0.0
        )
    else:
        assert False
