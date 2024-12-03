import numpy as np
import torch
import torch.nn as nn
import copy,statistics
import random,time
from torch.utils.data import DataLoader
from typing import List, Tuple
import math
import tensorflow as tf
class SLE:
    def __init__(self,
                sizerate:float,
                num_clients: int,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple],
                batch_size: int,
                rand_percent: int,
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu',
                threshold: float = 0.01,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID.
            loss: The loss function.
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.01
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.initweight = 0
        self.sizerate=sizerate
        self.num_clients = num_clients
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.SLE = None # Learnable local aggregation weights.
        self.timecost = []
        self.notadptive=False
        self.start_phase = True
        self.updategrad=[]
        self.hasinit=False
        self.global_model=None
        self.stop=False
        self.etaj=0.01
        self.currGrad = {}
        self.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    self.etaj, 100)


    def initSLE(self, global_model: nn.Module):
        '''
        利用js距离初始化聚合参数，形状与globalmodel相似
        Args:
            global_model:

        Returns:

        '''
        #实际上self.initweight =client.train_samples / active_train_samples
        if self.initweight==0:
            raise ValueError("%%%%%%%%%%%%%%%%%%%%%%%%%%ERROR:AAW.self.initweight is zero%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        params = list(global_model.parameters())
        self.SLE = [torch.mul(torch.ones_like(param.data), self.initweight).to(self.device) for param in params]

    def adaptiveSLE(self, global_model: nn.Module, local_model: nn.Module,round :int) -> None:
        """
        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """
        if self.notadptive or self.stop:
            return
        #初始化权重
        if not self.hasinit:
            self.initSLE(self.global_model)
            self.hasinit = True

        #所有数据
        local_model_t = copy.deepcopy(local_model)

        rand_loader = DataLoader(self.train_data, len(self.train_data), drop_last=True)
        params_g = list(global_model.parameters())  # 获取全局模型的参数列表
        params_l_t = list(local_model_t.parameters())

        global_model_t = copy.deepcopy(global_model)
        params_g_t = list(global_model_t.parameters())

        optimizer = torch.optim.SGD(params_g_t, lr=0)  # 使用全局模型的参数列表创建优化器
        #计算出所有的梯度
        self.calculate_gradients(global_model)
        # #梯度信息
        # self.updategrad=[(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]


        losses = []  # 记录损失值

        # 记录更新前的时间戳
        start_time = time.time()


        for pit, upgrad in zip(self.SLE, self.currGrad):
            pit.data = pit - torch.mul(upgrad, self.sizerate * self.sizerate*phi_derivative(pit))


        # 计算梯度张量的平均范数
        mean_gradient_norm = sum(torch.norm(g) for g in self.updategrad) / len(self.updategrad)
        # 判断平均范数是否小于阈值
        if mean_gradient_norm < 0.00001:
            print("平均梯度范数小于阈值，停止训练")
            self.stop=True
        if len(aloss)==0:
            print("aloss {aloss},rand_loader :{len(rand_loader)}")
        losses.append(statistics.mean(aloss))


        # 计算更新所使用的时间
        update_time = time.time() - start_time
        self.timecost.append(update_time)


        if statistics.mean(aloss)<0.5:
            self.stop = True

        # 平滑函数的近似







