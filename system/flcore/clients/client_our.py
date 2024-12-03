import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientOUR(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        params_l=list(self.model.parameters())
        self.currGrad = [(torch.ones_like(param.data) * 0).to(self.device) for param in params_l]
        self.round=0
        self.k=100000000000

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        #计算梯度


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def update_parameter(self, gradient, eta_P):
        """
        更新参数 P_{i,t+1} 依据公式 (e4)

        参数:
        - pit: 当前参数 P_{i,t} (标量)
        - gradients: 当前的梯度 (张量)
        - W_i: 当前的权重数组 (张量)
        - D_i: 当前数据大小 (标量)
        - D: 总数据大小 (标量)
        - eta_P: 学习率 (标量)
        - t: 当前时间步 (标量)
        - k: 常量 (标量)

        返回:
        - P_{i,t+1} 的更新值
        """
        # 计算 \tilde{\phi}(P_{i,t}) 和导数
        if self.round==0:
            self.pit=self.sizerate
        self.round += 1
        #print("init pit",self.pit)
        _,phi_deriv = self.phi_derivative(self.pit)
        params_l = list(self.model.parameters())
        # phi_deriv= [(torch.ones_like(param.data) * phi_deriv).to(self.device) for param in params_l]
        pit=[(torch.ones_like(param.data) * self.pit).to(self.device) for param in params_l]
        # 更新 P_{i,t+1}
        #print("before self.pit",phi_deriv)
        for p, allg in zip(pit,gradient):
            p.data = p- torch.mul(allg,0.001 *self.sizerate*self.sizerate)

        #print("梯度",pit)  # 打印梯度
        mean_gradient_norm = sum(torch.norm(g) for g in pit) / len(pit)
        #print("self.pit", mean_gradient_norm)
        self.pit=mean_gradient_norm.item()

    def phi(self, x):
        """计算函数 \tilde{\phi}(x) 的值"""
        # 计算 phi 值(-self.k * (x - 0.5)))
        #print("qqq",x,1.0 / (1 +np.exp(- (x - 0.5) )))
        return 1.0 / (1 + np.exp(- (x-0.5)))

    def phi_derivative(self, x):
        """计算函数 \tilde{\phi}(x) 的导数"""

        phi_value = self.phi(x)
        #print("phi_value",phi_value)
        # 计算导数
        return phi_value, self.k * phi_value * (1 - phi_value)

    # def phi(self, x):
    #     """计算函数 \tilde{\phi}(x) 的值"""
    #     # 确保输入是一个 tensor，如果是列表则将其转换为 tensor
    #     x_tensor = torch.tensor(x) if isinstance(x, list) else x
    #     # 计算 phi 值
    #     return 1 / (1 + torch.exp(-self.k * (x_tensor - 0.5)))
    #
    # def phi_derivative(self, x):
    #     """计算函数 \tilde{\phi}(x) 的导数"""
    #     # 确保输入是一个 tensor，如果是列表则将其转换为 tensor
    #     x_tensor = torch.tensor(x) if isinstance(x, list) else x
    #     phi_value = self.phi(x_tensor)
    #     # 计算导数
    #     return phi_value, self.k * phi_value * (1 - phi_value)

    def calculate_gradients(self, global_model: nn.Module):
        trainloader = self.load_train_data()
        globallosess = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=0)
        global_model.eval()  # 设置模型为评估模式
        print(self.id,len(trainloader))
        params_g = list(global_model.parameters())
        self.currGrad = [(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]
        for i, (x, y) in enumerate(trainloader):
            # 将输入数据移动到指定设备
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            # 前向传播
            output = global_model(x)
            # 计算损失
            loss = globallosess(output, y)
            # 清除之前的梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 保存当前梯度
            for para_g, upgrad in zip(params_g, self.currGrad):
                upgrad.data = upgrad + torch.mul(para_g.grad,1)
            #print(type(self.currGrad))
                # # 计算梯度张量的平均范数
                # mean_gradient_norm = sum(torch.norm(g) for g in self.updategrad) / len(self.updategrad)
                # # 判断平均范数是否小于阈值
                # if mean_gradient_norm < 0.00001:
                #     print("平均梯度范数小于阈值，停止训练")
            # 不调用 self.optimizer.step()，这样不会更新模型参数

    def localtrain(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.localmodel.train()

        # differential privacy
        if self.privacy:
            self.localmodel, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.localmodel, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.localmodel(x)
                loss = self.local_losss(output, y)
                self.localoptimizer.zero_grad()
                loss.backward()
                self.localoptimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")









