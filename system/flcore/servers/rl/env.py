import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools


class FederatedLearningEnv(gym.Env):
    def __init__(self, num_clients, k):
        super(FederatedLearningEnv, self).__init__()
        self.num_clients = num_clients
        self.select_num = k  # 每次选择 k 个客户端
        self.feature=4
        #为每一个客户端输出一个价值。
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(num_clients,))
        # 状态空间：每个客户端的状态（如计算能力、数据量、延迟等）
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_clients, self.feature), dtype=np.float32)
        # 初始化每个客户端的状态
        self.client_states = self.initClientStates(self.num_clients, self.feature)


    def reset(self):
        # 重置环境状态：客户端的计算能力、数据量、延迟等
        self.client_states = self.initClientStates(self.num_clients, self.feature)
        return self.client_states,True

    def step(self, action):
        # 根据动作选择客户端组合，action 是组合的索引
        #print("action is ",action)
        selected_clients = action
        # 计算奖励，基于选择的客户端
        reward = self._calculate_reward(selected_clients)

        # 更新选择的客户端状态（随机变化）
        self.updateClientStates(selected_clients)

        done = False
        #什么情况下可以终止当次循环，什么情况下终止调整策略需要处理？
        terminated=False
        truncated=False

        return self.client_states, reward, terminated,truncated,done

    def updateClientStates(self,selected_clients):
        #print("self.client_states is ",self.client_states)
        pass

    def _calculate_reward(self, selected_clients):
        # 根据选择的客户端计算任务的完成情况，简单示例
        # 训练，聚合，计算每个客户端的准确率，计算总体准确率，计算总体损失，计算reward
        #print("selected_clients",selected_clients)
        reward = 1.0
        return reward

    def initClientStates(self,num_clients, feature):
        client_states = np.random.rand(num_clients, feature)
        print("get client states",client_states)
        return client_states

