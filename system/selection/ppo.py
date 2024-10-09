import argparse
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.distribution import Categorical
from paddle.io import RandomSampler, BatchSampler, Dataset
from visualdl import LogWriter

# Parameters
num_clients = 10  # 客户端数量
gamma = 0.99
buffer_capacity = 8000
batch_size = 64

# 客户端状态和准确率
class Client:
    def __init__(self, id):
        self.id = id
        self.state = np.random.rand(5)  # 示例状态
        self.accuracy = np.random.rand()  # 随机初始化准确率

    def update_accuracy(self):
        # 更新准确率的逻辑
        self.accuracy = np.random.rand()  # 替换为真实准确率更新逻辑

# 客户端选择策略网络
class Actor(nn.Layer):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_clients, 128)
        self.action_head = nn.Linear(128, num_clients)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), axis=1)
        return action_prob

# PPO算法
class PPO:
    def __init__(self):
        self.actor_net = Actor()
        self.buffer = []
        self.writer = LogWriter('./exp')
        self.actor_optimizer = optim.Adam(parameters=self.actor_net.parameters(), learning_rate=1e-3)

    def select_action(self, state):
        state = paddle.to_tensor(state, dtype="float32").unsqueeze(0)
        action_prob = self.actor_net(state)
        dist = Categorical(action_prob)
        action = dist.sample([1]).squeeze(0)
        return action.cpu().numpy()[0], action_prob[:, int(action)].numpy()[0]

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        states = paddle.to_tensor([t['state'] for t in self.buffer], dtype="float32")
        actions = paddle.to_tensor([t['action'] for t in self.buffer], dtype="int64").reshape([-1, 1])
        rewards = [t['reward'] for t in self.buffer]

        R = 0
        Gt = []
        for r in rewards[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = paddle.to_tensor(Gt, dtype="float32")

        for i in range(10):  # PPO 更新次数
            for index in BatchSampler(sampler=RandomSampler(Dataset(len(self.buffer))), batch_size=batch_size, drop_last=False):
                index = paddle.to_tensor(index)
                Gt_index = paddle.index_select(Gt, index).reshape([-1, 1])
                action_prob = self.actor_net(paddle.index_select(states, index))

                ratio = action_prob / paddle.index_select(actions, index)
                surr1 = ratio * Gt_index
                surr2 = paddle.clip(ratio, 1 - 0.2, 1 + 0.2) * Gt_index

                surr = paddle.concat([surr1, surr2], 1)
                action_loss = -paddle.min(surr, 1).mean()
                self.actor_optimizer.clear_grad()
                action_loss.backward()
                self.actor_optimizer.step()

        del self.buffer[:]  # 清空经验

def main():
    agent = PPO()
    clients = [Client(i) for i in range(num_clients)]

    for epoch in range(1000):
        total_accuracy = 0

        for client in clients:
            state = client.state
            action, action_prob = agent.select_action(state)

            # 更新客户端的准确率
            client.update_accuracy()
            total_accuracy += client.accuracy

            transition = {
                'state': state,
                'action': action,
                'action_prob': action_prob,
                'reward': total_accuracy  # 奖励为所有客户端的准确率之和
            }
            agent.store_transition(transition)

        if len(agent.buffer) >= batch_size:
            agent.update()

if __name__ == '__main__':
    main()
    print("end")
