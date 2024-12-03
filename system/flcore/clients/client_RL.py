import torch, random
import torch.nn as nn
import numpy as np
import time, copy
from flcore.clients.clientbase import Client
from utils.privacy import *
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientRL(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        self.loss_decay = args.loss_decay
        self.global_client_profile = []
        self.enable_dropout = args.enable_dropout
        self.nextClientDropoutRatio = None


    def train(self,tr):
        #if client.id ==0:
        #print(f"client {self.id} start train round is {round}-------------------- ")
        # 1.--------- score = -1
        self.stale=tr-self.td
        last_model_tensors = []
        for idx, param in enumerate(self.model.parameters()):
            last_model_tensors.append(copy.deepcopy(param.data))
        self.loss = nn.CrossEntropyLoss(reduction='none')

        # -------------------

        trainloader = self.load_train_data()
        # self.model.to(self.device)

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        flag=True
        i=0
        #统计资源消耗
        self.costedResoure=0
        while flag:
            #print(f"client {self.id} start the {i} th local epoch,max_local_epochs is {max_local_epochs} ")
            #-------取全局模型-----
            start_time = time.time()
            #记录获取全局模型的时间
            self.td=time.time()+self.communicate
            for new_param, old_param in zip(self.globalModel.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
            self.costedResoure+=self.communicate
            self.model.train()
            #------------------------

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
                    loss =loss.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                #缓存每次训练后的模型local model---------
                for new_param, old_param in zip(self.model.parameters(), self.localmodel.parameters()):
                    old_param.data = new_param.data.clone()

                #-----------------------------------------------
                #计算本地模型的损失
                losses, train_num=self.train_metricsWithmodel()
                self.currentloss = (losses * 1.0)
                self.avgloss = (losses * 1.0) / train_num
                #print(f"update client is{self.id},self.currentloss is {self.currentloss}")
                #计算资源差异
                # time.sleep(self.compute)
            self.costedResoure += self.compute*max_local_epochs
            self.costedResoure += self.communicate

            flag=False
        #print(f"client {self.id} end the {i} th local epoch,max_local_epochs is {max_local_epochs} ")

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        #print(f"Client {self.id} success!")
        # if self.privacy:
        #     eps, DELTA = get_dp_params(privacy_engine)
        #     print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

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

    def train_metricsWithmodel(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.localmodel.eval()
        closs = nn.CrossEntropyLoss()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.localmodel(x)
                #print("model print",output ,y)
                #print("test:",loss.shape)
                loss = closs(output, y)
                loss=loss.mean()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num




    def test_metrics_global(self, model):
        testloaderfull = self.load_test_data()
        if testloaderfull is None:
            print("client test_metrics Error: Failed to load test data.")
            return None

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        # 记录NaN值的数量
        nan_x = 0
        nan_y = 0
        nan_output = 0
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 检查输入数据中是否存在NaN值
                if self.dataset != 'agnews':
                    if torch.isnan(x).any():
                        nan_x += 1
                        continue
                    if torch.isnan(y).any():
                        nan_y += 1

                output = model(x)

                # 检查模型输出中是否存在NaN值
                if self.dataset != 'agnews':
                    if torch.isnan(output).any():
                        nan_output += 1
                        continue

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        # self.model.cpu()
        # self.save_model(self.model, 'model')
        nan_count = nan_x + nan_y + nan_output
        if nan_count > 0:
            nan_ratio = nan_count / len(testloaderfull)  # 计算NaN值在测试数据中的比例
            print(
                f"client {self.id} ,nan_x {nan_x},nan_y {nan_y},nan_output {nan_output},total NaN value ratio in test data: {nan_ratio:.2%}")
        if nan_count != len(testloaderfull):
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            return test_acc, test_num, auc
        else:
            print(f"ERROR:testloaderfull {len(testloaderfull)},nan_count {nan_count}, test_num {test_num}")
            return 0, test_num, 0

    def train_metrics_global(self, model):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                # print("model print",output ,y)
                loss = self.loss(output, y)
                # print("test:",loss.shape)
                loss = loss.mean()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num




