import torch,random
import torch.nn as nn
import numpy as np
import time,copy
from flcore.clients.clientbase import Client
from utils.privacy import *
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientAVG(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        self.loss_decay=args.loss_decay
        self.global_client_profile=[]
        self.enable_dropout=args.enable_dropout
        self.nextClientDropoutRatio = None

    def train(self,queue):
        #1.--------- score = -1
        score = -1
        LocalDropoutRatio = 0 if self.nextClientDropoutRatio == None or not self.enable_dropout else self.nextClientDropoutRatio[
            self.id]
        dropout_ratio = LocalDropoutRatio
        trainedModels = []
        preTrainedLoss = []
        trainedSize = []
        trainSpeed = []
        virtualClock = []
        ranClients = []
        local_trained = 0
        count = 0
        last_model_tensors = []
        for idx, param in enumerate(self.model.parameters()):
            last_model_tensors.append(copy.deepcopy(param.data))
        epoch_train_loss = None
        self.loss=nn.CrossEntropyLoss(reduction='none')
        #-------------------

        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()


        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()
        run_start = time.time()
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
                #------------------------
                # only measure the loss of the first epoch
                if step==1:
                    local_trained += len(y)
                    temp_loss = 0.
                    # loss_list = loss.tolist() if args.task != 'nlp' else [loss.item()]
                    loss_list = loss.tolist()
                    for l in loss_list:
                        temp_loss += l ** 2
                    loss_cnt = len(loss_list)
                    temp_loss = temp_loss / float(loss_cnt)
                    if epoch_train_loss is None:
                        epoch_train_loss = temp_loss
                    else:
                        epoch_train_loss = (1. - self.loss_decay) * epoch_train_loss + self.loss_decay * temp_loss
                count += len(y)
                loss = loss.mean()  # 对损失取平均值
                # ------------------------------

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



        # self.model.cpu()
        #
        #---------------------------------------------------------------
        time_spent = time.time() - run_start
        if count > 0:
            speed = time_spent / float(count)
        if self.id in self.global_client_profile:
            time_cost = self.global_client_profile[self.id ][0] * count + self.global_client_profile[self.id ][1]
        else:
            time_cost = time_spent
        model_param = [(param.data - last_model_tensors[idx]).cpu().numpy() * (random.uniform(0, 1) >= dropout_ratio)
                       for idx, param in enumerate(self.model.parameters())]
        trainedModels.append(model_param)
        preTrainedLoss.append(epoch_train_loss if score == -1 else score)
        trainedSize.append(local_trained)
        trainSpeed.append(str(speed) + '_' + str(count))
        virtualClock.append(time_cost)
        ranClients.append(self.id)
        #---------------------------------------------------------------------
        #print("ssss:",preTrainedLoss,trainedSize,trainSpeed,virtualClock,ranClients)
        isComplete=True
        testResults=None
        # queue.put({self.id: [trainedModels, preTrainedLoss, trainedSize, isComplete, ranClients, trainSpeed, testResults,
        #                   virtualClock]})

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")


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

    def test_metrics_global(self,model):
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

    def train_metrics_global(self,model):
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




