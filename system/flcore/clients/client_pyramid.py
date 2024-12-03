import torch, random
import torch.nn as nn
import numpy as np
import time, copy
from flcore.clients.clientbase import Client
from utils.privacy import *
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientPyramid(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        self.loss_decay = args.loss_decay
        self.local_epochs=args.upload_epoch
        self.global_client_profile = []
        self.enable_dropout = args.enable_dropout
        self.nextClientDropoutRatio = None

    def train(self, queue):
        #print("~~"*20,f"client training is start ,client is {self.id}")
        # 1.--------- score = -1
        score = -1
        LocalDropoutRatio = 0 if self.nextClientDropoutRatio == None or not self.enable_dropout else \
        self.nextClientDropoutRatio
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
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # -------------------

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
        #print(f"client is{self.id},epoch is  {max_local_epochs},drop ratio is{self.nextClientDropoutRatio}")
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        #print(self.id, "max_local_epochs is ", max_local_epochs,len(trainloader))
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
                # ------------------------
                # only measure the loss of the first epoch

                if step == 0:
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
        # ---------------------------------------------------------------
        time_spent = time.time() - run_start
        if count > 0:
            speed = time_spent / float(count)
        if self.id in self.global_client_profile:
            time_cost = self.global_client_profile[self.id][0] * count + self.global_client_profile[self.id][1]
        else:
            time_cost = time_spent
        model_param = [(param.data - last_model_tensors[idx]).cpu().numpy() * (random.uniform(0, 1) >= dropout_ratio)
                       for idx, param in enumerate(self.model.parameters())]

        #print("client local training",self.id,epoch_train_loss,local_trained,str(speed) + '_' + str(count),time_cost)
        trainedModels.append(model_param)
        preTrainedLoss.append(epoch_train_loss if score == -1 else score)
        trainedSize.append(local_trained)
        trainSpeed.append(str(speed) + '_' + str(count))
        virtualClock.append(time_cost)
        ranClients.append(self.id)
        # ---------------------------------------------------------------------
        # print("ssss:",preTrainedLoss,trainedSize,trainSpeed,virtualClock,ranClients)
        isComplete = True
        testResults = None
        queue.put({self.id: [trainedModels, preTrainedLoss, trainedSize, isComplete, ranClients, trainSpeed, testResults,
                          virtualClock]})
        #print(f"client is {self.id},queue size is {queue.qsize()}")
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



