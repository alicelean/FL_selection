import torch, random
import torch.nn as nn
import numpy as np
import time, copy
from flcore.clients.clientbase import Client
from utils.privacy import *
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientFedOORT(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        self.loss_decay = args.loss_decay
        self.local_epochs=args.upload_epoch
        self.global_client_profile = []
        self.enable_dropout = args.enable_dropout
        self.nextClientDropoutRatio = None

    def train(self):
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
                output = self.model(x)
                loss = self.loss(output, y)
                # ------------------------
                # only measure the loss of the first epoch
                if step == 0:
                    local_trained += len(y)
                    loss_list = loss.tolist()

                    temp_loss = 0
                    for l in loss_list:
                        temp_loss += l ** 2
                    temp_loss = temp_loss / float(len(loss_list))

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

        # ---------------------------------------------------------------
        time_spent = time.time() - run_start

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time_spent


