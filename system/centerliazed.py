import time
import copy
from flcore.clients.client_centeralized import clientCenter
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from utils.data_utils import read_client_data
class Center(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.client=None
        self.set_clients( clientCenter)




    def set_clients(self, clientCenter):
        totalsamples = 0
        all_traindata = []
        all_testdata = []
        train_slow=None
        send_slow=None
        #print("train_slow,send_slow", self.num_clients, self.train_slow_clients, self.send_slow_clients)
        # 读取所有的训练数据和测试数据
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            totalsamples += len(train_data) + len(test_data)
            #print("train_slow,send_slow",train_slow,send_slow)
            for j in test_data:
                all_testdata.append(j)
            for j in train_data:
                all_traindata.append(j)
        # 本地客户端上数据是所有数据
        self.client = clientCenter(self.args,
                                id=0,
                                traindata=all_traindata,
                                testsdata=all_testdata,
                                train_samples=len(all_traindata),
                                test_samples=len(all_testdata),
                                train_slow=False,
                               send_slow=False)
        print(len(all_testdata),len(all_traindata))


    def train(self):
        colum_value = []
        select_id = []
        s_t = time.time()
        for i in range(self.global_rounds + 1):
            self.client.train()
            ct, num_samples, auc = self.client.test_metrics()
            print("ssssssss",ct, num_samples, auc)