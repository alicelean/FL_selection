import torch
import os
import numpy as np
import h5py
import copy
import time,math
import random
import pandas as pd
from utils.distance import jensen_shannon_distance
from utils.data_utils import read_client_data
from utils.dlg import DLG
from sklearn.metrics import confusion_matrix
import ast
import matplotlib.pyplot as plt
import numpy as np
# Programpath="/Users/alice/Desktop/python/PFL/"
# mpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Programpath = "/".join(mpath.split("/")[:-1])
# print("serverbase",mpath,Programpath)

from utils.vars import Programpath,Aphla

#Programpath=" /home/alice/Desktop/python/PFL/"
class Server(object):
    def __init__(self, args, times,filedir="test"):
        #记录数据文件的位置，同一个数据集不同程度的数据分布差异
        self.filedir=args.dataset
        self.select_mode = args.select_mode
        self.method =None
        self.fix_ids = False
        self.select_idlist = []
        self.randomSelect = False
        self.Budget = []
        #聚合误差
        self.aggreErr=[]
        # 设置client数据
        self.client_batch_size = args.batch_size
        self.client_learning_rate = args.local_learning_rate
        self.client_local_epochs = args.local_epochs
        self.fix_ids = False
        self.ISAAW=False
        self.programpath=Programpath
        self.device = args.device
        if args.dataset == 'agnews':
            self.labellenght =4
        if args.dataset=='Cifar100':
            self.labellenght =100
        if args.dataset == 'Cifar10':
            self.labellenght =10
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            self.labellenght =10
        if args.dataset == 'Tiny-imagenet':
            self.labellenght = 200
        # 记录所有的weights
        self.allweights = []
        self.total_time=36000
        self.alpha=Aphla





        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

    def set_clients_origin(self, clientObj):
        print("**************************1.INfo,serverbase set_clients ,init data********************")
        samples=0
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples+=len(train_data)+len(test_data)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.sizerate=(client.train_samples+client.test_samples)/samples
        # 根据label 来计算distance
        self.setdistance()
            #print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self,round=-1):
        selected_clients=[]
        if round==-1:
            if self.random_join_ratio:
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            if self.select_mode == 'pyramidFy':
                pass
                if len(self.sampledClientSet)>self.current_num_join_clients:
                    selected_ids = list(np.random.choice(list(self.sampledClientSet), self.current_num_join_clients, replace=False))
                    print(" pyramidFy selected_ids select set",selected_ids,self.sampledClientSet,self.current_num_join_clients)
                else:
                    selected_ids=list(self.sampledClientSet)
                    print(" pyramidFy selected_ids set to list", selected_ids,self.sampledClientSet,self.current_num_join_clients)
                for client in self.clients:
                    if client.id in selected_ids:
                        selected_clients.append(client)
                for i in selected_clients:
                    print("select client i ",i,type(i))

            if self.select_mode == 'random':
                selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
                #flcore.clients.clientavg.clientAVG
                # for i in selected_clients:
                #     print("select client i ",i,type(i))

            return selected_clients

        else:
            # 按照制定方式选择clients,self.select_idlist[group],设定每一轮要选择的client,方便对不同算法进行比较
            selected_clients = []
            ids = []
            for c in self.clients:
                if c.id in self.select_idlist[round]:
                    selected_clients.append(c)
                    ids.append(c.id)
            print(f"INFO:----------fix client id is :group is {round} select id list is:", ids, type(selected_clients[0]))
            return selected_clients


    def send_models(self):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)

        # add更新模型参数，将globalmodel复制给本地模型----------------------
        for client in self.selected_clients:
            if self.ISAAW:
                client.local_initialization(self.global_model, round)
        #------------------------------------

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        '''
        根据设定的客户端丢失率、时间阈值和客户端的训练时间消耗，
        选择符合条件的活跃客户端，并收集其模型和样本权重。最后，对样本权重进行归一化，以便后续在联邦学习中使用。
        Returns:

        '''
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        active_clients=self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples















    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w



    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metricswithMosel(self,model=None):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def test_metrics_global(self,model):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_global(model)
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_global(self,model):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_global(model)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def train_metrics_last(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        truelabels=[]
        predictlabels=[]
        for c in self.clients:
            train_num, truelabel, predictlabel= c.train_metrics_last()
            num_samples.append(train_num)
            for i in truelabel:
                truelabels.append(i)
            for i in predictlabel:
                predictlabels.append(i)


        ids = [c.id for c in self.clients]

        return ids,num_samples, truelabels,predictlabels

    # evaluate selected clients
    def test_metrics_center(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            if c.id==0:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)

        ids = [0]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_center(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            if c.id == 0:
                cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl * 1.0)
        ids = [0]

        return ids, num_samples, losses
    def evaluate_center(self, group, acc=None, loss=None):
        stats = self.test_metrics_center()
        #ids, num_samples, losses
        stats_train = self.train_metrics_center()

        try:
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_acc = 0.0

        try:
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_auc = 0.0

        try:
            #平均损失，print(stats_train)
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        except ZeroDivisionError:
            train_loss = 0.0

        try:
            accs = [a / n for a, n in zip(stats[2], stats[1])]
        except ZeroDivisionError:
            accs = [0.0] * len(stats[2])

        try:
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
        except ZeroDivisionError:
            aucs = [0.0] * len(stats[3])

        # test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]


    def evaluate(self, group, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        try:
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_acc = 0.0

        try:
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_auc = 0.0

        try:
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        except ZeroDivisionError:
            train_loss = 0.0

        try:
            accs = [a / n for a, n in zip(stats[2], stats[1])]
        except ZeroDivisionError:
            accs = [0.0] * len(stats[2])

        try:
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
        except ZeroDivisionError:
            aucs = [0.0] * len(stats[3])

        # test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
    def evaluate_global(self, group,model=None, acc=None, loss=None):
        stats = self.test_metrics_global(model)
        stats_train = self.train_metrics_global(model)
        try:
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_acc = 0.0

        try:
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_auc = 0.0

        try:
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        except ZeroDivisionError:
            train_loss = 0.0

        try:
            accs = [a / n for a, n in zip(stats[2], stats[1])]
        except ZeroDivisionError:
            accs = [0.0] * len(stats[2])

        try:
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
        except ZeroDivisionError:
            aucs = [0.0] * len(stats[3])

        # test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]

    def evaluatewithModel(self, group, model=None,acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        try:
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_acc = 0.0

        try:
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_auc = 0.0

        try:
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        except ZeroDivisionError:
            train_loss = 0.0

        try:
            accs = [a / n for a, n in zip(stats[2], stats[1])]
        except ZeroDivisionError:
            accs = [0.0] * len(stats[2])

        try:
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
        except ZeroDivisionError:
            aucs = [0.0] * len(stats[3])

        # test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        #增加新的client
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            #将全局模型参数复制给client
            client.set_parameters(self.global_model)
            for e in range(self.fine_tuning_epoch):
                print(f"fine_tuning_epoch is {fine_tuning_epoch}")
                client.train()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    #add
    def read_selectInfo(self,file_path):
        print("INFO:---------Using fix select id list--------------------------------")
        print(file_path)
        data = pd.read_csv(file_path)

        print(data)
        rounds = data['global_rounds'].tolist()
        # print(data['ids'].tolist()[0])
        ids = [ast.literal_eval(i) for i in data['id_list'].tolist()]
        # print("rounds",rounds)
        # print("ids",ids)
        id_dict = {}
        for i in range(len(rounds)):
            id_dict[rounds[i]] = ids[i]
        print(id_dict)
        return id_dict

    def evaluate_origin(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    def writeparameters(self):
       # -----------1.写入超参数
       # 1:minist,
       redf = pd.DataFrame(
           columns=["dataset", "global_rounds", "client_enpoches", "client_batch_size", "client_learning_rate", "ratio",
                    "client_num", "Dirichlet alpha"])
       redf.loc[len(redf) + 1] = ["dataset", "global_rounds", "client_enpoches", "client_batch_size",
                                  "client_learning_rate", "ratio", "client_num", "Dirichlet alpha"]
       redf.loc[len(redf) + 1] = [self.dataset, self.global_rounds, self.client_local_epochs, self.client_batch_size,
                                  self.client_learning_rate, self.join_ratio, self.num_clients, 0.1]
       path = self.programpath + "/res/" + self.method + "/" + self.dataset + "_canshu.csv"
       redf.to_csv(path, mode='a', header=False)
       print("-------------------------------------------------------------------------write exe canshu,txt path is",path)
       # ---------------------------------------

    def writeclientInfo(self):
        '''
        写入设置的clients数据信息
        Returns:

        '''
        clientinfo = []
        dataname = ["id", "train_len", "test_len","sizerate", "train_slow", "send_slow","label"]
        # 如果文件夹不存在，创建文件夹
        folder_path=self.programpath + "res/" + self.method
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = folder_path+ "/clientsInfo.csv"
        for c in self.clients:
            clientinfo.append([c.id, c.train_samples, c.test_samples,c.sizerate, c.train_slow, c.send_slow,c.label])
        redf = pd.DataFrame(columns=dataname)
        for value in clientinfo:
            redf.loc[len(redf) + 1] = value
        redf.to_csv(path, mode='a', header=False)

    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        print("train_slow,send_slow", self.num_clients, self.train_slow_clients, self.send_slow_clients)
        # 读取所有的训练数据和测试数据
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples += len(train_data) + len(test_data)
            client = clientObj(self.args,
                               id=i,
                               traindata=train_data,
                               testsdata=test_data,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.sizerate = (client.train_samples + client.test_samples) / samples
        # 根据label 来计算distance
        self.setdistance()

        # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def select_weight_vector(self):
        active_distance = 0
        activelabel = [0 for i in range(10)]
        active_train_samples = 0

        for client in self.selected_clients:
            active_train_samples += client.train_samples
            for j in range(10):
                activelabel[j] += client.label[j]
        for client in self.selected_clients:
            # 计算参与训练的client分布差异
            client.distance = jensen_shannon_distance(client.label, activelabel)
            active_distance += client.distance

        for client in self.selected_clients:
            scale_factor = 0.1
            jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
                    client.train_samples / active_train_samples)

            client.AAW.initweight = jsweight
            client.AAW.initweight = client.train_samples / active_train_samples
            # 第一次参与训练需要初始化它的weight，weight 按照js计算weight的方式进行
            if not client.AAW.hasinit:
                client.AAW.init_weight(self.global_model)
                client.AAW.hasinit = True


    def weight_from_distance2(self,distance, scale_factor=0.1, exponent=0.5):
        '''
        distance：表示数据分布的距离。
        scale_factor：用于调整权重的比例因子。
        exponent：用于控制指数函数的指数。
        确定合适的scale_factor和exponent值是根据具体问题和数据分布而定的，没有固定的通用值。你可以根据实际情况进行实验和调整，以找到适合你问题的最佳值。

一般来说，scale_factor的选择可以考虑数据的范围和分布。如果数据的范围很大，可以选择较大的scale_factor，以使权重下降得更快。如果数据的范围较小，可以选择较小的scale_factor，以使权重下降得更慢。

对于exponent，它控制指数函数的指数，可以调整权重下降的速率和曲线的形状。较大的exponent会使权重下降得更快，而较小的exponent会使权重下降得更慢。你可以根据问题的需求和期望的权重分布形状进行调整。

建议尝试不同的值并观察权重的变化和模型的表现。通过反复实验和调整，你可以找到适合你问题的最佳scale_factor和exponent值。
        Args:
            scale_factor:
            exponent:

        Returns:

        '''
        if distance == 0:
            return 1.0
        else:
            return math.exp(-exponent * distance * scale_factor)

    def plot_dataset(self):
        print("start plot data set")
        label = []
        for client in self.clients:
            label.append(client.label)
        n_clients = len(label)
        n_classes = len(label[0])
        # 创建一个数组来存储每个类别的计数
        class_counts = np.zeros(n_classes)
        # 遍历每个客户端的标签数据，累加每个类别的计数
        for client_label in label:
            class_counts += np.array(client_label)

        # 绘制直方图
        plt.figure(figsize=(10, 5))
        plt.hist(label, stacked=True, bins=np.arange(n_classes + 1) - 0.5, rwidth=0.8)
        plt.xlabel(self.dataset+' Class (Dirichlet Parameter='+str(self.alpha)+')')
        plt.ylabel('Count')
        plt.xticks(np.arange(n_classes))
        legend = plt.legend(['Client {}'.format(i) for i in range(n_clients)], loc='upper right',
                            bbox_to_anchor=(1, 1), ncol=2)
        plt.savefig("/Users/alice/Desktop/python/PFL/res/dataset/" + self.dataset + str(self.alpha) + ".png",
                    format='png', bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.show()

    # def plot_CM(self):
    #     '''
    #     绘制混淆矩阵热力图，保存预测标签和实际标签的数据
    #     Returns:
    #
    #     '''
    #     print("start plot_CM")
    #     ids, num_samples, true_labels, predicted_labels = self.train_metrics_last()
    #     # print("ids, num_samples, truelabels, predictlabels", ids, num_samples, truelabels, predictlabels)
    #
    #     true_labels = np.concatenate([label.numpy() for label in true_labels])
    #     predicted_labels = np.concatenate([label.numpy() for label in predicted_labels])
    #     # 保存true_labels和predicted_labels到文件
    #     tpath = self.programpath + "/res/" + self.method + '/true_labels_' + self.alpha + '.npy'
    #     ppath = self.programpath + "/res/" + self.method + '/predicted_labels_' + self.method + self.alpha + '.npy'
    #     picpath = self.programpath + "/res/" + self.method + '/CM_' + self.method + self.alpha + '.eps'
    #     np.save(tpath, true_labels)
    #     np.save(ppath, predicted_labels)
    #
    #     # 从文件中读取true_labels和predicted_labels
    #     true_labels = np.load(tpath)
    #     predicted_labels = np.load(ppath)
    #     # 计算混淆矩阵
    #     confusion = confusion_matrix(true_labels, predicted_labels)
    #     #print("confusion_matrix", confusion_matrix)
    #     # 可视化混淆矩阵
    #     classes = np.unique(true_labels)  # 获取所有类别
    #
    #     # 创建热力图
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    #     plt.xlabel("Predicted Labels")
    #     plt.ylabel("True Labels")
    #     plt.title("Confusion Matrix")
    #     # 保存图片为EPS格式
    #     plt.savefig(picpath, format='eps')
    #     plt.show()

    def get_randomseed(self):

        start_seed = 0  # 随机种子的起始值
        end_seed = 10000  # 随机种子的结束值
        # 生成随机种子池
        seed_pool = list(range(start_seed, end_seed + 1))

        # 从种子池中随机选择k个种子
        self.random_seed_list = random.sample(seed_pool, self.round)


        #print("随机种子列表:", self.random_seed_list)

    def calculate_hellinger_distance(self, p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)
        hellinger_distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2)) / np.sqrt(2)
        return hellinger_distance
    def setdistance(self):
        alllabel = [0 for i in range(self.labellenght)]
        alldistance = 0
        for client in self.clients:
            for j in range(self.labellenght):
                alllabel[j] += client.label[j]

        for client in self.clients:
            client.distance = self.calculate_hellinger_distance(client.label, alllabel)
            alldistance += client.distance
        for client in self.clients:
            client.alldistance = alldistance
            client.alllabel = alllabel

    def read_fix_id(self):
        #读取代码
        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + str(
                self.num_clients) + "_" + str(self.join_ratio) + "/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            print("self path is :", file_path)
            self.select_idlist = self.read_selectInfo(file_path)

    def get_selected_clients(self,i):
        # 设置不同的选择方式
        if self.fix_ids:
            self.selected_clients = self.select_clients(i)
        else:
            self.selected_clients = self.select_clients()
            # ----------------------3.写入每次选择的client的数据----------------------------
            ids = []
            for client in self.selected_clients:
                ids.append(client.id)
        return ids

    def addvalue(self, res):
        #写入评估算法
        resc = [self.filedir]
        for line in res:
            resc.append(line)
        return resc
    def write_selectids(self,select_id):
        #写入选择的客户端信息
        if not self.fix_ids:
            folder_path=self.programpath + "/res/selectids/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            idpath = folder_path + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            redf = pd.DataFrame(columns=["global_rounds", "id_list"])
            #redf.loc[len(redf) + 1] = ["*********************", "*********************"]
            redf.loc[len(redf) + 1] = ["global_rounds", "id_list"]
            for v in range(len(select_id)):
                redf.loc[len(redf) + 1] = select_id[v]
            redf.to_csv(idpath, mode='a', header=False)
            print("write select id list ", idpath)
    def write_acc(self,colum_value):
        print("colum_value is ",colum_value)

        #写入评估指标
        folder_path = self.programpath + "/res/" + self.method + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        accpath = folder_path + self.dataset + "_acc.csv"
        allpath = self.programpath + "/res/ " + self.dataset + "_allacc_" + str(self.join_ratio) + "_" + str(
            self.num_clients) + "_" + str(self.alpha) + ".csv"

        colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC","select_mode"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            colum_value[i].append(self.select_mode)
            redf.loc[len(redf) + 1] = colum_value[i]
        redf.to_csv(accpath, mode='a', header=False)
        redf.to_csv(allpath, mode='a', header=False)
        print("success training write acc txt", accpath,allpath)
        print(colum_value)
    def wirte_time(self):
        #记录一下时间资源消耗
        timepath = self.programpath + "/res/timeCost_" + self.dataset + str(self.alpha) + ".csv"
        redf = pd.DataFrame(
            columns=["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"])
        redf.loc[len(redf) + 1] = ["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"]
        redf.loc[len(redf) + 1] = [self.dataset, self.method, self.global_rounds, self.join_ratio,
                                   sum(self.Budget[1:]) / len(self.Budget[1:]), sum(self.Budget[1:]), self.Budget[1:]]
        redf.to_csv(timepath, mode='a', header=False)
        print("success training write acc txt", timepath)
    def write_info(self,select_id,colum_value):
        self.write_selectids(select_id)
        self.write_acc(colum_value)
        self.wirte_time()

