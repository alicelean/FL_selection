import time,os,queue
from flcore.clients.client_pyramid import clientPyramid
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from selection.PramidFy import *
from utils.data_utils import read_client_data
import random,copy

class FedPyramid(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedPyramid"
        self.select_mode="pyramidFy"
        self.select_mode = "oort"
        # 设置客户段基本信息
        self.client_path = self.programpath + "/res/" + self.method + "/" + self.dataset + "client_profile.pkl"
        self.selection = PramidFy(args, self.current_num_join_clients,self.client_path)
        self.selection.mode=self.select_mode
        #print("PramidFy num_join_clients is ",self.num_join_clients)
        self.queue = queue.Queue()
        #客户端信息队列
        self.InfoQueue= queue.Queue()
        self.stop_signal=queue.Queue()
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPyramid)

        if not os.path.exists(self.client_path):
            print(f"self.client_path is {self.client_path}")
            self.set_global_client_profile(self.client_path)

        #在初始化客户端，在注册器中注册每个客户端的信息------
        #探索和利用，有一个超参数
        self.sampledClientSet=set()
        #print(f" self.selection.mode is { self.selection.mode}")
        #初始化客户端采集器
        self.clientSampler = self.selection.initiate_sampler_query(self.selection.mode,self.InfoQueue, args.num_clients)
        for nextClientIdToRun in range(args.num_clients):
            self.clientSampler.clientOnHost([nextClientIdToRun], nextClientIdToRun)
            self.sampledClientSet.add(nextClientIdToRun)
            self.clientSampler.clientLocalEpochOnHost([1], nextClientIdToRun)
            self.clientSampler.clientDropoutratioOnHost([0], nextClientIdToRun)

        print("after initiate_sampler_query self.queue",self.queue.qsize(),self.clientSampler.clientOnHosts)



        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def select_clients(self):
        selected_clients = []

        print("select num is :", self.current_num_join_clients)

        if self.select_mode == 'pyramidFy' or self.select_mode == 'oort':
                # 判定采集的客户端元组中数量
            if len(self.sampledClientSet) > self.current_num_join_clients:
                selected_ids = list(
                    np.random.choice(list(self.sampledClientSet), self.current_num_join_clients, replace=False))
                print(f" pyramidFy ,len(self.sampledClientSet) is {len(self.sampledClientSet)}", selected_ids, self.sampledClientSet,
                          self.current_num_join_clients)
            else:
                selected_ids = list(self.sampledClientSet)
                print(" pyramidFy selected_ids set to list", selected_ids, self.sampledClientSet,
                          self.current_num_join_clients)
            print(f"len(selected_ids)is {len(selected_ids)},selected id is :",selected_ids)
            for client in self.clients:
                if client.id in selected_ids:
                    selected_clients.append(client)
            self.selected_clients=selected_clients
            self.sampledClientSet=set(selected_ids)
            return selected_ids



    def train(self):
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
       # ————————————————————————————————————————————————————————————————————————————————

        for i in range(self.global_rounds+1):
            i+=1
            if self.cost_time > self.total_time or round == self.global_rounds:
                break
            s_t = time.time()
            #训练之前先计算客户端的状态

            # 新增2————————————————————————————————————————————————————————————————————————————————

            ids=self.select_clients()
            select_id.append([i, ids])

            # ————————————————————————————————————————————————————————————————————————————————
            #down load global model--------
            self.send_models()
            # local training----------------
            print(f"~~" * 20, f"local training is start,self.selected_clients is {len(self.selected_clients)} ")


            local_training_list = []
            threa = ThreadPoolExecutor(max_workers=self.num_clients)

            for client in self.selected_clients:
                client.train(self.queue)

            # 计算选中的客户端与总体的差距
            kl_div, js_div, emd = self.get_select_distance()
            # 新增：统计每一轮训练的资源消耗------
            highClientNum = 0
            round_time = []
            for client in self.clients:
                if client.id in ids:
                    client.select_time += 1
                    client.last_select = i
                    if client.isRichResource:
                        highClientNum += 1
                    round_time.append(client.costedResoure)
            self.cost_time += max(round_time)




            # for client in self.selected_clients:
            #     client.train(self.queue)




            print("~~" * 20, "local training is end")

            # 关闭线程池
            threa.shutdown(wait=True)

            #------------------compute client utinity--and select client -------------
            #print(f" 2.self.selection.mode is {self.selection.mode}")
            self.sampledClientSet=self.selection.run(self.global_model, self.queue, self.stop_signal, self.clientSampler,self.sampledClientSet,i)
            for client in self.clients:
                if client.id in self.sampledClientSet:
                    #print(client.id,"before",client.nextClientDropoutRatio,client.local_epochs)
                    client.nextClientDropoutRatio = self.clientSampler.getclientDropoutratioOnHost(client.id)[0]
                    client.local_epochs = int(self.clientSampler.getclientLocalEpochOnHost(client.id)[0])
                    #print(client.id,"after", client.nextClientDropoutRatio, client.local_epochs)
            #print(f" 3.self.selection.mode is {self.selection.mode}")
            print("###"*20,f"  round is {i} select client set is : {self.sampledClientSet}")
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            #评估训练后的全局模型在每个本地的损失
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #localResult=[self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
                # localResult = self.evaluate(i)
                res = self.evaluate_global(i,self.global_model)
                # res.append(distance)
                #print("res is:",res)
                #print("resg is:", res1)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc=self.addvalue(res)
                resc.append(kl_div)
                resc.append(js_div)
                resc.append(emd)
                resc.append(max(round_time))
                resc.append(highClientNum)

                colum_value.append(resc)
                #print("colum_value is",colum_value,resc is {resc})
                # ————————————————————————————————————————————————————————————————————————————————





            self.Budget.append(time.time() - s_t)
            print('-'*25,"Round is ",i,'-'*25, 'time cost', '-'*25, self.Budget[-1])
            print('-'*25,"Budget is :",self.Budget)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

       # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id,colum_value)
       # ————————————————————————————————————————————————————————————————————————————————

        self.save_results()
        #self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        #print("train_slow,send_slow", self.num_clients, self.train_slow_clients, self.send_slow_clients)
        # 初始化资源信息
        clientinf_path = self.programpath + "dataset/" + self.alpha + "/" + self.dataset + "/clientInfo.json"
        self.clientsResource = self.read_clients(clientinf_path)

        # 读取所有的训练数据和测试数据
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            tmp_dict = {}
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
            #   每个客户端和总体的距离，这里定义为客户端的数据量
            distanceVec = [len(train_data)+len(test_data)]
            sizeVec = [len(train_data)+len(test_data)]
            tmp_dict[i]=[]
            tmp_dict[i].append(distanceVec)
            tmp_dict[i].append(sizeVec)
            self.InfoQueue.put(tmp_dict)
            #self.selection.InfoQueue.put(tmp_dict)

        for client in self.clients:
            label=client.setlabel()
            client.sizerate = (client.train_samples + client.test_samples) / samples
            for j in range(len(label)):
               self.alllabel[j] +=label[j]
            client.compute = self.clientsResource[client.id]["compute"]
            client.communicate = self.clientsResource[client.id]["comm"]
            client.offline = self.clientsResource[client.id]["dropout"]
            client.costedResoure = client.communicate * 2 + client.compute
            self.clientsResource[client.id]["size"] = client.size

        # 根据label 来计算distance
        self.setdistance()

        # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def receive_models(self):
        '''
        根据设定的客户端丢失率、时间阈值和客户端的训练时间消耗，
        选择符合条件的活跃客户端，并收集其模型和样本权重。最后，对样本权重进行归一化，以便后续在联邦学习中使用。
        Returns:

        '''
        print(f"self.selected_clients is {len(self.selected_clients)}")
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

        active_clients = self.selected_clients

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
    def send_models(self):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)

        #将全局模型发送给每个客户端
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)









