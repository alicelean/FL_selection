import time,os,queue
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from selection.PramidFy import *
from utils.data_utils import read_client_data
import random,copy

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedAvg"

        self.selection = PramidFy(args,self.num_join_clients)
        self.queue = queue.Queue()
        self.InfoQueue= queue.Queue()
        self.stop_signal=queue.Queue()



        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        #在初始化客户端以后需要在注册器中注册每个客户端的信息
        self.sampledClientSet=set()
        self.clientSampler = self.selection.initiate_sampler_query(self.InfoQueue, args.num_clients)
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



    def train(self):
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        if self.fix_ids:
            self.read_fix_id()
       # ————————————————————————————————————————————————————————————————————————————————

        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————

            ids=self.get_selected_clients(i)
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————


            self.send_models(ids)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #self.evaluate()
                #res = self.evaluate(i)
                res = self.evaluate_global(i,self.global_model)
                print("res is:",res)
                #print("resg is:", res1)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc=self.addvalue(res)
                colum_value.append(resc)
                #print("colum_value is",colum_value,resc is {resc})
                # ————————————————————————————————————————————————————————————————————————————————

            # 参与客户端训练本地数据（所有客户端参与训练）
            # for client in self.clients:
            #     client.selected = True
            #     client.train(self.queue)
                #client.localtrain()

            tmp_r_list = []
            t = ThreadPoolExecutor(max_workers=5)
            j=0
            for client in self.clients:
                if client.id in ids:
                #print(f"client is {client.id,i},client training start")
                    tmp_r_list.append(t.submit(client.train, self.queue))
                    tmp_r_list[j].result()
                    j+=1



            self.sampledClientSet=self.selection.run(self.global_model, self.queue, self.stop_signal, self.clientSampler,self.sampledClientSet,i)
            print("###"*20,f"  round is {i}self.selection.run {self.sampledClientSet}")
            # threads = [Thread(target=client.train(queue))
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            #------
            # aggreError = []
            # for client in self.clients:
            #     # i=0时，不是聚合得到的全局模型。不需要计算全局模型在本地的损失
            #     if i > 0 and client.isselected:
            #         # 上一轮训练得到的全局模型，还没有开始本地训练，但是已经传送全局模型过去了。
            #         client.calculate_gobal_loss(self.global_model)
            #         # print(client.globalloss, client.localloss)
            #         # print(f"{i},client {client.id},global loss is {client.globalloss[-1]:.4f},local loss is {client.localloss[-1]:.4f},aggragation error is {client.globalloss[-1] - client.localloss[-1]:.4f}")
            #         aggreError.append(client.globalloss[-1] - client.localloss[-1])
            #         client.error.append(client.globalloss[-1] - client.localloss[-1])
            # self.aggreErr.append([i, np.mean(aggreError), np.var(aggreError)])
            # print([i, np.mean(aggreError), np.var(aggreError)])
            # for client in self.clients:
            #     client.isselected = False
            # for client in self.selected_clients:
            #     client.train()
            #     client.isselected = True
            #     # client.selected = True
            #     client.calculate_local_loss()

            #---
            # aggreError=0
            #
            # for client in self.selected_clients:
            #     #i=0时，不是聚合得到的全局模型。不需要计算全局模型在本地的损失
            #     if i >0 :
            #         #上一轮训练得到的全局模型，还没有开始本地训练，但是已经传送全局模型过去了。
            #         client.calculate_gobal_loss(self.global_model)
            #         print(client.globalloss,client.localloss)
            #         print(f"{i},client {client.id},global loss is {client.globalloss[-1]:.4f},local loss is {client.localloss[-1]:.4f},aggragation error is {client.globalloss[-1]-client.localloss[-1]:.4f}")
            #         aggreError=+(client.globalloss[-1] - client.localloss[-1])
            #
            #     client.train()
            #     client.calculate_local_loss(self.global_model)
            #     print(i,client.id,client.localloss)
            #     #print(f"client {client.id},local loss is{client.localloss[-1]:.4f}")
            # #print(f"round {i}, aggregation error {aggreError:.4f}")
            # self.aggreErr.append(aggreError)
        #-------------------------------------------------------------------





            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # -------------------------------------
            # aggreError = []
            # for client in self.selected_clients:
            #     client.calculate_gobal_loss(self.global_model)
            #     client.calculate_local_loss()
            #     aggreError.append(client.globalloss[-1] )
            #     client.error.append(client.localloss[-1])
            # self.aggreErr.append([i, np.mean(aggreError), np.var(aggreError)])
            # --------------------------------------------

            self.Budget.append(time.time() - s_t)
            print('-'*25,"Round is ",i,'-'*25, 'time cost', '-'*25, self.Budget[-1])
            print('-'*25,"Budget is :",self.Budget)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        # # --------------7.训练过程中error
        # print("error",self.aggreErr)
        # redf = pd.DataFrame(columns=["group", "error", "var"])
        # redf.loc[len(redf) + 1] = ["group", "error", "var"]
        # for i in range(len(self.aggreErr)):
        #     redf.loc[len(redf) + 1] = self.aggreErr[i]
        # errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_errorg11.csv"
        # redf.to_csv(errorpath, mode='a', header=False)
        # print("success training write acc txt", errorpath)
        # # 记录一下每个客户端的变化情况
        # redf = pd.DataFrame(columns=["clientid", "distance","error"])
        # redf.loc[len(redf) + 1] = ["clientid", "distance", "error"]
        # for client in self.clients:
        #     redf.loc[len(redf) + 1] = [client.id,client.distance, client.error]
        # errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_clients_errorg11.csv"
        # redf.to_csv(errorpath, mode='a', header=False)
        # print("success training write acc txt", errorpath)
       # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id,colum_value)
       # ————————————————————————————————————————————————————————————————————————————————

        # 绘制混淆矩阵热力图---------------
        # self.plot_CM()
        # -----------------------------------------------------------------------------------------------
        # #写入训练时间---------------------------------------------------------------------------
        # redf = pd.DataFrame(
        #     columns=["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"])
        # redf.loc[len(redf) + 1] = ["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"]
        # redf.loc[len(redf) + 1] = [self.dataset, self.method, self.global_rounds, self.join_ratio,
        #                            sum(self.Budget[1:]) / len(self.Budget[1:]), sum(self.Budget[1:]), self.Budget[1:]]
        # accpath = self.programpath + "/res/time_cost_" + self.dataset + str(self.alpha) + "_time.csv"
        # print("success training write acc txt", accpath)
        # redf.to_csv(accpath, mode='a', header=False)






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
        print("train_slow,send_slow", self.num_clients, self.train_slow_clients, self.send_slow_clients)

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
            client.setlabel()
            client.sizerate = (client.train_samples + client.test_samples) / samples
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
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

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
    def send_models(self,ids):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)

        # add更新模型参数，将globalmodel复制给本地模型----------------------
        # for client in self.selected_clients:
        #     if self.ISAAW:
        #         client.local_initialization(self.global_model, round)
        # #------------------------------------

        for client in self.clients:
            if client.id in ids:
                start_time = time.time()
                client.set_parameters(self.global_model)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)



