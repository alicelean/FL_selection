import time
from flcore.clients.client_centeralized import clientCenter
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from utils.data_utils import read_client_data
class Center(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "Center"
        self.fix_ids = True
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCenter)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):

        colum_value = []
        select_id = []

        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            file_path = self.programpath + "/res/selectids/" + str(
                self.num_clients) + "_" + str(self.join_ratio) + "/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            print("self path is :",file_path)
            self.select_idlist = self.read_selectInfo(file_path)

        for i in range(self.global_rounds+1):
            s_t = time.time()
            # ----2.设置不同的选择方式---------------------------------------------------
            if self.fix_ids:
                self.selected_clients = self.select_clients(i)
            else:
                self.selected_clients = self.select_clients()
                # ----------------------3.写入每次选择的client的数据----------------------------
                ids = []
                for client in self.selected_clients:
                    ids.append(client.id)
                select_id.append([i, ids])
                # -------------------------------------------------------------------------

            # -------------------------------------------------------------------------

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #self.evaluate()
                #res = self.evaluate(i)
                res = self.evaluate_center(i)
                # --------------------4.记录当前模型的状态，loss,accuracy----------------------
                resc = [self.filedir]
                for line in res:
                    resc.append(line)
                # resc.append(self.join_ratio)
                colum_value.append(resc)
                # print("envaluate:",i,colum_value)
        # -------------------------------------------------------------------------
            # 参与客户端训练本地数据
            for client in self.clients:
                client.selected = True
                client.train()
                #client.localtrain()
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



            # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # -------------------------------------
            aggreError = []
            for client in self.selected_clients:
                client.calculate_gobal_loss(self.global_model)
                client.calculate_local_loss()
                aggreError.append(client.globalloss[-1] )
                client.error.append(client.localloss[-1])
            self.aggreErr.append([i, np.mean(aggreError), np.var(aggreError)])
            # --------------------------------------------

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        # --------------7.训练过程中error
        print("error",self.aggreErr)
        redf = pd.DataFrame(columns=["group", "error", "var"])
        redf.loc[len(redf) + 1] = ["group", "error", "var"]
        for i in range(len(self.aggreErr)):
            redf.loc[len(redf) + 1] = self.aggreErr[i]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_errorg.csv"
        redf.to_csv(errorpath, mode='a', header=False)
        print("success training write acc txt", errorpath)
        # 记录一下每个客户端的变化情况
        redf = pd.DataFrame(columns=["clientid", "distance","error"])
        redf.loc[len(redf) + 1] = ["clientid", "distance", "error"]
        for client in self.clients:
            redf.loc[len(redf) + 1] = [client.id,client.distance, client.error]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_clients_errorg.csv"
        redf.to_csv(errorpath, mode='a', header=False)
        print("success training write acc txt", errorpath)


        # 6.写入idlist----保证整个客户端选择一致，更好的判别两种算法的差异---------------------------------------
        if not self.fix_ids:
            redf = pd.DataFrame(columns=["global_rounds", "id_list"])
            redf.loc[len(redf) + 1] = ["*********************", "*********************"]
            redf.loc[len(redf) + 1] = ["global_rounds", "id_list"]
            for v in range(len(select_id)):
                redf.loc[len(redf) + 1] = select_id[v]
            idpath = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            redf.to_csv(idpath, mode='a', header=False)
            print("write select id list ", idpath)
        # --------------7.训练过程中的全局模型的acc
        colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            redf.loc[len(redf) + 1] = colum_value[i]
        accpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_acc.csv"
        print("success training write acc txt", accpath)
        redf.to_csv(accpath, mode='a', header=False)
        allpath = self.programpath + "/res/ " + self.dataset + "_allacc_" +str(self.join_ratio)+"_"+str(self.num_clients)+"_"+ str(self.alpha) + ".csv"
        redf.to_csv(allpath, mode='a', header=False)
        print(colum_value)
        # ---------------记录数据-----------
        redf = pd.DataFrame(
            columns=["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"])
        redf.loc[len(redf) + 1] = ["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"]
        redf.loc[len(redf) + 1] = [self.dataset, self.method, self.global_rounds, self.join_ratio,
                                   sum(self.Budget[1:]) / len(self.Budget[1:]), sum(self.Budget[1:]), self.Budget[1:]]
        accpath = self.programpath + "/res/time_cost_" + self.dataset + str(self.alpha) + ".csv"
        print("success training write acc txt", accpath)
        redf.to_csv(accpath, mode='a', header=False)
        # 绘制混淆矩阵热力图---------------
        # self.plot_CM()
        # -----------------------------------------------------------------------------------------------






        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCenter)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        print("train_slow,send_slow", self.num_clients, self.train_slow_clients, self.send_slow_clients)
        alltrain=[]
        alltest=[]
        allsamples=0
        # 读取所有的训练数据和测试数据
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples += len(train_data) + len(test_data)
            for j  in  train_data:
                alltrain.append(j)
            for j in test_data:
                alltest.append(j)
            allsamples += samples
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients,
                                                self.send_slow_clients):

            client = clientObj(self.args,
                               id=i,
                               traindata=alltrain,
                               testsdata=alltest,
                               train_samples=len(alltrain),
                               test_samples=len(alltest),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.sizerate = (client.train_samples + client.test_samples) / samples
            print(client.id,client.sizerate)
        # 根据label 来计算distance
        self.setdistance()

        # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def send_models(self):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)