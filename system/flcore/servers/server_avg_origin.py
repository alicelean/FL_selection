import time,os
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedAvg"

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

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


            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #self.evaluate()
                res = self.evaluate(i)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc=self.addvalue(res)
                colum_value.append(resc)
                # ————————————————————————————————————————————————————————————————————————————————

            # 参与客户端训练本地数据（所有客户端参与训练）
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


        # --------------7.训练过程中error
        print("error",self.aggreErr)
        redf = pd.DataFrame(columns=["group", "error", "var"])
        redf.loc[len(redf) + 1] = ["group", "error", "var"]
        for i in range(len(self.aggreErr)):
            redf.loc[len(redf) + 1] = self.aggreErr[i]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_errorg11.csv"
        redf.to_csv(errorpath, mode='a', header=False)
        print("success training write acc txt", errorpath)
        # 记录一下每个客户端的变化情况
        redf = pd.DataFrame(columns=["clientid", "distance","error"])
        redf.loc[len(redf) + 1] = ["clientid", "distance", "error"]
        for client in self.clients:
            redf.loc[len(redf) + 1] = [client.id,client.distance, client.error]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_clients_errorg11.csv"
        redf.to_csv(errorpath, mode='a', header=False)
        print("success training write acc txt", errorpath)
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
