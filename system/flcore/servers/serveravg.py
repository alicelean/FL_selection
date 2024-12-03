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
        self.select_mode="random"
        self.Isgenerate_ClientInfo=True
        #self.select_mode="datasize"
        self.current_num_join_clients=int(self.join_ratio*self.num_clients)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.setHighResourceClient()



        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    def send_models(self):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)

        # add更新模型参数，将globalmodel复制给本地模型----------------------
        # for client in self.selected_clients:
        #     if self.ISAAW:
        #         client.local_initialization(self.global_model, round)
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

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        active_clients=self.selected_clients
        print("select client num is :",len(active_clients),"self.samples is :",self.samples)
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples=0
        for client in active_clients:
            total_samples+=client.size
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.size)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
            print("weight is :",self.uploaded_weights[i])

    def train(self):
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        # if self.fix_ids:
        #     print("read fix_ids...........")
        #     self.read_fix_id()
       # ————————————————————————————————————————————————————————————————————————————————
        self.cost_time=0
        lastid=None
        for i in range(self.global_rounds+1):
            round_time=[]
            highClientNum=0
            if self.total_time<self.cost_time:
                print(f"time is out,self.total_time is {self.total_time},self.cost_time is {self.cost_time}")
                break
            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            ids=self.get_selected_clients(i)
            if lastid!=None:
                lastlabel=self.getlabel(lastid)
                currentlabel=self.getlabel(lastid)



            lastid=ids
            select_id.append([i, ids])
            #新增：统计每一轮训练的资源消耗------
            for client in self.clients:
                if client.id in ids:
                    client.select_time +=1
                    client.last_select = i
                    if client.isRichResource:
                        highClientNum+=1
                    round_time.append(client.costedResoure)
            self.cost_time+=max(round_time)
            print("self.cost_time is :",self.cost_time,"ids is ",ids)
            # 计算选中的客户端与总体的差距
            kl_div, js_div, emd  = self.get_select_distance()
            # ————————————————————————————————————————————————————————————————————————————————
            self.send_models()



            # 参与客户端训练本地数据（所有客户端参与训练）
            for client in self.selected_clients:
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

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #self.evaluate()
                #res = self.evaluate(i)
                res = self.evaluate_global(i, self.global_model)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc=self.addvalue(res)
                resc.append(kl_div)
                resc.append(js_div)
                resc.append(emd)
                resc.append(max(round_time))
                resc.append(highClientNum)
                colum_value.append(resc)
                # ————————————————————————————————————————————————————————————————————————————————


            # # -------------------------------------
            # aggreError = []
            # for client in self.selected_clients:
            #     client.calculate_gobal_loss(self.global_model)
            #     client.calculate_local_loss()
            #     aggreError.append(client.globalloss[-1] )
            #     client.error.append(client.localloss[-1])
            # self.aggreErr.append([i, np.mean(aggreError), np.var(aggreError)])
            # # --------------------------------------------

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








        self.save_results()
        #self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
