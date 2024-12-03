import time,os,torch,random
from flcore.clients.client_our import clientOUR
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np

class FedOUR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "Our"
        self.k=1000
        # if self.random_join_ratio:
        #     # 计算客户端选择数量
        #     self.current_num_join_clients = \
        #     np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        # else:
        #     self.current_num_join_clients = self.num_join_clients
        print(f"self.num_join_clients is {self.num_join_clients}")
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOUR)

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

    def train(self):
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        if self.fix_ids:
            self.read_fix_id()
       # ————————————————————————————————————————————————————————————————————————————————
        nextids = random.sample(range(20), 10)
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            # ids=self.get_selected_clients(i)
            ids=nextids
            print(f"round is{i},select id is{ids}")
            self.selected_clients=[]
            for client in self.clients:
                if client.id in ids:
                    self.selected_clients.append(client)
            select_id.append([i, ids])

            # 计算选中的客户端与总体的差距
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

            # ————————————————————————————————————————————————————————————————————————————————
            self.send_models()

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

            # 参与客户端训练本地数据（所有客户端参与训练）
            for client in self.clients:
                client.train()
            # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # # -------------------------------------
            nextids=self.update()
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

    def update(self):
        total_gradients=self.caculate_allgrad()
        data={}
        for client in self.clients:
            # # 确保模型权重和梯度具有兼容的形状
            # if model_weights.shape != gradients.shape:
            #     raise ValueError("Model weights and gradients must have the same shape.")
            #
            # # 使用 torch.dot 进行点乘运算
            # result = torch.dot(model_weights.view(-1), gradients.view(-1))

            client.update_parameter(total_gradients,0.001)
            data[client.id]=client.pit
        values = torch.tensor(list(data.values()))
        # for key, value in data.items():
        #     print(f"_data {key}: {value:.4f}")
        min_val = values.min()
        max_val = values.max()
        # 2. 将归一化后的值与键配对
        normalized_data = {k: (v - min_val) / (max_val - min_val) for k, v in data.items()}
        # for key, value in normalized_data.items():
        #     print(f"normalized_data {key}: {value:.4f}")
        # 3. 排序
        sorted_items = sorted(normalized_data.items(), key=lambda item: item[1], reverse=True)
        # 4. 提取排序后的键值对
        sorted_dict = {k: v for k, v in sorted_items}
        #print("Sorted and normalized dictionary:")
        select_id=[]
        for key, value in sorted_dict.items():
            #print(f"{key}: {value:.4f}")
            select_id.append(key)
            if len(select_id)==self.current_num_join_clients:
                break
        return select_id


    def caculate_allgrad(self):
        for client in self.clients:
            client.calculate_gradients(self.global_model)
        params_g = list(self.global_model.parameters())
        total_gradients= [(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]
        for client in self.clients:
            for upgrad,cur in zip(total_gradients, client.currGrad):
                upgrad.data = upgrad + cur

        return total_gradients






