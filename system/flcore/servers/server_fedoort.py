import time,os,queue
from flcore.clients.client_fedoort import clientFedOORT
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from selection.PramidFy import *
from utils.data_utils import read_client_data
import random,copy

class FedOORT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedOORT"
       #oort----------------
        self.prng_state=None
        self.blacklist_num =10
        self.cut_off =0.95
        self.util_history = []
        self.exploration_factor =0.9
        self.step_window = 2
        self.pacer_step = 50
        self.penalty = 2
        self.penalty_beta = 0
        self.desired_duration = 50
        self.client_utilities = {
            client_id: 0 for client_id in range(0, self.num_clients )
        }
        self.client_durations = {
            client_id: 0 for client_id in range(0, self.num_clients )
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(0, self.num_clients )
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(0, self.num_clients )
        }

        self.unexplored_clients = list(range(0, self.num_clients ))

        self.select_mode = "oort"
        # 设置客户段基本信息
        self.client_path = self.programpath + "/res/" + self.method + "/" + self.dataset + "client_profile.pkl"


        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedOORT)

        if not os.path.exists(self.client_path):
            print(f"self.client_path is {self.client_path}")
            self.set_global_client_profile(self.client_path)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def calc_client_util(self, client_id):
        # ("""Calculate the client utility.""",
        #  client_durations,desired_duration,current_round,client_last_rounds,client_utilities)
        #print("self.client_utilities,client_id",self.client_utilities,client_id)
        client_utility = self.client_utilities[client_id] + math.sqrt(
            0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
        )

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (
                                     self.desired_duration / self.client_durations[client_id]
                             ) ** self.penalty
            client_utility *= global_utility
        return client_utility

    def weights_aggregated(self):
        """after training updates info"""

        for client in self.selected_clients:
            self.client_utilities[client.id] = client.utility
            self.client_durations[client.id] = client.durations
            self.client_last_rounds[client.id] = self.current_round
            self.client_selected_times[client.id]=client.select_time
            # Calculate client utilities of explored clients
        for client in self.selected_clients:
            self.client_utilities[client.id] = self.calc_client_util( client.id)
            if self.client_selected_times[client.id] > self.blacklist_num :
                if client.id not in self.blacklist:
                    self.blacklist.append(client.id)

        # Adjust pacer
        utilities=[]
        for client in self.clients:
            utilities.append(client.utility)
        self.util_history.append(
            sum(utility for utility in utilities)
        )

        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window: -self.step_window]
            )
            current_pacer_rounds = sum(self.util_history[-self.step_window:])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step


    def select_clients(self):
        print("select mode  is oort ,num is :", self.current_num_join_clients)
        clientpool=[]

        for client in self.clients:
            clientpool.append(client.id)
        selected_clients=self.choose_clients(clientpool, self.current_num_join_clients)
        for client in self.clients:
            if client.id in selected_clients:
                self.selected_clients.append(client)
        return selected_clients



    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        if self.current_round > 1:
            # Exploitation
            exploited_clients_count = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in clients_pool
            ]

            # Calculate cut-off utility
            cut_off_util = (
                    self.client_utilities[sorted_by_utility[exploited_clients_count - 1]]
                    * self.cut_off
            )

            # Include clients with utilities higher than the cut-off
            exploited_clients = []
            for client_id in sorted_by_utility:
                if (
                        self.client_utilities[client_id] > cut_off_util
                        and client_id not in self.blacklist
                ):
                    exploited_clients.append(client_id)

            # Sample clients with their utilities根据客户端的效用值，计算客户端被选中的概率，并将概率归一化
            total_utility = float(
                sum(self.client_utilities[client_id] for client_id in exploited_clients)
            )

            probabilities = [
                self.client_utilities[client_id] / total_utility
                for client_id in exploited_clients
            ]
            print("exploited_clients_count",exploited_clients_count,"sorted_by_utility:",sorted_by_utility,"self.blacklist",self.blacklist)
            #从一组客户端中根据计算的概率选择若干客户端，且保证每个客户端被选择的概率与其效用成正比
            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(
                    exploited_clients,
                    min(len(exploited_clients), exploited_clients_count),
                    p=probabilities,
                    replace=False,
                )
                selected_clients = selected_clients.tolist()
            #用于记录当前选择的最后一个客户端在排序列表中的位置，后续用于决定是否需要进一步选择更多客户端来满足要求。
            last_index = (
                sorted_by_utility.index(exploited_clients[-1])
                if exploited_clients
                else 0
            )

            # If the result of exploitation wasn't enough to meet the required length
            #如果通过基于效用值的随机选择（exploitation）得到的客户端数量少于预期数量（exploited_clients_count），
            # 那么从剩余的客户端中补充，直到达到所需的客户端数量
            if len(selected_clients) < exploited_clients_count:
                for index in range(last_index + 1, len(sorted_by_utility)):
                    if (
                            not sorted_by_utility[index] in self.blacklist
                            and len(selected_clients) < exploited_clients_count
                    ):
                        selected_clients.append(sorted_by_utility[index])

        # Exploration
        #处理 未探索客户端（unexplored clients） 的随机选择，
        # 确保在客户端选择中包含那些尚未被探索的客户端，并且更新相应的状态信息
        # 将 Python 随机数生成器（PRNG，Pseudorandom Number Generator）的状态设置为之前保存的状态 (self.prng_state)
        if self.prng_state :
            random.setstate(self.prng_state)

        # Select unexplored clients randomly
        #从self.unexplored_clients列表中随机选择一部分客户端，直到选出的客户端数量满足需求。
        samplesize=min( clients_count - len(selected_clients),len(self.unexplored_clients))
        selected_unexplore_clients = random.sample(
            self.unexplored_clients, samplesize
        )

        #将当前的随机数生成器状态保存到 self.prng_state 中
        self.prng_state = random.getstate()
        #self.explored_clients 是一个记录已经被选择（探索过）的客户端的列表。通过这行代码，程序更新了已经被探索的客户端列表，避免这些客户端在未来的轮次中再次被选择
        self.explored_clients += selected_unexplore_clients
        #print("selected_unexplore_clients,self.explored_clients", selected_unexplore_clients,self.explored_clients)
        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplore_clients

        # for client in selected_clients:
        #     self.client_selected_times[client] += 1

      #  print("selected_clients",selected_clients)
        return selected_clients

    def train(self):
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
       # ————————————————————————————————————————————————————————————————————————————————

        for i in range(self.global_rounds+1):
            i += 1
            self.current_round=i
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
            #print(f"~~" * 20, f"local training is start,self.selected_clients is {len(self.selected_clients)} ID:{ids} ")
            for client in self.selected_clients:
                client.train()
                client.select_time+=1
            #更新训练的数据
            self.weights_aggregated()
            print("round is ",i,"self.blacklist is :",self.blacklist)
            # 计算选中的客户端与总体的差距-----------------------------

            kl_div, js_div, emd,round_time,highClientNum=self.record_update(i,ids)
          # for client in self.selected_clients:
            #     client.train(self.queue)
            #print("~~" * 20, "local training is end")

            # # 关闭线程池
            # threa.shutdown(wait=True)
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            #评估训练后的全局模型在每个本地的损失
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                res = self.evaluate_global(i,self.global_model)
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
        #print(f"self.selected_clients is {len(self.selected_clients)}")
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









