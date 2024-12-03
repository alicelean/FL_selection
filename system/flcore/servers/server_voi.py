import time,os,queue
from flcore.clients.client_voi import clientVOI
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from selection.PramidFy import *
from utils.data_utils import read_client_data
import random,copy
import gymnasium as gym
import torch
import sys
sys.path.append('/Users/alice/Desktop/python/FL_selection/system/flcore/servers')
from ppo import PPO
from network import FeedForwardNN
from env import FederatedLearningEnv
from concurrent.futures import as_completed
from torch import nn


class FedVOI(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedVOI"
        self.omiga=5
        self.tr=0
        self.stop=False

        # self.queue = queue.Queue()
        # self.InfoQueue= queue.Queue()
        #------------voi
        #时间资源
        #用来训练rl的轮次
        self.updateround=100
        if self.updateround>self.global_rounds:
            self.updateround=self.global_rounds



        self.StatesQueue = queue.Queue()
        self.actionQueue= queue.Queue()
        self.hyperparameters = {
            'timesteps_per_batch': 20,
            'max_timesteps_per_episode': 5,
            'gamma': 0.98,
            'n_updates_per_iteration': 20, #在每次采集到新数据后，进行多少次策略更新
            'lr': 3e-4,
            'clip': 0.2,
            'render': True,

            'render_every_i': 10
        }
        self.popReward = 0
        self.mode = 'train'
        print(f"self.num_join_clients is {self.num_join_clients},client is:{args.num_clients}")
        self.env = FederatedLearningEnv(num_clients=args.num_clients, k=self.num_join_clients)
        self.actor_model = args.actor_model
        self.critic_model = args.critic_model
        self.total_timesteps = self.updateround
        # 初始化PPO------------
        # self.StatesQueue,self.actionQueue
        self.ppomodel = PPO(policy_class=FeedForwardNN, env=self.env, **self.hyperparameters)
        if self.actor_model != '' and self.critic_model != '':
            print("INFO:----------", "load_state_dict actor and critic-------------")
            self.ppomodel.actor.load_state_dict(torch.load(self.actor_model))
            self.ppomodel.critic.load_state_dict(torch.load(self.critic_model))

        #资源设计
        # self.computeResource()
        # self.communicateResource()


#-------------------------------------------
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientVOI)
        #在初始化客户端以后需要在注册器中注册每个客户端的信息
        self.sampledClientSet=set()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def update_client_states(self):
        self.client_states=[]
        for client in self.clients:
            print("client id state  :",client.currentloss, client.size, client.stale, client.age)
            client.states = [client.currentloss, client.size, client.stale, client.age]
            self.client_states.append(client.states)

    def caculate_reward(self,train_loss):
        self.popReward=self.omiga-train_loss

    def get_selected_clients(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)  # 将 numpy.ndarray 转换为 PyTorch Tensor


        #先归一化成概率再选最高的topk
        Actionprobabilities = torch.softmax(action, dim=1)
        top_k_probs, top_k_indices = torch.topk(Actionprobabilities, self.env.select_num)
        ids=top_k_indices.tolist()[0]
        for client in self.clients:
            if client.id in ids:
                self.selected_clients.append(client)
        #print("top_k_indices is ",top_k_indices)
        return ids
    def get_client_states(self):
        clientStates=[]
        for c in self.clients:
            c.states=[c.currentloss,c.size,c.stale,c.age]
            clientStates.append(c.states)


        return torch.tensor(clientStates)



    import numpy as np
    def computeResource(self,mean_training_time=100,std_dev_training_time=50):
        # 从正态分布中生成训练时间
        training_times = np.random.normal(mean_training_time, std_dev_training_time, self.num_clients)
        # 确保训练时间为正数，并裁剪到合理范围
        training_times = np.clip(training_times, 1, None)

        # 输出每个客户端的训练时间
        timedict={}
        for client_id, time in enumerate(training_times):
            print(f"客户端 {client_id}: 训练时间 {time:.2f} 秒")
            timedict[client_id]=time
        for client in self.clients:
            client.compute=timedict[client.id]

    def communicateResource(self,mean_training_time=50,std_dev_training_time=100):
        # 从正态分布中生成训练时间
        training_times = np.random.normal(mean_training_time, std_dev_training_time, self.num_clients)
        # 确保训练时间为正数，并裁剪到合理范围
        training_times = np.clip(training_times, 1, None)
        # 输出每个客户端的训练时间
        timedict={}
        for client_id, time in enumerate(training_times):
            print(f"客户端 {client_id}: 通信时间 {time:.2f} 秒")
            timedict[client_id]=time
        for client in self.clients:
            client.communicate=timedict[client.id]

    def  rl_train(self,round):
        #------------------------------
        # 新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        # ————————————————————————————————————————————————————————————————————————————————
        print("~"*50,"rl,training is start!","~"*50)

        #actor,critic模型训练-------------
        #print(f"{self.ppomodel.timesteps_per_batch} timesteps per batch for a total of {self.total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far,total_timesteps类似于T
        i_so_far = 0  # Iterations ran so far
        #total_timesteps是总体训练时间
        up=0
       # t_so_far < self.updateround and self.cost_time < self.total_time and
        while  not self.stop:  # ALG STEP 2
            #print(f"1.round {round} t_so_far {t_so_far} self.total_timesteps {self.total_timesteps} round+self.ppomodel.timesteps_per_batch{round+self.ppomodel.timesteps_per_batch} self.updateround{self.updateround}")
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            # 进行一次轨迹收集
            #batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.ppomodel.rollout()  # ALG STEP 3
            up+=1
            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []
            r_t = 0
            costtime=0
            truncated=False
            terminated=False
            start_timelist=[]
            done=False
            #初次随机选择客户端：
            lastaction=None
            b_t=0
            #多条轨迹采样后一次更新，采集不够一个就不采集数据了
            #while r_t < self.ppomodel.timesteps_per_batch and round + self.ppomodel.timesteps_per_batch <= self.updateround:

           #采集数据-------------------------------------------------

            while b_t < self.ppomodel.timesteps_per_batch and not self.stop:
                start_time = time.time()
                ep_rews = []  # rewards collected per episode,一条轨迹

                for ep_t in range(self.ppomodel.max_timesteps_per_episode):
                    if self.stop:
                        break
                    print("~" * 50, f"round is {round}", "~" * 50)
                    print(f" ep_t {ep_t},t_so_far {t_so_far} self.cost_time is {self.cost_time},self.total_time  is {self.total_time},updateround is {self.updateround}")
                    if   ep_t >= self.updateround or b_t>=self.updateround  or t_so_far >=self.updateround or self.cost_time>=self.total_time:
                        self.stop=True


                    s_t = time.time()
                    b_t+=1 # Increment timesteps ran this batch so far
                    round+=1
                    # server 初始发送全局模型----------
                    #print("#" * 50,"1.global model send to all client")
                    self.tr = time.time()
                    self.send_models(round)
                    # -----------local training--------------------------
                    #print("#" * 50, "2.all client start asynchronous local training")
                    for client in self.selected_clients:
                        client.train(self.tr)
                        #print("#" * 50, f"client {client.id}")

                    #print("#" * 50, "2.all client start asynchronous local training")
                    #计算消耗的资源--------------
                    #self.updates_costedResoure()

                    # 输入状态获得动作------rl data collected----------------------
                    rl_time = time.time()
                    obs = self.get_client_states()
                    batch_obs.append(obs)
                    #killprint("obs",obs)
                    if isinstance(obs, list):
                        obs = torch.tensor(obs, dtype=torch.float)
                    action, log_prob = self.ppomodel.get_action(obs, True)
                    self.cost_time += time.time() - rl_time
                    #----------------------fl training------------------------------
                    # 新增2————————————————————————————————————————————————————————————————————————————————
                    ids = self.get_selected_clients(action)
                    if lastaction is not None:
                        if np.array_equal(ids, lastaction):
                            print("ERROR : action is equal to last action ",)
                    #print(f"action is {action}")
                    lastaction = ids
                    select_id.append([round, ids])

                    # 计算选中的客户端与总体的差距
                    kl_div, js_div, emd = self.get_select_distance()
                    # 新增：统计每一轮训练的资源消耗------
                    highClientNum = 0
                    round_time = []
                    for client in self.clients:
                        if client.id in ids:
                            client.select_time += 1
                            client.last_select = round
                            if client.isRichResource:
                                highClientNum += 1
                            round_time.append(client.costedResoure)
                    self.cost_time += max(round_time)
                    #记录选中的客户端与总体的差距------------
                    #distance=self.get_select_distance()
                    print(f"send select message to client ,select client id  is :", ids)
                    #print("2len(batch_obs) is", len(batch_obs))
                    #需要将选中的信息发送给客户端---------------------
                    for client in self.clients:
                        if client.id in ids:
                            client.isSelected=True
                            client.age = 0
                        else:
                            client.isSelected = False
                            client.age +=1
                    #----------------------------------------------
                    self.receive_models()
                    self.aggregate_parameters()


                    #print("~"*30,"global model aggragation and  send  to client ")

                    if round % self.eval_gap == 0:
                        #print("-"*50,f"------ep_t is {ep_t}, t is {r_t} -Round : {round}，Evaluate global mode-------------","-"*50)
                        # localResult=[self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
                        # # 更新需要计算本地的状态:loss
                        # _ = self.evaluate(round)
                        # print("res is:",res)
                        # 新增3————————————————————————————————————————————————————————————————————————————————
                        # 记录当前模型的状态，loss,accuracy
                        res = self.evaluate_global(round, self.global_model)
                        resc = self.addvalue(res)
                        resc.append(kl_div)
                        resc.append(js_div)
                        resc.append(emd)
                        resc.append(max(round_time))
                        resc.append(highClientNum)
                        colum_value.append(resc)
                        #print("-" * 30, "---------------------", "-" * 30)
                        # 根据损失计算一下回报
                        self.caculate_reward(res[2])
                        #print("self.popReward is :", self.popReward)
                        # ————————————————————————————————————————————————————————————————————————————————
                    print("add-----------------")


                    rew=self.popReward
                    ep_rews.append(rew)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)
                    self.Budget.append(time.time() - s_t)




                #print(f"the {r_t}tragtory collected---------")

                # Track episodic lengths and rewards
                print(f"ep_rews is {ep_rews}")
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
                start_timelist.append(time.time()-start_time)




            print("start  to update network ")
            print(f"Length of batch_obs: {len(batch_obs)}")
            if len(batch_obs) == 0:
                raise ValueError("ERROR:    batch_obs is empty, no observations collected.")

            update_time = time.time()

            #更新网络-----------------------------

            batch_obs = torch.stack(batch_obs)
            batch_acts = [torch.tensor(act) if isinstance(act, np.ndarray) else act for act in batch_acts]
            batch_acts = torch.stack(batch_acts)
            batch_log_probs = torch.stack(batch_log_probs, dim=0)  # 将多个张量沿新的维度堆叠
            #print("batch_log_probs.shape,batch_acts.shape,batch_obs.shape is ,", batch_log_probs.shape,batch_acts.shape,batch_obs.shape)
            batch_rtgs = self.ppomodel.compute_rtgs(batch_rews)  # ALG STEP 4
            # Log the episodic returns and episodic lengths in this batch.
            self.ppomodel.logger['batch_rews'] = batch_rews
            self.ppomodel.logger['batch_lens'] = batch_lens
            # Log the episodic returns and episodic lengths in this batch.
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            # Increment the number of iterations
            i_so_far += 1
            #print("#" * 30, f"rl data collection success-t_so_far is {t_so_far},i_so_far is {i_so_far}-,update network-----")
            # Logging timesteps so far and iterations so far
            self.ppomodel.logger['t_so_far'] = t_so_far
            self.ppomodel.logger['i_so_far'] = i_so_far
            # Calculate advantage at k-th iteration
            V, _ = self.ppomodel.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            #在每次采集到新数据后，进行多少次策略更新
            for _ in range(self.ppomodel.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.ppomodel.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.ppomodel.clip, 1 + self.ppomodel.clip) * A_k
                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.ppomodel.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.ppomodel.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.ppomodel.critic_optim.zero_grad()
                critic_loss.backward()
                self.ppomodel.critic_optim.step()

                # Log actor loss
                self.ppomodel.logger['actor_losses'].append(actor_loss.detach())



            #更新完成-------------------------------------------------




            update_time=time.time()-update_time
            self.cost_time += update_time
            print("update onece time is :",update_time)
            # Print a summary of our training so far
            self.ppomodel._log_summary()
            if self.ppomodel.max_timesteps_per_episode + round > self.updateround:
                self.stop = True


        # 新增4————————————————————————————————————————————————————————————————————————————————
        if round==self.global_rounds:
            self.write_info(select_id, colum_value)
        # ————————————————————————————————————————————————————————————————————————————————
        print("~"*50,"rl,training is over!","~"*50)


        return colum_value,select_id,round

    def train(self):
        round=0
       #新增1————————————————————————————————————————————————————————————————————————————————
        colum_value,select_id,round=self.rl_train(round)
        print(f"train current round is {round} ")
       # ————————————————————————————————————————————————————————————————————————————————

        for i in range(round+1,self.global_rounds+1):
            #资源耗尽
            if self.cost_time >self.total_time or round == self.global_rounds:
                if self.cost_time > self.total_time:
                    print(f"TIME OUT : self.cost_time {self.cost_time},self.total_time {self.total_time}")
                if round == self.global_rounds:
                    print(f"ROUND ARRIVED : round {round} self.global_rounds {self.global_rounds}")
                break
            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            #策略网络选择客户端--select id------------------
            obs = self.get_client_states()
            if isinstance(obs, list):
                obs = torch.tensor(obs, dtype=torch.float)
            action, log_prob = self.ppomodel.get_action(obs, True)
            ids = self.get_selected_clients(action)
            print(" local training select id is :",ids)
            # 计算选中的客户端与总体的差距
            #distance = self.get_select_distance()
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————
            self.tr = time.time()
            self.send_models(i)
         # ————————————————————————————————————————————————————————————————————————————————
            #本地训练
            for client in self.selected_clients:
                client.train(self.tr)

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



            self.receive_models()

            # if self.dlg_eval and i % self.dlg_gap == 0:
            #     self.call_dlg(i)

            self.aggregate_parameters()
            print("~" * 30, "global model aggragation and  send  to client ")

            #--------------------------------------------------------------

            #评估全局模型，获得总体损失
            if i%self.eval_gap == 0:
                print(f"\n-------------Training  Round number: {i}-------------")
                print("\nEvaluate global model")
                # 记录当前模型的状态，loss,accuracy
                res = self.evaluate_global(i, self.global_model)
                resc=self.addvalue(res)
                resc.append(kl_div)
                resc.append(js_div)
                resc.append(emd)
                resc.append(max(round_time))
                resc.append(highClientNum)
                colum_value.append(resc)
                print("res is:", resc)
            #----------------------------------------------------------------
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

       # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id,colum_value)
       # ————————————————————————————————————————————————————————————————————————————————
        self.save_results()



    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        # 初始化资源信息
        clientinf_path = self.programpath + "dataset/" + self.alpha + "/" + self.dataset + "/clientInfo.json"
        self.clientsResource = self.read_clients(clientinf_path)
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
            # distanceVec = [len(train_data)+len(test_data)]
            # sizeVec = [len(train_data)+len(test_data)]
            # tmp_dict[i]=[]
            # tmp_dict[i].append(distanceVec)
            # tmp_dict[i].append(sizeVec)
            # self.InfoQueue.put(tmp_dict)


        for client in self.clients:
            label = client.setlabel()
            for j in range(len(label)):
                self.alllabel[j] += label[j]
            client.size = client.train_samples + client.test_samples
            client.sizerate = (client.size*1.0) / samples
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
        assert (len(self.selected_clients) > 0)
        #print("receive_models start-------------")
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
            #print("aggregation select id is:",client.id)
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                #从客户端缓存中取模型
                self.uploaded_models.append(copy.deepcopy(client.localmodel))
            else:
                print("Error--------client receive model-",client.id)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    def send_models(self,round):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)
        for client in self.clients:
            #self.model
            #client.set_parameters(self.global_model)
            #global缓存
            client.set_parameters_global(self.global_model, round)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] +=client.communicate


    def evaluate(self, group, acc=None, loss=None):
        #利用本地模型进行评估
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

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        # print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # print("Averaged Test AUC: {:.4f}".format(test_auc))
        # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]







