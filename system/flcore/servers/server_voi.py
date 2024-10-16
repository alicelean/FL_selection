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


class FedVOI(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedAvg"

        self.selection = PramidFy(args,self.num_join_clients)
        self.queue = queue.Queue()
        self.InfoQueue= queue.Queue()


        #------------voi
        self.timeResoure=1000
        self.StatesQueue = queue.Queue()
        self.actionQueue= queue.Queue()
        self.hyperparameters = {
            'timesteps_per_batch': 100,
            'max_timesteps_per_episode': 20,
            'gamma': 0.99,
            'n_updates_per_iteration': 10,
            'lr': 3e-4,
            'clip': 0.2,
            'render': True,
            'render_every_i': 10
        }
        self.popReward = 0
        self.mode = 'train'
        print(f"self.num_join_clients is {self.num_join_clients},arg.nu{args.num_clients}")
        self.env = FederatedLearningEnv(num_clients=args.num_clients, k=self.num_join_clients)
        self.actor_model = args.actor_model
        self.critic_model = args.critic_model
        self.total_timesteps = 20000
        # 初始化PPO------------
        # self.StatesQueue,self.actionQueue
        self.ppomodel = PPO(policy_class=FeedForwardNN, env=self.env, **self.hyperparameters)
        if self.actor_model != '' and critic_model != '':
            print("INFO:----------", "load_state_dict actor and critic-------------")
            self.ppomodel.actor.load_state_dict(torch.load(self.actor_model))
            self.ppomodel.critic.load_state_dict(torch.load(self.critic_model))


#-------------------------------------------
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientVOI)
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

    def update_client_states(self):
        self.client_states=[]
        for client in self.clients:
            client.states = [client.loss, client.size, client.stale, client.age]
            self.client_states.append(client.states)

    def caculate_reward(self,train_loss):
        self.popReward=100000-train_loss

    def get_selected_clients(self, action):
        #先归一化成概率再选最高的topk
        Actionprobabilities = torch.softmax(action, dim=1)
        top_k_probs, top_k_indices = torch.topk(Actionprobabilities, self.env.select_num)
        for client in self.clients:
            if client.id in top_k_indices:
                selected_clients.append(client)

        return top_k_indices
    def get_client_states(self):
        clientStates=[]
        for c in self.clients:
            clientStates.append(c.states)


    def  rl_train(self):
        #------------------------------
        # 新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        if self.fix_ids:
            self.read_fix_id()
        # ————————————————————————————————————————————————————————————————————————————————

        #actor,critic模型训练-------------
        print(f"Learning... Running {self.ppomodel.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.ppomodel.timesteps_per_batch} timesteps per batch for a total of {self.total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < self.total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            # 进行一次轨迹收集
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.ppomodel.rollout()  # ALG STEP 3
            print("#" * 30, "rl  data start -------")
            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []
            t = 0
            round=0
            costtime=0
            truncated=False
            terminated=False
            while t < self.ppomodel.timesteps_per_batch:
                start_time = time.time()
                ep_rews = []# rewards collected per episode,一条轨迹

                obs = self.get_client_states()
                print("#" * 30, f" t is {t} ;obs is {obs}")
                done = False
                for ep_t in range(self.ppomodel.max_timesteps_per_episode):
                    t += 1  # Increment timesteps ran this batch so far
                    batch_obs.append(obs)
                    print("-" * 30,
                          f"ep_t is {ep_t},t is {t},self.max_timesteps_per_episode is {self.ppomodel.max_timesteps_per_episode},self.timesteps_per_batch is {self.ppomodel.timesteps_per_batch}")
                    action, log_prob = self.get_action(obs, True)
                    print("Actor output action:", action.shape, "log_prob", log_prob.shape, )
                    #----------------------fl training------------------------------
                    round+=1
                    # 新增2————————————————————————————————————————————————————————————————————————————————

                    ids = self.get_selected_clients(action)
                    select_id.append([round, ids])
                    # ————————————————————————————————————————————————————————————————————————————————
                   #send global model to all clients
                    self.send_models(ids)
                    # selected clients local training
                    local_training_list = []
                    t = ThreadPoolExecutor(max_workers=self.num_clients)
                    j = 0
                    # 提交所有客户端的训练任务
                    for client in self.clients:
                        future = t.submit(client.train, self.queue)  # 提交任务
                        local_training_list.append(future)  # 将 Future 对象添加到列表
                    # 使用 as_completed 来异步处理任务结果
                    for future in as_completed(local_training_list):
                        try:
                            result = future.result()  # 获取任务结果，阻塞直到任务完成
                            print(f"Task completed with result: {result}")
                        except Exception as e:
                            print(f"Task generated an exception: {e}")
                    # 关闭线程池
                    t.shutdown(wait=True)
                    self.receive_models()
                    if self.dlg_eval and i % self.dlg_gap == 0:
                        self.call_dlg(i)
                    self.aggregate_parameters()
                    if i % self.eval_gap == 0:
                        print(f"\n-----------rl--Round number: {i}-------------")
                        print("\nEvaluate global model")
                        # localResult=[self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
                        # 更新需要计算本地的状态:loss
                        _ = self.evaluate(i)
                        # print("res is:",res)
                        # 新增3————————————————————————————————————————————————————————————————————————————————
                        # 记录当前模型的状态，loss,accuracy
                        res = self.evaluate_global(i, self.global_model)
                        resc = self.addvalue(res)
                        colum_value.append(resc)
                        # 根据损失计算一下回报
                        self.caculate_reward(res[2])
                        # ————————————————————————————————————————————————————————————————————————————————
                        # 更新强化学习环境状态
                        self.update_client_states()


                    # ----------------------------------------------------------------
                    obs=self.get_client_states()
                    rew=self.popReward
                    costtime += time.time() - start_time
                    if round>=self.global_rounds or self.timeResoure<costtime:
                        terminated=True

                    # obs, rew, terminated, truncated, _ = self.env.step(action)
                    print("env output obs:", obs.shape, "rew", rew, )
                    # 或运算
                    done = terminated | truncated

                    # Track recent reward, action, and action log probability
                    ep_rews.append(rew)
                    batch_acts.append(action)
                    # print("log_prob,batch_log_probs",log_prob.shape,len(batch_log_probs))
                    batch_log_probs.append(log_prob)

                    # If the environment tells us the episode is terminated, break
                    if done:
                        break
                # Track episodic lengths and rewards
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
            # Reshape data as tensors in the shape specified in function description, before returning
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            batch_log_probs = torch.stack(batch_log_probs, dim=0)  # 将多个张量沿新的维度堆叠

            batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4
            # Log the episodic returns and episodic lengths in this batch.
            print("#" * 30, "rollouting end -------")
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for _ in range(self.ppomodel.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()
            # 新增4————————————————————————————————————————————————————————————————————————————————
            self.write_info(select_id, colum_value)
            # ————————————————————————————————————————————————————————————————————————————————

            print("rl,training is over!")




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
            self.send_models(ids,i)
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
                tmp_r_list.append(t.submit(client.train, self.queue))
                tmp_r_list[j].result()
                j+=1



            self.sampledClientSet=self.selection.run(self.global_model, self.queue, self.stop_signal, self.clientSampler,self.sampledClientSet,i)
            print("###"*20,f"  round is {i}self.selection.run {self.sampledClientSet}")
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            #--------------------------------------------------------------
            
            
            #评估全局模型，获得总体损失
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #localResult=[self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
                #更新需要计算本地的状态:loss
                _ = self.evaluate(i)
                #print("res is:",res)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                res = self.evaluate_global(i, self.global_model)
                resc=self.addvalue(res)
                colum_value.append(resc)
                # 根据损失计算一下回报
                self.caculate_reward(res[2])
                # ————————————————————————————————————————————————————————————————————————————————
                # 更新强化学习环境状态
                self.update_client_states()
            
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
    def send_models(self,selectids):
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
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            #选中的客户端要更新他们的全局客户端的
            if client.id in selectids:
                set_parameters_global(self, model, round)







