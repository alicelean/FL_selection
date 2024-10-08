import time,os
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.clients.clientala_aaw import clientALA_AAW
from utils.data_utils import read_client_data
from utils.distance import jensen_shannon_distance
import pandas as pd
import numpy as np
import math,torch
import random
class FedALA_AAW(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        # select slow clients
        self.set_slow_clients()
        self.method = "FedALA_AAW"
        self.set_clients(clientALA_AAW)
        self.fix_ids = True
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("------------------Finished creating server and clients.----------------------")

        # self.load_model()
        self.Budget = []



    def train(self):
        print("*************************** server_FedALA_AAW train ***************************")
        self.writeparameters()
        colum_value = []
        select_id = []


        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            file_path = self.programpath + "/res/selectids/" + str(
                self.num_clients) + "_" + str(self.join_ratio) + "/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            print(f"______________reading fix id from {file_path}-----------------------")
            self.select_idlist = self.read_selectInfo(file_path)


        for i in range(self.global_rounds+1):
            s_t = time.time()
            print("*************************** 2.server_FedALA_AAW select_clients ***************************\n")

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

            print(f"**************************round is {i}* 3.server_aaw init_weight_vector ***************************\n")
            #每一轮要选择的客户端进行权重归一化(Di/Ds)
            self.select_weight_vector()

            print("*************************** 4.server_FedALA_AAW send_models,client.local_initialization ***************************\n")
            #发送模型信息
            self.send_models(i)

            #评估模型
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                #[self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]
                res=self.evaluate(i)
                #记录当前模型的状态，loss,accuracy等
                if res[3]>=0.99:
                    for client in self.selected_clients:
                        client.AAW.stop=True
                resc = [self.filedir]
                for line in res:
                    resc.append(line)
                colum_value.append(resc)
            print("*************************** 5.server_FedALA_AAW selected_clients ,train（）***************************\n")

            #参与客户端训练本地数据
            # for client in self.selected_clients:
            #     client.selected=True
            #     client.train()
                #client.localtrain()

            # -------------------------------------------------------------------------
            aggreError = 0

            # for client in self.clients:
            #     if i != 0 and client.selected:
            #         #曾经参与过的客户端计算本地模型和全局模型的差异，因为训练过所以是有本地模型的，全局模型就是上一次聚合后的模型。
            #         client.calculate_loss(self.global_model,False)
            #         aggreError = +(client.globalloss[-1] - client.localloss[-1])
            #     for sclient in self.selected_clients:
            #         client.selected=True
            #         #本次被选择的客户端，需要训练
            #         if client.id==sclient.id:
            #             client.train()
            #             break
            #     #每个客户端都需要计算本地模型，如果没有训练过，也就是本次没有参加，则计算之前的本地模型对应的损失。
            #     client.calculate_local_loss(True)
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
            # aggreError = []
            # for client in self.selected_clients:
            #     # i=0时，不是聚合得到的全局模型。不需要计算全局模型在本地的损失
            #     if i != 0:
            #         client.calculate_gobal_loss(self.global_model)
            #         #client.calculate_global_loss_aaw(self.global_model)
            #         # print(f"client {client.id},global loss is {client.globalloss[-1]:.4f},local loss is {client.localloss[-1]:.4f},aggragation error is {client.globalloss[-1]-client.localloss[-1]:.4f}")
            #         aggreError = +(client.globalloss[-1])
            #     client.train()
            #     #client.calculate_local_loss(self.global_model)
            # print(f"round {i}, aggregation error {aggreError:.4f}")
            # self.aggreErr.append([i,aggreError])

            # 计算上一轮聚合后模型在本地数据上的表现
            aggreError = []
            for client in self.clients:
                # i=0时，不是聚合得到的全局模型。不需要计算全局模型在本地的损失
                if i > 0 and client.isselected:
                    client.isselected = False
                    # 上一轮训练得到的全局模型，还没有开始本地训练，但是已经传送全局模型过去了。
                    client.calculate_gobal_loss(self.global_model)
                    # print(client.globalloss, client.localloss)
                    # print(f"{i},client {client.id},global loss is {client.globalloss[-1]:.4f},local loss is {client.localloss[-1]:.4f},aggragation error is {client.globalloss[-1] - client.localloss[-1]:.4f}")
                    aggreError.append(client.globalloss[-1])
                    client.error.append(client.globalloss[-1])
            self.aggreErr.append([i, np.mean(aggreError), np.var(aggreError)])
            # 本轮的模型训练
            for client in self.selected_clients:
                client.train()
                client.isselected = True
            # -------------------------------------------------------------------

            print(
                "*************************** 6.server_FedALA_AAW adpativeweight ,client.local_initialization***************************\n")
            #自适应更新权重
            self.adpativeweight()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            print(
                "*************************** 7.server_FedALA_AAW receive_models ***************************\n")

            print("\n FedALA_AAW receive_models")
            #将发送模型参数
            self.receive_models(i)

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            #生成新的聚合模型
            self.aggregate_parameters()

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
        redf = pd.DataFrame(columns=["group", "error", "var"])
        redf.loc[len(redf) + 1] = ["group", "error", "var"]
        for i in range(len(self.aggreErr)):
            redf.loc[len(redf) + 1] = self.aggreErr[i]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_errorg1.csv"
        redf.to_csv(errorpath, mode='a', header=False)
        print("success training write acc txt", errorpath)
        #记录一下每个客户端的变化情况
        redf = pd.DataFrame(columns=["clientid", "error"])
        redf.loc[len(redf) + 1] = ["clientid", "error"]
        for client in self.clients:
            for i in range(len(self.aggreErr)):
                redf.loc[len(redf) + 1] = [client.id,client.error]
        errorpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_clients_errorg1.csv"
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
        allpath = self.programpath + "/res/ " + self.dataset + "_allacc_" + str(self.alpha) + ".csv"
        allpath = self.programpath + "/res/ " + self.dataset + "_allacc_" + str(self.join_ratio) + "_" + str(
            self.num_clients) + "_" + str(self.alpha) + ".csv"

        redf.to_csv(allpath, mode='a', header=False)
        print(colum_value)
        # ---------------记录数据-----------
        redf = pd.DataFrame(
            columns=["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"])
        redf.loc[len(redf) + 1] = ["dataset", "method", "round", "ratio", "average_time_per", "total_time", "time_list"]
        redf.loc[len(redf) + 1] = [self.dataset, self.method, self.global_rounds, self.join_ratio,
                                   sum(self.Budget[1:]) / len(self.Budget[1:]), sum(self.Budget[1:]), self.Budget[1:]]
        accpath = self.programpath + "/res/time_cost_" + self.alpha + ".csv"
        accpath = self.programpath + "/res/ " + self.dataset + "_timeCost_" + str(self.join_ratio) + "_" + str(
            self.num_clients) + "_" + str(self.alpha) + ".csv"
        print("success training write acc txt", accpath)
        redf.to_csv(accpath, mode='a', header=False)
        # -----------------------------------------------------------------------------------------------

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            print(f" self.num_new_clients is ----{ self.num_new_clients }----")
            self.eval_new_clients = True
            self.set_new_clients(clientALA_AAW)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self,round):
        assert (len(self.clients) > 0)

        #for client in self.clients:
        for client in self.selected_clients:
            client.local_initialization(self.global_model,round)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.local_initialization(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.train()

    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples += len(train_data) + len(test_data)
            client = clientObj(self.args,
                               id=i,
                               traindata = train_data,
                                testsdata = test_data,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.AAW.sizerate= (client.train_samples + client.test_samples) / samples
            client.AAW.global_model = self.global_model
        #根据label 来计算distance
        self.setdistance()

            # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def setdistance(self):
        alllabel = [0 for i in range (self.labellenght)]
        for client in self.clients:
            #print(f"before alllabel is {alllabel},client.label is{client.label}")
            for j in range(self.labellenght):
                alllabel[j]+=client.label[j]
            #print(f"after alllabel is {alllabel}")
        alldistance = 0
        for client in self.clients:
            #client.distance = jensen_shannon_distance(client.label, alllabel)
            client.distance = self.calculate_hellinger_distance(client.label, alllabel)
            alldistance += client.distance
           # print(f"INFormation-------------serveralajs----------------------init client {client.id} JS distance is {client.distance}-------------------------------")

        for client in self.clients:
            client.alldistance = alldistance
            client.alllabel = alllabel
            #print(f"set clients info : client {client.id},distance rate is {client.distance/client.alldistance},sizerate:{client.sizerate}")



    def receive_models(self,round):
        '''
        根据设定的客户端丢失率、时间阈值和客户端的训练时间消耗，
        选择符合条件的活跃客户端，并收集其模型和样本权重。最后，对样本权重进行归一化，以便后续在联邦学习中使用。
        Returns:

        '''
        assert (len(self.selected_clients) > 0)
        active_clients=self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []


        # #AAW：利用其他client的梯度对weight 进行更新，实现最终的动态迭代---------------
        for client in active_clients:
            for otherclient in active_clients:
                if client.id!=otherclient.id:
                    for ugrd, weight in zip(otherclient.AAW.updategrad,client.AAW.AAweights):
                        weight.data = torch.clamp(weight - ugrd, 0, 1)

       #——————————————————————————————————————————————————————————————————————————————————

        # active_train_samples=0
        # for client in active_clients:
        #     active_train_samples+=client.train_samples
        # active_distance = 0
        # activelabel=[0 for i in range(10)]
        # for client in active_clients:
        #     for j in range(10):
        #         activelabel[j] += client.label[j]
        #
        # for client in active_clients:
        #     client.distance =jensen_shannon_distance(client.label, activelabel)
        #     #client.distance = self.calculate_hellinger_distance(client.label, activelabel)
        #     active_distance += client.distance



        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:

                #a=0.5
                # if a == 0:
                #     jsweight = (1 / (client.distance / client.alldistance))
                # elif a==1:
                #     jsweight = client.train_samples / active_train_samples
                # else:
                #     distanceweight = (1 / (client.distance / active_distance))*(1-a)
                #     jsweight=distanceweight*a * (client.train_samples / active_train_samples)
                #print(f"origin weight is {client.train_samples / active_train_samples},jsweight is {jsweight}")

                #1.origin datasize

                #jsweight = client.train_samples / active_train_samples
                #----------

                #2. distance,目前只选择这个定义权重的公式
                # scale_factor = 0.1
                # jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
                #             client.train_samples / active_train_samples)

                #------------

                #jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor)
                #3.
                #jsweight=self.weight_from_distance1(client.distance / active_distance, scale_factor)

                #jsweight=self.weight_from_distance3( client.distance / active_distance, scale_factor, client.train_samples / active_train_samples)

                #+0.5 * (client.train_samples / active_train_samples)

               # jsweight=self.weight_from_distance3(client.distance / client.alldistance, scale_factor, client.train_samples / active_train_samples)


                #print(f"origin weight is {client.train_samples / active_train_samples},jsweight is {jsweight}")


                #self.uploaded_weights.append(client.AAW.initweight)
                #将聚合权重更新为AAW.AAweights
                self.uploaded_weights.append(client.AAW.AAweights)
                #print("aggregation weight is :",type(client.AAW.AAweights))
                self.uploaded_models.append(client.model)
                self.uploaded_ids.append(client.id)
        #记录每一轮的权重信息，用于权重变化分析
        #self.allweights.append([round, self.uploaded_weights])
        #print("before guiyihua",self.uploaded_weights)
        #self.guiyihua_weight()
        self.normalize_update_weight()
        #print("after guiyihua", self.uploaded_weights)





    def guiyihua_weight(self):

        #weights_array = np.array(self.uploaded_weights)
        total_sum = np.sum(self.uploaded_weights)
        self.uploaded_weights = [i/total_sum for i in self.uploaded_weights ]


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

    def weight_from_distance3(self,distance, scale_factor, data_size):
        '''
                  其中：
            distance 表示数据点的距离。
            scale_factor 是一个比例因子，用于调整 Sigmoid 函数的斜率。
            data_size 表示数据集的大小，影响 Sigmoid 函数的输入范围。
            通过调整 scale_factor 的值，你可以控制 Sigmoid 函数的斜率和权重的变化速率。而数据集的大小 data_size 可以影响 Sigmoid 函数的输入范围，进而影响权重的分布。
            Args:
                distance:
                scale_factor:
                data_size:

            Returns:
                Args:
                    distance:
                    scale_factor:
                    data_size:

                Returns:

        '''


        sigmoid_input = scale_factor * (data_size / 2 - distance)
        weight = 1 / (1 + math.exp(sigmoid_input))


        return weight


   #不同的距离计算

    def calculate_kl_divergence(self,p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        kl_divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return kl_divergence



    # # 示例数据
    # label1 = [10, 15, 5, 20]  # 第一个标签的样本个数分布
    # label2 = [5, 10, 10, 25]  # 第二个标签的样本个数分布
    #
    # # 计算 KL 散度
    # kl_divergence = calculate_kl_divergence(label1, label2)
    # print("KL 散度:", kl_divergence)
    #
    # # 计算 Hellinger 距离
    # hellinger_distance = calculate_hellinger_distance(label1, label2)
    # print("Hellinger 距离:", hellinger_distance)




    def select_weight_vector(self):
        '''
        初始化每个被选的客户端的聚合权重，只在参与的时候进行初始化，并且初始化的权重只与当前的参与样本总体相关=Di/Ds。
        Returns:

        '''
        active_distance = 0
        activelabel = [0 for i in range(self.labellenght)]
        active_train_samples = 0 #选择客户端的样本总数
        #统计参与客户端的信息：所有参与客户端的样本总数active_train_samples，参与客户端的编号列表activelabel
        for client in self.selected_clients:
            active_train_samples += client.train_samples
            for j in range(self.labellenght):
                activelabel[j] += client.label[j]

        # #此部分是计算客户端分布与选择总体分布的距离------
        # for client in self.selected_clients:
        #     #计算参与训练的client分布差异
        #     client.distance = jensen_shannon_distance(client.label, activelabel)
        #     active_distance += client.distance

        for client in self.selected_clients:
            # scale_factor = 0.1
            # jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
            #                 client.train_samples / active_train_samples)
            # client.AAW.initweight =jsweight
            #实际上初始化是用了Di/Ds
            client.AAW.initweight =client.train_samples / active_train_samples
            #第一次参与训练需要初始化它的weight，weight 按照js计算weight的方式进行
            if not client.AAW.hasinit:
                #初始化权重init_weight
                client.AAW.init_weight(self.global_model)
                client.AAW.hasinit=True



    def adpativeweight(self):
        pass

    def add_parameters(self, aaw, client_model):
        '''
        使用张量权重聚合模型
        Args:
            aaw:
            client_model:

        Returns:

        '''
        #print("aaw aggregator --->global model",type(aaw),type(self.global_model.parameters()),type(client_model.parameters()))
        for w,server_param, client_param in zip(aaw,self.global_model.parameters(), client_model.parameters()):
            if torch.is_tensor(w) and torch.is_tensor(client_param.data) and w.shape == client_param.data.shape:
                server_param.data += torch.mul(client_param.data.clone(), w)
            else:
                print(f"Error: serveraaw add_parameters Invalid tensor shape or type,w.shape {w.shape} but client_param.data.shape {client_param.data.shape}")

    def normalize_update_weight(self):
        '''
        tensor list进行归一化
        Returns:

        '''

        #将每一个权重对应位置的tensor进行归一化，
        tensorlists=[]
        for i in range(len(self.uploaded_weights[0])):
            tensorlists.append([])
        for weight in self.uploaded_weights:
            for j in range(len(weight)):
                tensorlists[j].append( weight[j])

        for j in range(len(tensorlists)):
            tensor_list=tensorlists[j]
            tensor_sum = sum(tensor_list)
            normalized_list = [tensor / tensor_sum for tensor in tensor_list]
            tensorlists[j]=normalized_list

        #更新self.uploaded_weights
        self.uploaded_weights=[]
        for i in range(len(tensorlists[0])):
            self.uploaded_weights.append([0 for j in range(len(tensorlists))])


        for i in range(len(tensorlists)):
            tensor_list=tensorlists[i]
            for j in range(len(tensor_list)):
                self.uploaded_weights[j][i]=tensor_list[j]













