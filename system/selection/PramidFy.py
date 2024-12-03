from selection.helper.Selecter import *
import queue,torch,gc
import json
class PramidFy(Selecter):
    def __init__(self, args,num_join_clients,client_path):
        super().__init__(args)
        #存放所有客户端的基本信息
        self.args=args
        #self.mode='PramidFy'
        self.client_path=client_path
        self.mode='oort'
        self.num_join_clients=num_join_clients
        self.exploredPendingWorkers=[]
        self.InfoQueue=queue.Queue()
        self.learner_cache_step = {l: 0 for l in range(args.num_clients)}
        self.learner_local_step = {l: 0 for l in range(args.num_clients)}

    def prune_client_tasks(self,clientSampler, sampledClientsRealTemp, numToRealRun, global_virtual_clock):
        # 从已采样的客户端中筛选出一部分真实运行的客户端（剔除离线客户端和慢速客户端），并根据一些条件调整客户端的本地训练轮数和掉线率
        args=self.args

          # sampledClientsReal提出不可用之后的客户端
        sampledClientsReal = []
        # 1. remove dummy clients that are not available to the end of training，从临时采样客户端中剔除那些在当前轮训练结束前不可用的客户端（即离线客户端）。
        for virtualClient in sampledClientsRealTemp:
            # 完整一轮训练的总时间、本地训练时间、通信时间
            roundDuration, roundDurationLocal, roundDurationComm = clientSampler.getCompletionTime(virtualClient,
                                                                                                   batch_size=args.batch_size,
                                                                                                   upload_epoch=args.upload_epoch,
                                                                                                   model_size=args.model_size * args.clock_factor)
            # print(
            #     f"virtualClient is{virtualClient},roundDuration is {roundDuration}, roundDurationLocal is {roundDurationLocal}, roundDurationComm is {roundDurationComm}")

            if clientSampler.isClientActive(virtualClient, roundDuration + global_virtual_clock):
                sampledClientsReal.append(virtualClient)

        # 2. we decide to simulate the wall time and remove 1. stragglers 2. off-line
        # 计算出每个筛选后的客户端的训练完成时间，并记录它们的通信、本地训练时间以及梯度
        completionTimes = []  # 记录每个客户端的总训练时间
        virtual_client_clock = {}
        # 分别记录本地训练和通信时间
        completionTimesLocal = []
        completionTimesComm = []
        # 记录每个客户端的梯度反馈，用于后续的奖励排名
        rewardListRaw = []
        for virtualClient in sampledClientsReal:
            roundDuration, roundDurationLocal, roundDurationComm = clientSampler.getCompletionTime(virtualClient,
                                                                                                   batch_size=args.batch_size,
                                                                                                   upload_epoch=args.upload_epoch,
                                                                                                   model_size=args.model_size * args.clock_factor)


            completionTimes.append(roundDuration)

            completionTimesLocal.append(roundDurationLocal)

            completionTimesComm.append(roundDurationComm)

            feedback = clientSampler.getClientGradient(virtualClient)
            rewardListRaw.append(feedback['gradient'])
            virtual_client_clock[virtualClient] = roundDuration

            # print(
            #     f"virtualClient2 is{virtualClient},,gradinet is {feedback['gradient']},roundDuration is {roundDuration}, roundDurationLocal is {roundDurationLocal}, roundDurationComm is {roundDurationComm}")


        #print(f"prune_client_tasks run len(sampledClientsRealTemp) is{len(sampledClientsRealTemp)} detail is {sampledClientsRealTemp} ,sampledClientsReal is{sampledClientsReal}")

        # 对客户端按照completionTimes排序，得到排序后的索引列表
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k: completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:numToRealRun]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        # print(
        #     f"virtual_client_clock is{virtual_client_clock},completionTimes is {completionTimes} sortedWorkersByCompletion{sortedWorkersByCompletion} numToRealRun {numToRealRun} top_k_index {top_k_index},clients_to_run {clients_to_run}")

        ## TODO: return the adaptive local epoch
        # 其余的客户端作为 dummy_clients，表示它们不会参与本轮训练。
        dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[numToRealRun:]]
        # 所选客户端中的最大完成时间 round_duration，这代表本轮训练的总时间
        round_duration = completionTimes[top_k_index[-1]]
        # 计算客户端的奖励排名，并基于此调整掉线率
        # rewardListRanking用于保存每个客户端的排名
        rewardList = [rewardListRaw[k] for k in top_k_index]
        rewardListSorted = sorted(rewardList, reverse=True)
        rewardListRanking = [rewardListSorted.index(rewardList[k]) for k in range(len(rewardList))]
        #print(f"rewardList   :{rewardList},rewardListRanking  k :{rewardListRanking}")

        # 如果启用了 dropout，则根据客户端的排名调整掉线率。掉线率会影响通信时间，进而调整客户端的总训练时间
        if args.enable_dropout:
            increment_factor = (args.dropout_high - args.dropout_low) / args.total_worker
            clients_to_run_dropout_ratio = [args.dropout_low + k * increment_factor for k in rewardListRanking]
            for k_index, k in enumerate(top_k_index):
                completionTimes[k] = completionTimesLocal[k] + (1 - clients_to_run_dropout_ratio[k_index]) * \
                                     completionTimesComm[k]
            round_duration = max([completionTimes[k] for k in top_k_index])
        else:
            clients_to_run_dropout_ratio = [0 for k in top_k_index]
        # 根据训练时间的差异动态调整每个客户端的本地训练轮数，如果没有启用，则默认本地训练轮数为1
        if args.enable_adapt_local_epoch:
            clients_to_run_local_epoch_ratio = [min(10, args.adaptive_epoch_beta * math.floor(
                (round_duration - completionTimes[k]) / (
                            completionTimesLocal[k] / args.upload_epoch)) / args.upload_epoch) + 1 for k in top_k_index]
        else:
            clients_to_run_local_epoch_ratio = [1 for k in top_k_index]

        # 启用了观测功能，将各客户端的完成时间、本地时间、通信时间等数据保存到文件中，供后续分析使用。
        if args.enable_obs_local_epoch:
            scipy.io.savemat(logDir + '/obs_local_epoch_time.mat',
                             dict(completionTimes=[completionTimes[k] for k in sortedWorkersByCompletion],
                                  completionTimesLocal=[completionTimesLocal[k] for k in sortedWorkersByCompletion],
                                  completionTimesComm=[completionTimesComm[k] for k in sortedWorkersByCompletion],
                                  rewardListRaw=[rewardListRaw[k] for k in sortedWorkersByCompletion]))


        #print("=" * 20, f"prune_client_tasks end ,clients_to_run_dropout_ratio is{clients_to_run_dropout_ratio},clients_to_run_local_epoch_ratio is{clients_to_run_local_epoch_ratio}")

        return clients_to_run, dummy_clients, virtual_client_clock, round_duration, clients_to_run_local_epoch_ratio, clients_to_run_dropout_ratio

    def run(self,model, queue, stop_signal, clientSampler,sampledClientSet,round=-1):
        print(f"PramidFy run and get sample")
        args=self.args
        device_id = args.gpu_device
        device = torch.device('cuda:' + str(device_id))
        workers = [i for i in range(self.args.num_clients)]
        epoch_train_loss = 0
        data_size_epoch = 0  # len(train_data), one epoch
        epoch_count = 1
        global_virtual_clock = 0.
        round_duration = 0.
        staleness = 0
        #跟踪每个客户端的训练延迟
        learner_staleness = {l: 0 for l in range(args.num_clients)}
        learner_cache_step = {l: 0 for l in range(args.num_clients)}

        pendingWorkers = {}
        test_results = {}
        virtualClientClock = {}
        exploredPendingWorkers = []
        avgUtilLastEpoch = 0.
        avgGradientUtilLastEpoch = 0.
        s_time = time.time()
        epoch_time = s_time
        global_update = 0
        received_updates = 0
        clientsLastEpoch = []
        sumDeltaWeights = []
        clientWeightsCache = {}
        last_sampled_clients = None
        last_model_parameters = [torch.clone(p.data) for p in model.parameters()]

        # random component to generate noise
        median_reward = 1.

        # gradient_controller = None
        # # initiate yogi if necessary
        # if self.args.gradient_policy == 'yogi':
        #     print(f"self.args.gradient_policy is {self.args.gradient_policy}")
        #     gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

        # clientInfoFile = os.path.join(logDir, 'clientInfoFile')
        # # dump the client info
        # with open(clientInfoFile, 'wb') as fout:
        #     # pickle.dump(clientSampler.getClientsInfo(), fout)
        #     pickle.dump(clientSampler, fout)

        # if args.load_model:
        #     training_history_path = os.path.join(args.model_path, 'aggregator/training_perf')
        #     with open(training_history_path, 'rb') as fin:
        #         training_history = pickle.load(fin)
        #     load_perf_epoch_retrieved = list(training_history['perf'].keys())
        #     load_perf_epoch = load_perf_epoch_retrieved[-1]
        #     load_perf_clock = training_history['perf'][load_perf_epoch]['clock']
        #
        # else:
        #     training_history = {'data_set': args.data_set,
        #                         'model': args.model,
        #                         'sample_mode': args.sample_mode,
        #                         'gradient_policy': args.gradient_policy,
        #                         'task': args.task,
        #                         'perf': collections.OrderedDict()}
        #
        #     load_perf_clock = 0
        #     load_perf_epoch = 0
        workersToSend = []
        avgUtilLastEpoch = 0.
        epoch_train_loss = 0.
        clientsLastEpoch = []
        avgGradientUtilLastEpoch = 0.

        while True:
            if queue.empty():
                break
            else:
                #1.检查消息队列，获取数据
                handlerStart = time.time()
                #存放梯度信息
                if args.enable_obs_local_epoch and epoch_count > 1:
                    gradient_l2_norm_list = []
                    gradientUtilityList = []
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]
                [iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock] = \
                    [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                workersToSend.append(rank_src)






                # clientSampler.registerSpeed(rank_src, clientId, speed)
                # print("client info is :",rank_src,iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock)
                # if isWorkerEnd:
                #     logging.info("====Worker {} has completed all its data computation!".format(rank_src))
                #     learner_staleness.pop(rank_src)
                #     # print("=="*10,"learner_staleness is :",learner_staleness)
                #     if (len(learner_staleness) == 0):
                #         stop_signal.put(1)
                #         break
                    # continue

                #learner_local_step[rank_src] += 1
                self.learner_local_step[rank_src]=round
                delta_wss = tmp_dict[rank_src][0]
                #记录上一轮的客户端id
                clientsLastEpoch += clientIds

               # logging.info("====Start to merge models")
                #local_epoch怎么调整？


                # test_only=False，not args.test_only=True
                #处理客户端数据-----
                for i, clientId in enumerate(clientIds):
                    gradients = None
                    gradient_l2_norm = 0
                    ranSamples = float(speed[i].split('_')[1])
                    data_size_epoch += trained_size[i]
                    # fraction of total samples on this specific node
                    ratioSample = clientSampler.getSampleRatio(clientId, rank_src, args.is_even_avg)
                    delta_ws = delta_wss[i]
                    # clientWeightsCache[clientId] = [torch.from_numpy(x).to(device=device) for x in delta_ws]
                    # TODO:ADD LOSS AVERAGELY
                    #print(clientId,"iteration_loss[i]",iteration_loss[i],ratioSample)
                    epoch_train_loss += ratioSample * iteration_loss[i]
                    isSelected = True if clientId in sampledClientSet else False

                    # apply the update into the global model if the client is involved
                    #2.记录客户端的模型信息和模型差异信息-----
                    for idx, param in enumerate(model.parameters()):
                        if args.mechinedevice == 'cpu':
                            model_weight = torch.from_numpy(delta_ws[idx])
                        else:
                            model_weight = torch.from_numpy(delta_ws[idx]).to(device=device)

                        # model_weight is the delta of last model
                        #进行模型更新
                        if isSelected:
                            # the first received client
                            if received_updates == 0:

                                sumDeltaWeights.append(model_weight * ratioSample)
                            else:
                                sumDeltaWeights[idx] += model_weight * ratioSample

                        gradient_l2_norm += ((model_weight - last_model_parameters[idx]).norm(2) ** 2).item()



                    # bias term for global speed
                    virtual_c = virtualClientClock[clientId] if clientId in virtualClientClock else 1.
                    size_of_sample_bin = 1.
                    if args.capacity_bin == True:
                        if not args.enable_adapt_local_epoch:
                            size_of_sample_bin = min(clientSampler.getClient(clientId).size,
                                                     args.upload_epoch * args.batch_size)
                            # print(f"round is {round},1,size_of_sample_bin", size_of_sample_bin)
                        else:
                            size_of_sample_bin = min(clientSampler.getClient(clientId).size, trained_size[i])


                    # 3.register the score----------------------
                    clientUtility = math.sqrt(iteration_loss[i]) * size_of_sample_bin
                    gradientUtility = math.sqrt(gradient_l2_norm) * size_of_sample_bin / 100

                    # print(
                    #     f"queue is {queue.qsize()},round is {round},clientId is {clientId},,clientUtility is {clientUtility},gradientUtility is {gradientUtility},data_size_epoch is :",
                    #     data_size_epoch,
                    #     f"ratioSample is {ratioSample},epoch_train_loss is {epoch_train_loss},size_of_sample_bin is{size_of_sample_bin},sampledClientSet is {sampledClientSet}")
                    # #记录每个客户端的梯度效用和梯度范数
                    if args.enable_obs_local_epoch and epoch_count > 1:
                        gradient_l2_norm_list.append(gradient_l2_norm)
                        gradientUtilityList.append(gradientUtility)
                    # add noise to the utility
                    # if args.noise_factor > 0:
                    #     noise = np.random.normal(0, args.noise_factor * median_reward, 1)[0]
                    #     clientUtility += noise
                    #     clientUtility = max(1e-2, clientUtility)

                    clientSampler.registerScore(clientId, clientUtility, gradientUtility,
                                                auxi=math.sqrt(iteration_loss[i]), time_stamp=epoch_count,
                                                duration=virtual_c)

                    if isSelected:
                        received_updates += 1

                    avgUtilLastEpoch += ratioSample * clientUtility
                    avgGradientUtilLastEpoch += ratioSample * gradientUtility
                handlerDur = time.time() - handlerStart

                global_update += 1
                # get the current minimum local staleness_sum_epoch，最小的模型
                #若当前客户端的本地步数 learner_local_step[rank_src] 超过阈值 args.stale_threshold + currentMinStep，
                # 则将其添加到 pendingWorkers 字典中，并记录其步数。这样的客户端会被挂起，等待同步。
                # 若客户端的本地缓存步数 learner_cache_step[rank_src] 落后于当前步数 learner_local_step[rank_src] - args.stale_threshold，
                # 则将其也加入 pendingWorkers 字典中，准备进行同步更新。

                currentMinStep = min([self.learner_local_step[key] for key in self.learner_local_step.keys()])
                staleness += 1
                learner_staleness[rank_src] = staleness

                # if the worker is within the staleness, then continue w/ local cache and do nothing
                # Otherwise, block it
                # if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                #     pendingWorkers[rank_src] = learner_local_step[rank_src]
                #     # lock the worker
                #     logging.info(
                #         "Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                #         " , while globalStep is " + str(currentMinStep) + "\n")
                #
                # # if the local cache is too stale, then update it,stale_threshold=0
                # elif self.learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold:
                #     pendingWorkers[rank_src] = learner_local_step[rank_src]

                # release all pending requests, if the staleness does not exceed the staleness threshold in SSP
                # handle_dur = time.time() - handle_start



                # for pworker in pendingWorkers.keys():
                #     # check its staleness，pworker没有参与的工作节点
                #     if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                #         # start to send param, to avoid synchronization problem, first create a copy here?
                #         workersToSend.append(pworker)

                del delta_wss, tmp_dict
                #将workend改成了当前被选中的客户端


        if len(workersToSend) > 0:
            # assign avg reward to explored, but not ran workers,exploredPendingWorkers={}
            print(f"exploredPendingWorkers is {exploredPendingWorkers}")
            for clientId in exploredPendingWorkers:
                clientSampler.registerScore(clientId, avgUtilLastEpoch, avgGradientUtilLastEpoch,
                                                    time_stamp=round, duration=virtualClientClock[clientId],
                                                    success=False
                                                    )

            workersToSend = sorted(workersToSend)

            # resampling the clients if necessary
            if round % args.resampling_interval == 0 or round == 2:
            #执行此段,选择客户端
                sampledClientsRealTemp = sorted(
                                clientSampler.resampleClients(self.num_join_clients, cur_time=round))

                last_sampled_clients = sampledClientsRealTemp

                # remove dummy clients that we are not going to run
                # 实际运行的客户端
                # clientsToRun、探索未运行的客户端
                # exploredPendingWorkers、虚拟时钟
                # virtualClientClock、轮次持续时间
                # round_duration，以及本地训练轮次比率
                # clients_to_run_local_epoch_ratio
                # 和丢包率clients_to_run_dropout_ratio。
                clientsToRun, exploredPendingWorkers, virtualClientClock, round_duration, clients_to_run_local_epoch_ratio, clients_to_run_dropout_ratio = self.prune_client_tasks(
                            clientSampler, sampledClientsRealTemp, self.num_join_clients, global_virtual_clock)
                sampledClientSet = set(clientsToRun)
                re_equal_ignore_order = set(sampledClientsRealTemp) == set(sampledClientSet)
                print(f"round is {round}，currentMinStep is {currentMinStep},workersToSend {workersToSend},sampledClientsRealTemp is{sampledClientsRealTemp},clientsToRun is {clientsToRun}")

            #print(f"sampledClientsRealTemp and sampledClientSet is {re_equal_ignore_order}，exploredPendingWorkers {exploredPendingWorkers}")



                    # 这段代码的核心目的是根据客户端数据量和批次数量，使用两种策略（轮询或按负载分配）将客户端任务分配给合适的worker，确保负载均衡。
                    # 每个worker分配的客户端还带有相关的本地训练轮数和掉线率信息，这些信息会通过clientSampler传递并应用到实际训练中。
            #采集好的客户端重新分配给工作节点。
            allocateClientToWorker = {}
            #本地训练轮数
            allocateClientLocalEpochToWorker = {}
            #掉线率信息
            allocateClientDropoutRatioToWorker = {}
            allocateClientDict = {rank: 0 for rank in workers}
            #print(f"allocateClientDict {allocateClientDict}，clientsToRun {clientsToRun}")
            # for those device lakes < # of clients, we use round-bin for load balance,clientsToRun 选中的ID
            #只考虑那些客户端
            print(f"clientsToRun len is{len(clientsToRun)}")
            for idc, id in enumerate(clientsToRun):
                clientDataSize = clientSampler.getClientSize(id)
                numOfBatches = int(math.ceil(clientDataSize / args.batch_size))
                #重新分配
                workerId=id
                if workerId not in allocateClientToWorker:
                    allocateClientToWorker[workerId] = []
                    allocateClientLocalEpochToWorker[workerId] = []
                    allocateClientDropoutRatioToWorker[workerId] = []

                allocateClientToWorker[workerId].append(id)
                #更新各自的local epoch,dropout ratio
                allocateClientLocalEpochToWorker[workerId].append(clients_to_run_local_epoch_ratio[idc])
                allocateClientDropoutRatioToWorker[workerId].append(clients_to_run_dropout_ratio[idc])

                allocateClientDict[workerId] = allocateClientDict[workerId] + 1

            #更新客户端的信息（客户端采集器的）
            for workerId in allocateClientToWorker.keys():
                #print("clientOnHost before",clientSampler.getclientOnHost(workerId))
                clientSampler.clientOnHost(allocateClientToWorker[workerId], workerId)
                #print("after", clientSampler.getclientOnHost(workerId))
                #print(" clientLocalEpochOnHost before", clientSampler.getclientLocalEpochOnHost(workerId))
                clientSampler.clientLocalEpochOnHost(allocateClientLocalEpochToWorker[workerId], workerId)
                #print("after", clientSampler.getclientLocalEpochOnHost(workerId))
                #print("clientDropoutratioOnHost before", clientSampler.getclientDropoutratioOnHost(workerId))
                clientSampler.clientDropoutratioOnHost(allocateClientDropoutRatioToWorker[workerId], workerId)
                #print("after", clientSampler.getclientDropoutratioOnHost(workerId))
                if allocateClientToWorker[workerId][0]!=workerId:
                    print(f"ERROR workerId is {workerId}, {allocateClientToWorker[workerId]}")



            clientIdsToRun = [currentMinStep]
            clientsList = []
            clientsListLocalEpoch = []
            clientsListDropoutRatio = []

            endIdx = 0
            print(f"round is {round}, currentMinStep {currentMinStep}")
            # for worker in workers:
            #             #print(f"self.learner_cache_step[worker]  {self.learner_cache_step[worker] }")
            #     self.learner_cache_step[worker] = currentMinStep
            #             #返回工作节点上的客户端的数量
            #     endIdx += clientSampler.getClientLenOnHost(worker)
            #     clientIdsToRun.append(endIdx)
            #     clientsList += clientSampler.getCurrentClientIds(worker)
            #     clientsListLocalEpoch += clientSampler.getCurrentClientLocalEpoch(worker)
            #     clientsListDropoutRatio += clientSampler.getCurrentClientDropoutRatio(worker)
            #             # print(
            #             #     f"endIdx {endIdx} worker{worker} clientsListDropoutRatio {clientsListDropoutRatio} clientsListLocalEpoch {clientsListLocalEpoch}clientsList {clientsList}clientSampler.getCurrentClientLocalEpoch(worker) {clientSampler.getCurrentClientLocalEpoch(worker)}")
            #             # print(
            #             #     f"clientSampler.getCurrentClientDropoutRatio(worker) {clientSampler.getCurrentClientDropoutRatio(worker)}")
            #
            #             # remove from the pending workers
            #             #del pendingWorkers[worker]
            #
            #         # transformation of gradients if necessary
            #         # if gradient_controller is not None:
            #         #     print(f"gradient_controller {gradient_controller}")
            #         #     sumDeltaWeights = gradient_controller.update(sumDeltaWeights)
            #
            #         # update the clientSampler and model
            #         # with open(clientInfoFile, 'wb') as fout:
            #         #     pickle.dump(clientSampler, fout)
            #         #这里对模型进行了聚合？
            #         # if len(sumDeltaWeights) < len(list(model.parameters())):
            #         #     # Pad sumDeltaWeights with zeros for missing elements
            #         #     for _ in range(len(list(self.global_model.parameters())) - len(sumDeltaWeights)):
            #         #         sumDeltaWeights.append(torch.zeros_like(next(self.global_model.parameters())))
            #
            # for idx, param in enumerate(model.parameters()):
            #     if not args.test_only:
            #         if (not args.load_model or epoch_count > 2):
            #                     # Add debug information before the problematic
            #             if idx >= len(sumDeltaWeights):
            #                 print(f"round is {round},model.parametersError: idx {idx} is out of range for sumDeltaWeights with length {len(sumDeltaWeights)}")
            #                 break
            #             print(f"round is {round},Updating model param {idx}: param size = {param.size()}, sumDeltaWeights size = {sumDeltaWeights[idx].size()}")
            #
            #             param.data += sumDeltaWeights[idx]
            #                 # dist.broadcast(tensor=(param.data.to(device=device)), src=0)

            # last_model_parameters = [torch.clone(p.data) for p in model.parameters()]


            # update the virtual clock
            global_virtual_clock += round_duration
            received_updates = 0

            # for idx, param in enumerate(model.parameters()):
            #     param.data += sumDeltaWeights[idx]


            gc.collect()

                # The training stop,训练轮次
        # if (epoch_count >= args.epochs):
        #             stop_signal.put(1)
        #             logging.info('Epoch is done: {}'.format(epoch_count))
        #             break


            # e_time = time.time()
            #
            # if (e_time - s_time) >= float(args.timeout):
            #     print('Time up: {}, Stop Now!'.format(e_time - s_time))
            #     break

        return sampledClientSet


