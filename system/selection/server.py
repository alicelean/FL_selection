from dataset.generate_mnist import num_clients
from helper.clientSampler import *
from helper.client import *

def init_myprocesses(rank, size, model, queue, param_q, stop_signal, fn, backend):
    global sampledClientSet

    #初始化客户端采集器
    clientSampler = initiate_sampler_query(queue, len(num_clients))
    clientIdsToRun = []
    for nextClientIdToRun in range(num_clients):
        clientSampler.clientOnHost([nextClientIdToRun], nextClientIdToRun)
        clientIdsToRun.append([nextClientIdToRun])
        sampledClientSet.add(nextClientIdToRun)
    #搜集到所有的客户端信息，sampledClientSet同样
    # Start the PS service
    fn(model, queue, param_q, stop_signal, clientSampler)

def initiate_sampler_query(self,queue, numOfClients):
    #global logDir
    # Initiate the clientSampler
    if self.args.sampler_path is None:
        # if not args.load_model and args.sampler_path is None:
        client_sampler = clientSampler(self.args.sample_mode, self.args.score_mode, args=self.args, filter=self.args.filter_less,
                                       sample_seed=self.args.sample_seed)
    else:
        # load sampler
        args.sampler_path = os.path.join(args.model_path, 'aggregator/clientInfoFile')
        with open(args.sampler_path, 'rb') as loader:
            client_sampler = pickle.load(loader)
        logging.info("====Load sampler successfully\n")

    # load client profiles
    global_client_profile = {}
    if os.path.exists(args.client_path):
        with open(args.client_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)

    collectedClients = 0
    initial_time = time.time()
    clientId = 1
    passed = False
    num_client_profile = max(1, len(global_client_profile))

    # In this simulation, we run data split on each worker, which amplifies the # of datasets
    # Waiting for the data information from clients, or timeout
    if args.enable_obs_client:
        roundDurationList = []
        roundDurationLocalList = []
        roundDurationCommList = []
        computationList = []
        communicationList = []
        #对客户端完成度和时间两方面做限制
    while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        if not queue.empty():
            tmp_dict = queue.get()

            # we only need to go over once
            if not passed and args.sampler_path is None:
                rank_src = list(tmp_dict.keys())[0]
                distanceVec = tmp_dict[rank_src][0]
                sizeVec = tmp_dict[rank_src][1]
                for index, dis in enumerate(distanceVec):
                    # since the worker rankId starts from 1, we also configure the initial dataId as 1
                    mapped_id = max(1, clientId % num_client_profile)
                    systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [1.0,

                    #客户端采集器  里面初始化客户端的信息                                                                                1.0]
                    client_sampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                    client_sampler.registerDuration(clientId,
                                                    batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                                    model_size=args.model_size * args.clock_factor)
                    if args.enable_obs_client:
                        roundDuration, roundDurationLocal, roundDurationComm = client_sampler.getCompletionTime(
                            clientId,
                            batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                            model_size=args.model_size * args.clock_factor)

                        roundDurationList.append(roundDuration)
                        roundDurationLocalList.append(roundDurationLocal)
                        roundDurationCommList.append(roundDurationComm)
                        computationList.append(systemProfile[
                                                   'computation'])
                        communicationList.append(systemProfile[
                                                     'communication'])

                    clientId += 1

                passed = True

            collectedClients += 1

    logging.info("====Info of all feasible clients {}".format(client_sampler.getDataInfo()))
    #parser.add_argument('--enable_obs_client', type=bool, default=False, help="enable debug mode")
    if args.enable_obs_client:
        scipy.io.savemat(logDir + '/obs_client_time.mat', dict(roundDurationList=roundDurationList,
                                                               roundDurationLocalList=roundDurationLocalList,
                                                               roundDurationCommList=roundDurationCommList,
                                                               computationList=computationList,
                                                               communicationList=communicationList))
        logging.info("====Save obs_client====")
        stop_signal.put(1)

    return client_sampler


def run(model, queue, param_q, stop_signal, clientSampler):
    global logDir, sampledClientSet
    workers = [int(v) for v in str(args.learners).split('-')]
    epoch_train_loss = 0
    data_size_epoch = 0  # len(train_data), one epoch
    epoch_count = 1
    global_virtual_clock = 0.
    round_duration = 0.
    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
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

    gradient_controller = None
    # initiate yogi if necessary
    if args.gradient_policy == 'yogi':
        gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

    clientInfoFile = os.path.join(logDir, 'clientInfoFile')
    # dump the client info
    with open(clientInfoFile, 'wb') as fout:
        # pickle.dump(clientSampler.getClientsInfo(), fout)
        pickle.dump(clientSampler, fout)
    if args.load_model:
        training_history_path = os.path.join(args.model_path, 'aggregator/training_perf')
        with open(training_history_path, 'rb') as fin:
            training_history = pickle.load(fin)
        load_perf_epoch_retrieved = list(training_history['perf'].keys())
        load_perf_epoch = load_perf_epoch_retrieved[-1]
        load_perf_clock = training_history['perf'][load_perf_epoch]['clock']

    else:
        training_history = {'data_set': args.data_set,
                            'model': args.model,
                            'sample_mode': args.sample_mode,
                            'gradient_policy': args.gradient_policy,
                            'task': args.task,
                            'perf': collections.OrderedDict()}

        load_perf_clock = 0
        load_perf_epoch = 0

    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]

                [iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock] = \
                    [tmp_dict[rank_src][i] for i in range(1, len(tmp_dict[rank_src]))]
                # clientSampler.registerSpeed(rank_src, clientId, speed)

                if isWorkerEnd:
                    logging.info("====Worker {} has completed all its data computation!".format(rank_src))
                    learner_staleness.pop(rank_src)
                    if (len(learner_staleness) == 0):
                        stop_signal.put(1)
                        break
                    continue

                learner_local_step[rank_src] += 1

                handlerStart = time.time()
                delta_wss = tmp_dict[rank_src][0]
                clientsLastEpoch += clientIds
                ratioSample = 0

                logging.info("====Start to merge models")
                if args.enable_obs_local_epoch and epoch_count > 1:
                    gradient_l2_norm_list = []
                    gradientUtilityList = []
                if not args.test_only or epoch_count == 1:
                    for i, clientId in enumerate(clientIds):
                        gradients = None
                        ranSamples = float(speed[i].split('_')[1])

                        data_size_epoch += trained_size[i]

                        # fraction of total samples on this specific node
                        ratioSample = clientSampler.getSampleRatio(clientId, rank_src, args.is_even_avg)
                        delta_ws = delta_wss[i]
                        # clientWeightsCache[clientId] = [torch.from_numpy(x).to(device=device) for x in delta_ws]
                        # TODO:ADD LOSS AVERAGELY
                        epoch_train_loss += ratioSample * iteration_loss[i]
                        isSelected = True if clientId in sampledClientSet else False
                        gradient_l2_norm = 0
                        # apply the update into the global model if the client is involved
                        for idx, param in enumerate(model.parameters()):
                            model_weight = torch.from_numpy(delta_ws[idx]).to(device=device)
                            # model_weight is the delta of last model
                            if isSelected:
                                # the first received client
                                if received_updates == 0:
                                    sumDeltaWeights.append(model_weight * ratioSample)
                                else:
                                    sumDeltaWeights[idx] += model_weight * ratioSample
                            gradient_l2_norm += ((model_weight - last_model_parameters[idx]).norm(2) ** 2).item()
                        # bias term for global speed
                        virtual_c = virtualClientClock[clientId] if clientId in virtualClientClock else 1.
                        clientUtility = 1.
                        size_of_sample_bin = 1.

                        if args.capacity_bin == True:
                            if not args.enable_adapt_local_epoch:
                                size_of_sample_bin = min(clientSampler.getClient(clientId).size,
                                                         args.upload_epoch * args.batch_size)
                            else:
                                size_of_sample_bin = min(clientSampler.getClient(clientId).size, trained_size[i])

                        # register the score
                        clientUtility = math.sqrt(iteration_loss[i]) * size_of_sample_bin
                        gradientUtility = math.sqrt(gradient_l2_norm) * size_of_sample_bin / 100
                        if args.enable_obs_local_epoch and epoch_count > 1:
                            gradient_l2_norm_list.append(gradient_l2_norm)
                            gradientUtilityList.append(gradientUtility)
                        # add noise to the utility
                        if args.noise_factor > 0:
                            noise = np.random.normal(0, args.noise_factor * median_reward, 1)[0]
                            clientUtility += noise
                            clientUtility = max(1e-2, clientUtility)

                        clientSampler.registerScore(clientId, clientUtility, gradientUtility,
                                                    auxi=math.sqrt(iteration_loss[i]), time_stamp=epoch_count,
                                                    duration=virtual_c)

                        if isSelected:
                            received_updates += 1

                        avgUtilLastEpoch += ratioSample * clientUtility
                        avgGradientUtilLastEpoch += ratioSample * gradientUtility


                logging.info(
                    "====Done handling rank {}, with ratio {}, now collected {} clients".format(rank_src, ratioSample,
                                                                                                received_updates))
                if args.enable_obs_local_epoch and epoch_count > 1:
                    scipy.io.savemat(logDir + '/obs_local_epoch_gradient.mat',
                                     dict(gradient_l2_norm_list=gradient_l2_norm_list,
                                          gradientUtilityList=gradientUtilityList))
                    logging.info("====Save obs_local_epoch====")
                    stop_signal.put(1)
                # aggregate the test results
                updateEpoch = testRes[-1]
                if updateEpoch not in test_results:
                    # [top_1, top_5, loss, total_size, # of collected ranks]
                    test_results[updateEpoch] = [0., 0., 0., 0., 0]

                if updateEpoch != -1:
                    for idx, c in enumerate(testRes[:-1]):
                        test_results[updateEpoch][idx] += c

                    test_results[updateEpoch][-1] += 1
                    # have collected all ranks
                    if test_results[updateEpoch][-1] == len(workers):
                        top_1_str = 'top_1: '
                        top_5_str = 'top_5: '
                        try:
                            logging.info(
                                "====After aggregation in epoch: {}, virtual_clock: {}, {}: {} % ({}), {}: {} % ({}), test loss: {}, test len: {}"
                                .format(updateEpoch + load_perf_epoch, global_virtual_clock + load_perf_clock,
                                        top_1_str,
                                        round(test_results[updateEpoch][0] / test_results[updateEpoch][3] * 100.0, 4),
                                        test_results[updateEpoch][0], top_5_str,
                                        round(test_results[updateEpoch][1] / test_results[updateEpoch][3] * 100.0, 4),
                                        test_results[updateEpoch][1],
                                        test_results[updateEpoch][2] / test_results[updateEpoch][3],
                                        test_results[updateEpoch][3]))
                            if not args.load_model or epoch_count > 2:
                                training_history['perf'][updateEpoch + load_perf_epoch] = {
                                    'round': updateEpoch + load_perf_epoch,
                                    'clock': global_virtual_clock + load_perf_clock,
                                    top_1_str: round(
                                        test_results[updateEpoch][0] / test_results[updateEpoch][3] * 100.0, 4),
                                    top_5_str: round(
                                        test_results[updateEpoch][1] / test_results[updateEpoch][3] * 100.0, 4),
                                    'loss': test_results[updateEpoch][2] / test_results[updateEpoch][3],
                                    }

                                with open(os.path.join(logDir, 'training_perf'), 'wb') as fout:
                                    pickle.dump(training_history, fout)

                        except Exception as e:
                            logging.info(f"====Error {e}")

                handlerDur = time.time() - handlerStart
                global_update += 1

                # get the current minimum local staleness_sum_epoch，最小的模型
                currentMinStep = min([learner_local_step[key] for key in learner_local_step.keys()])

                staleness += 1
                learner_staleness[rank_src] = staleness

                # if the worker is within the staleness, then continue w/ local cache and do nothing
                # Otherwise, block it
                if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    # lock the worker
                    logging.info("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                 " , while globalStep is " + str(currentMinStep) + "\n")

                # if the local cache is too stale, then update it
                elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]

                # release all pending requests, if the staleness does not exceed the staleness threshold in SSP
                handle_dur = time.time() - handle_start

                workersToSend = []

                for pworker in pendingWorkers.keys():
                    # check its staleness，pworker没有参与的工作节点
                    if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                        # start to send param, to avoid synchronization problem, first create a copy here?
                        workersToSend.append(pworker)

                del delta_wss, tmp_dict

                if len(workersToSend) > 0:
                    # assign avg reward to explored, but not ran workers
                    for clientId in exploredPendingWorkers:
                        clientSampler.registerScore(clientId, avgUtilLastEpoch, avgGradientUtilLastEpoch,
                                                    time_stamp=epoch_count, duration=virtualClientClock[clientId],
                                                    success=False
                                                    )

                    workersToSend = sorted(workersToSend)
                    epoch_count += 1
                    avgUtilLastEpoch = 0.
                    avgGradientUtilLastEpoch = 0.
                    logging.info(
                        "====Epoch {} completes {} clients with loss {}, sampled rewards are: \n {} \n=========="
                        .format(epoch_count, len(clientsLastEpoch), epoch_train_loss,
                                {x: clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}))

                    epoch_train_loss = 0.
                    clientsLastEpoch = []
                    send_start = time.time()

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0 or epoch_count == 2:
                        logging.info("====Start to sample for epoch {}, global virtualClock: {}, round_duration: {}"
                                     .format(epoch_count, global_virtual_clock, round_duration))

                        numToSample = int(args.total_worker * args.overcommit)

                        if args.fixed_clients and last_sampled_clients:
                            sampledClientsRealTemp = last_sampled_clients
                        else:
                            sampledClientsRealTemp = sorted(
                                clientSampler.resampleClients(numToSample, cur_time=epoch_count))

                        last_sampled_clients = sampledClientsRealTemp

                        # remove dummy clients that we are not going to run
                        clientsToRun, exploredPendingWorkers, virtualClientClock, round_duration, clients_to_run_local_epoch_ratio, clients_to_run_dropout_ratio = prune_client_tasks(
                            clientSampler, sampledClientsRealTemp, args.total_worker, global_virtual_clock)
                        sampledClientSet = set(clientsToRun)

                        logging.info("====Try to resample clients, final takes: \n {}"
                                     .format(clientsToRun, ))  # virtualClientClock))

                        allocateClientToWorker = {}
                        allocateClientLocalEpochToWorker = {}
                        allocateClientDropoutRatioToWorker = {}
                        allocateClientDict = {rank: 0 for rank in workers}

                        # for those device lakes < # of clients, we use round-bin for load balance
                        for idc, c in enumerate(clientsToRun):
                            clientDataSize = clientSampler.getClientSize(c)
                            numOfBatches = int(math.ceil(clientDataSize / args.batch_size))

                            if numOfBatches > args.upload_epoch:
                                workerId = workers[(c - 1) % len(workers)]
                            else:
                                # pick the one w/ the least load
                                workerId = sorted(allocateClientDict, key=allocateClientDict.get)[0]

                            if workerId not in allocateClientToWorker:
                                allocateClientToWorker[workerId] = []
                                allocateClientLocalEpochToWorker[workerId] = []
                                allocateClientDropoutRatioToWorker[workerId] = []

                            allocateClientToWorker[workerId].append(c)
                            allocateClientLocalEpochToWorker[workerId].append(clients_to_run_local_epoch_ratio[idc])
                            allocateClientDropoutRatioToWorker[workerId].append(clients_to_run_dropout_ratio[idc])
                            allocateClientDict[workerId] = allocateClientDict[workerId] + 1

                        for w in allocateClientToWorker.keys():
                            clientSampler.clientOnHost(allocateClientToWorker[w], w)
                            clientSampler.clientLocalEpochOnHost(allocateClientLocalEpochToWorker[w], w)
                            clientSampler.clientDropoutratioOnHost(allocateClientDropoutRatioToWorker[w], w)

                    clientIdsToRun = [currentMinStep]
                    clientsList = []
                    clientsListLocalEpoch = []
                    clientsListDropoutRatio = []

                    endIdx = 0

                    for worker in workers:
                        learner_cache_step[worker] = currentMinStep
                        endIdx += clientSampler.getClientLenOnHost(worker)
                        clientIdsToRun.append(endIdx)
                        clientsList += clientSampler.getCurrentClientIds(worker)
                        clientsListLocalEpoch += clientSampler.getCurrentClientLocalEpoch(worker)
                        clientsListDropoutRatio += clientSampler.getCurrentClientDropoutRatio(worker)
                        # remove from the pending workers
                        del pendingWorkers[worker]

                    # transformation of gradients if necessary
                    if gradient_controller is not None:
                        sumDeltaWeights = gradient_controller.update(sumDeltaWeights)

                    # update the clientSampler and model
                    with open(clientInfoFile, 'wb') as fout:
                        pickle.dump(clientSampler, fout)
                    for idx, param in enumerate(model.parameters()):
                        if not args.test_only:
                            if (not args.load_model or epoch_count > 2):
                                param.data += sumDeltaWeights[idx]
                            dist.broadcast(tensor=(param.data.to(device=device)), src=0)



                    last_model_parameters = [torch.clone(p.data) for p in model.parameters()]

                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    # update the virtual clock
                    global_virtual_clock += round_duration
                    received_updates = 0

                    sumDeltaWeights = []
                    clientWeightsCache = {}

                    if args.noise_factor > 0:
                        median_reward = clientSampler.get_median_reward()
                        logging.info('For epoch: {}, median_reward: {}, dev: {}'
                                     .format(epoch_count, median_reward, median_reward * args.noise_factor))

                    gc.collect()

                # The training stop,训练轮次
                if (epoch_count >= args.epochs):
                    stop_signal.put(1)
                    logging.info('Epoch is done: {}'.format(epoch_count))
                    break

            except Exception as e:
                print("====Error: " + str(e) + '\n')

        e_time = time.time()

        if (e_time - s_time) >= float(args.timeout):
            print('Time up: {}, Stop Now!'.format(e_time - s_time))
            break


init_myprocesses(this_rank, world_size, model,
                    q, param_q, stop_signal, run, args.backend
                )