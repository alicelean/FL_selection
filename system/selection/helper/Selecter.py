from selection.helper.clientSampler import *
import time,os,queue
class Selecter(object):
    def __init__(self,args):
        self.sampler_path=None
        self.args=args
        self.model_path=None
        self.client_path=None
        self.enable_obs_client=False
        self.batch_size = args.batch_size
        self.upload_epoch = args.upload_epoch
        self.model_size =args.model_size
        self.clock_factor=args.clock_factor
        self.sample_mode=args.sample_mode
        self.score_mode=args.score_mode
        self.filter_less= args.filter_less
        self.sample_seed=args.sample_seed
        self.stop_signal=queue.Queue()

    def initiate_sampler_query(self, queue, numOfClients):
        # global logDir
        # Initiate the clientSampler
        if self.sampler_path is None:
            # if not args.load_model and args.sampler_path is None:
            client_sampler = clientSampler(self.sample_mode, self.score_mode, args=self.args,
                                           filter=self.filter_less,
                                           sample_seed=self.sample_seed)
        else:
            # load sampler
            self.sampler_path = os.path.join(self.model_path, 'aggregator/clientInfoFile')
            with open(self.sampler_path, 'rb') as loader:
                client_sampler = pickle.load(loader)
            logging.info("====Load sampler successfully\n")

        # load client profiles
        global_client_profile = {}
        if self.client_path and os.path.exists(self.client_path):
            print("global_client_profile is :",self.client_path)
            with open(self.client_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        collectedClients = 0
        initial_time = time.time()
        clientId = 1
        passed = False
        num_client_profile = max(1, len(global_client_profile))

        # In this simulation, we run data split on each worker, which amplifies the # of datasets
        # Waiting for the data information from clients, or timeout
        if self.enable_obs_client:
            roundDurationList = []
            roundDurationLocalList = []
            roundDurationCommList = []
            computationList = []
            communicationList = []
            # 对客户端完成度和时间两方面做限制
        # while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        while collectedClients < numOfClients :
            if not queue.empty():
                print("systemProfile", type(queue),queue.qsize())
                tmp_dict = queue.get()
                # we only need to go over once
                if not passed and self.sampler_path is None:
                    rank_src = list(tmp_dict.keys())[0]
                    distanceVec = tmp_dict[rank_src][0]
                    sizeVec = tmp_dict[rank_src][1]
                    for index, dis in enumerate(distanceVec):
                        # since the worker rankId starts from 1, we also configure the initial dataId as 1
                        clientId=rank_src
                        mapped_id = max(1, clientId % num_client_profile)
                        systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [
                            1.0, 1.0]
                        #print("client info is:",rank_src, clientId, dis, sizeVec[index],systemProfile)
                        client_sampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                        client_sampler.registerDuration(clientId,
                                                        batch_size=self.batch_size, upload_epoch=self.upload_epoch,
                                                        model_size=self.model_size * self.clock_factor)
                        if self.enable_obs_client:
                            roundDuration, roundDurationLocal, roundDurationComm = client_sampler.getCompletionTime(
                                clientId,
                                batch_size=self.batch_size, upload_epoch=self.upload_epoch,
                                model_size=self.model_size * self.clock_factor)

                        # roundDurationList.append(roundDuration)
                        # roundDurationLocalList.append(roundDurationLocal)
                        # roundDurationCommList.append(roundDurationComm)
                        # computationList.append(systemProfile[
                        #                            'computation'])
                        # communicationList.append(systemProfile[
                        #                              'communication'])

                        clientId += 1

                        #passed = True

                        collectedClients += 1

                        logging.info("====Info of all feasible clients {}".format(client_sampler.getDataInfo()))
                        # parser.add_argument('--enable_obs_client', type=bool, default=False, help="enable debug mode")
                        if self.enable_obs_client:
                            scipy.io.savemat(logDir + '/obs_client_time.mat', dict(roundDurationList=roundDurationList,
                                                                                   roundDurationLocalList=roundDurationLocalList,
                                                                                   roundDurationCommList=roundDurationCommList,
                                                                                   computationList=computationList,
                                                                                   communicationList=communicationList))
                        logging.info("====Save obs_client====")
                        self.stop_signal.put(1)


        return client_sampler
