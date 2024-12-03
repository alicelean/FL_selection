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

    def initiate_sampler_query(self,mode, queue, numOfClients):
        # global logDir
        # Initiate the clientSampler
        if self.sampler_path is None:
            print("----------------Initiating client_sampler query--------------------")
            # if not args.load_model and args.sampler_path is None:
            client_sampler = clientSampler(mode, self.score_mode, args=self.args,
                                           filter=self.filter_less,
                                           sample_seed=self.sample_seed)
        else:
            # load sampler
            self.sampler_path = os.path.join(self.model_path, 'aggregator/clientInfoFile')
            with open(self.sampler_path, 'rb') as loader:
                client_sampler = pickle.load(loader)
            logging.info("====Load sampler successfully\n")

        # load client profiles，客户端的属性
        global_client_profile = {}
        #print("self.client_path is ",self.client_path)

        if self.client_path and os.path.exists(self.client_path):
            #print("～～～～～～global_client_profile is :",self.client_path)
            with open(self.client_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        collectedClients = 0
        passed = False
        num_client_profile = max(1, len(global_client_profile))
        print(f"num_client_profile is {num_client_profile}")

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
            #加载客户端的数据信息
            if not queue.empty():
                #print("initiate client sampler systemProfile", type(queue),queue.qsize())
                tmp_dict = queue.get()
                # we only need to go over once
                if not passed and self.sampler_path is None:
                    rank_src = list(tmp_dict.keys())[0]
                    distanceVec = tmp_dict[rank_src][0]
                    sizeVec = tmp_dict[rank_src][1]
                    #print("rank_src,distanceVec,sizeVec ",rank_src,distanceVec,sizeVec )
                    for index, dis in enumerate(distanceVec):
                        # since the worker rankId starts from 1, we also configure the initial dataId as 1
                        clientId=rank_src
                        mapped_id = max(0, clientId % num_client_profile)
                        systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [
                            1.0, 1.0]
                        #print("client info is:",rank_src, mapped_id, dis, sizeVec[index],systemProfile)

                        client_sampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                        client_sampler.registerDuration(clientId,
                                                        batch_size=self.batch_size, upload_epoch=self.upload_epoch,
                                                        model_size=self.model_size * self.clock_factor)
                        if self.enable_obs_client:
                            roundDuration, roundDurationLocal, roundDurationComm = client_sampler.getCompletionTime(
                                clientId,
                                batch_size=self.batch_size, upload_epoch=self.upload_epoch,
                                model_size=self.model_size * self.clock_factor)

                            roundDurationList.append(roundDuration)
                            roundDurationLocalList.append(roundDurationLocal)
                            roundDurationCommList.append(roundDurationComm)
                            computationList.append(systemProfile[
                                                       'computation'])
                            communicationList.append(systemProfile[
                                                         'communication'])

                        clientId += 1

                        #passed = True

                        collectedClients += 1
                        # parser.add_argument('--enable_obs_client', type=bool, default=False, help="enable debug mode")
                        if self.enable_obs_client:
                            scipy.io.savemat(logDir + '/obs_client_time.mat', dict(roundDurationList=roundDurationList,
                                                                                   roundDurationLocalList=roundDurationLocalList,
                                                                                   roundDurationCommList=roundDurationCommList,
                                                                                   computationList=computationList,
                                                                                   communicationList=communicationList))




        return client_sampler
