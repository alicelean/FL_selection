
class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed,traces=None):
        self.hostId = hostId
        self.clientId = clientId
        # self.compute_speed = speed['computation']
        # self.bandwidth = speed['communication']
        self.compute_speed = speed[0]
        self.bandwidth = speed[1]
        self.distance = dis
        self.size = size
        self.score = dis
        self.traces = traces
        self.behavior_index = 0

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def isActive(self, cur_time):
        #self.traces = {
#     'finish_time': 100,  # 一个周期的总时长，例如100个时间单位
#     'active': [0, 40, 70],  # 客户端在每个周期的这些时间点开始活跃，例如在0、40、70时间点活跃
#     'inactive': [30, 60, 90]  # 客户端在每个周期的这些时间点变为非活跃，例如在30、60、90时间点停止活跃
# }

        if self.traces is None:
            return True
            
        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True

        return False

    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        roundDurationLocal=3.0 * batch_size * upload_epoch/float(self.compute_speed)

        roundDurationComm=model_size/float(self.bandwidth)

        roundDuration=roundDurationLocal+roundDurationComm
        
        return roundDuration,roundDurationLocal,roundDurationComm
        #return (3.0 * batch_size * upload_epoch*float(self.compute_speed)/1000. + model_size/float(self.bandwidth))