import copy

import torch
import torch.utils.data as data

B=5


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class FSDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=-1, eos_id=-1):
        '''作用：初始化数据集。
        参数：
        inputs：输入序列列表。
        targets：目标序列列表。
        train：是否是训练模式。
        sos_id：起始标志的 ID。
        eos_id：结束标志的 ID。'''
        super(FSDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = copy.deepcopy(inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
        # self.swap = swap
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = self.targets[index]
        encoder_input[encoder_input==-1] = self.eos_id

        if self.train:
            decoder_input = torch.cat((torch.tensor([self.sos_id]), encoder_input[:-1]))
            sample = {
                'encoder_input': encoder_input.long(),
                'encoder_target': encoder_target,
                'decoder_input': decoder_input.long(),
                'decoder_target': encoder_input.long(),
            }
        else:
            sample = {
                'encoder_input': encoder_input.long(),
                'decoder_target': encoder_input.long(),
            }
            if encoder_target is not None:
                sample['encoder_target'] = encoder_target
        return sample
    
    def __len__(self):
        return len(self.inputs)


class TripletFSDataset(torch.utils.data.Dataset):
    def __init__(self, h_inputs, t_inputs, targets=None, train=True, sos_id=-1, eos_id=-1):
        super().__init__()
        if targets is not None:
            assert len(h_inputs) == len(targets) and len(t_inputs) == len(targets)
        self.h_inputs = copy.deepcopy(h_inputs)
        self.t_input = copy.deepcopy(t_inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id

    def __getitem__(self, index):
        encoder_input = self.h_inputs[index]
        decoder_input = self.t_input[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = self.targets[index]
        encoder_input[encoder_input==-1] = self.eos_id
        decoder_input[decoder_input==-1] = self.eos_id

        if self.train:
            decoder_input = torch.cat((torch.tensor([self.sos_id]), decoder_input[:-1]))
            sample = {
                'encoder_input': encoder_input.long(),
                'encoder_target': encoder_target,
                'decoder_input': decoder_input.long(),
                'decoder_target': decoder_input.long(),
            }
        else:
            sample = {
                'encoder_input': encoder_input.long(),
                'decoder_target': decoder_input.long(),
            }
            if encoder_target is not None:
                sample['encoder_target'] = encoder_target
        return sample

    def __len__(self):
        return len(self.targets)


def pairwise_accuracy(la, lb):
    '''作用
    计算两个列表中值的 成对关系准确率。
    检查 la 和 lb 是否在成对关系上保持一致。
    参数
    la：第一个列表。
    lb：第二个列表。
    返回值
    成对关系的准确率（范围为 0 到 1）。'''
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    '''作用
    计算两个序列列表的 平均汉明距离，即每个序列之间的差异。
    参数
    la：第一个序列列表。
    lb：第二个序列列表。
    返回值
    平均汉明距离。'''
    N = len(la)
    assert N == len(lb)
  
    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c
  
    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


def generate_eval_points(eval_epochs, stand_alone_epoch, total_epochs):
    '''作用
    根据指定的评估间隔，生成联邦学习的评估轮次。
    参数
    eval_epochs：评估间隔（整数或列表）。
    stand_alone_epoch：独立训练的轮次。
    total_epochs：总训练轮次。
    返回值
    一个列表，包含所有的评估轮次。'''
    if isinstance(eval_epochs, list):
        return eval_epochs
    assert isinstance(eval_epochs, int)
    res = []
    eval_point = eval_epochs - stand_alone_epoch
    while eval_point + stand_alone_epoch <= total_epochs:
        res.append(eval_point)
        eval_point += eval_epochs
    return res
