from typing import List
import torch
import torch.nn.functional as F
import numpy


class Record(object):
    def __init__(self, operation, performance):
        if isinstance(operation, List):
            self.operation = numpy.array(operation)    
        elif isinstance(operation, torch.Tensor):
            self.operation = operation.numpy()
        else:
            assert isinstance(operation, numpy.ndarray)
            self.operation = operation
        self.performance = performance

    def get_permutated(self):
        pass

    def get_ordered(self):
        pass

    def repeat(self):
        pass

    def __eq__(self, other):
        '''该方法用于检查两个 Record 对象是否相等。
如果 other 不是 Record 类的实例，返回 False。
否则，比较 Record 对象的哈希值，如果两个对象的哈希值相同，则认为它们是相等的。'''
        if not isinstance(other, Record):
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        '''
        该方法返回 Record 对象的哈希值。
通过将 operation（选择操作）转换为字符串后调用 __hash__ 方法获取哈希值。
这样可以确保 Record 对象可以在集合（如 set）中进行存储和去重
        '''
        return str(self.operation).__hash__()


class SelectionRecord(Record):
    '''SelectionRecord 类扩展了 Record 类，提供了用于生成选择记录的变种和重复的功能，常用于数据增强。
get_permutated 方法通过随机排列设备选择来生成多个变种，以增加数据的多样性。
repeat 方法生成多个相同的选择记录，并在需要时进行填充。'''
    def __init__(self, operation, performance):    
        super().__init__(operation, performance)
        self.max_size = operation.shape[0]   

    def _get_ordered(self):
        '''该方法用于返回排序后的设备选择操作和其对应的性能。
torch.arange(0, self.max_size) 创建一个从 0 到 max_size-1 的张量，表示设备池中每个设备的索引。
然后通过 self.operation == 1 筛选出被选中的设备索引。
返回的是被选中设备的索引 indice_select 和一个包含性能值的 torch.FloatTensor，它是一个标量值的张量（即 self.performance）。'''
        indice_select = torch.arange(0, self.max_size)[self.operation == 1]    
        return indice_select, torch.FloatTensor([self.performance])   

    def get_permutated(self, num=25, padding=True, padding_value=-1):
        '''该方法用于生成多个排列（permutations）的选择记录，通常用于数据增强'''
        #获取排序后的设备选择索引 ordered 以及对应的性能值 performance
        #返回 shuffled_indices（多个排列后的设备选择）和 label（对应的性能值标签）
        ordered, performance = self._get_ordered()
        #获取被选中设备的数量
        size = ordered.shape[0]    
        shuffled_indices = torch.empty(num + 1, size)
        shuffled_indices[0] = ordered    
        label = performance.unsqueeze(0).repeat(num + 1, 1)
        for i in range(num):    
            shuffled_indices[i + 1] = ordered[torch.randperm(size)]
        if padding and size < self.max_size:    
            shuffled_indices = F.pad(shuffled_indices, (0, (self.max_size - size)), 'constant', padding_value)
        return shuffled_indices, label

    def repeat(self, num=25, padding=True, padding_value=-1):
        #也用于生成多个相同的选择记录，主要是为了数据增强和增加样本的多样性
        ordered, performance = self._get_ordered()
        size = ordered.shape[0]
        label = performance.unsqueeze(0).repeat(num + 1, 1)
        indices = ordered.unsqueeze(0).repeat(num + 1, 1)
        if padding and size < self.max_size:
            indices = F.pad(indices, (0, (self.max_size - size)), 'constant', padding_value)
        return indices, label


class RecordList(object):
    def __init__(self):
        self.r_list = set()

    def append(self, op, val):    
        self.r_list.add(SelectionRecord(op, val))

    def __len__(self):
        return len(self.r_list)

    def generate(self, num=25, padding=True, padding_value=-1):
        '''生成多个排列后的选择记录及其对应的标签
num：每个记录生成的排列数，默认为 25。
padding：是否对选择记录进行填充，默认为 True。
padding_value：填充时使用的值，默认为 -1。'''
        #results 用来存储生成的选择记录
        results = []
        labels = []
        for record in self.r_list:
            result, label = record.get_permutated(num, padding, padding_value)
            results.append(result)
            labels.append(label)

        return torch.cat(results, 0), torch.cat(labels, 0)
