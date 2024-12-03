import os
import torch
from selection.fedGCS.record import RecordList


class Evaluator(object):
    def __init__(self, task_type=None, dataset=None):
        self.records = RecordList()
        self.base_path=None

    def __len__(self):    
        return len(self.records)
  
    
    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

    
    def _flush_history(self, choices, performances, is_permuted, num, padding):
        '''该方法用于将选择和性能记录保存到磁盘，并根据不同的标志设置来决定保存的文件名,'''
        #增强
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        #填充
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/choice.pt')
        torch.save(performances, f'{base_path}/history/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{self.base_path}/history'):
            os.mkdir(f'{self.base_path}/history')

 
    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:    
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        #generate 生成选择和性能数据
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)    
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):
        '''遍历 self.records.r_list 中的每个记录，调用 record.get_permutated()
        获取增强后的数据（result）和标签（label），并将其添加到 results 和 labels 列表中。
使用 torch.cat() 将 results 和 labels 列表中的所有张量拼接成一个大张量，并返回。
'''
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)



    def report_performance(self, choice, performances, store=True, rp=True, flag=''):   
        '''
        如果 store 为 True，则调用 _store_history() 方法将选择和性能存储到记录中。
        其他参数（rp 和 flag）可能在报告过程中用于其他功能，但目前未详细解释。
        '''
        if store:
            self._store_history(choice, performances)





