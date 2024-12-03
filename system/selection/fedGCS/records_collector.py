import os
import sys

# add
sys.path.append("./IJCAI-AutoS")

print(sys.path)
from selection.fedGCS.device_env import Evaluator
import pickle
import torch
import warnings
import random

warnings.filterwarnings('ignore')

def convert_to_binary_list(selected_indices, list_size):
    '''将一个整数索引列表转换为二进制表示的列表。

输入：索引列表 selected_indices 和目标长度 list_size。
输出：长度为 list_size 的 PyTorch 浮点型张量，值为 0 或 1
eg:selected_indices = [0, 2, 4]
result: tensor([1., 0., 1., 0., 1.])'''
    binary_list = [0] * list_size
    for index in selected_indices:
        if 0 <= index < list_size:
            binary_list[index] = 1
    return torch.FloatTensor(binary_list)

def gen_device_selection(device_eval_,data,num_clients):
    '''
       模拟 300 次设备选择与性能评估，调用 Evaluator 对象的 _store_history 方法存储记录。
       示例： 如果第 1 次迭代中选中设备 [1, 3]，性能值为 100.5：
        selected_numbers: [0, 1, 0, 1, 0, ...]
        performances: 100.5
    '''
    for i in data:
        #随机生成设备选择数量
        number_of_selections = len(i[0])
        #从设备编号 1~200 中随机选择指定数量的设备
        selected_numbers = i[0]
        #将设备选择编码为二进制列表
        selected_numbers = convert_to_binary_list(selected_numbers, num_clients)
        #随机生成一个性能值
        performances = i[1]
        #调用 _store_history 方法将选择与性能存储到 device_eval_ 对象中
        device_eval_._store_history(selected_numbers, performances)



def process(data,num_clients,basepath):

    #创建 Evaluator 对象
    device_eval = Evaluator()
    #调用 gen_device_selection 生成设备选择历史
    gen_device_selection(device_eval,data,num_clients)
    #将 Evaluator 对象序列化并保存到文件
    file_path = f"{basepath}/history/device_env.pkl"
    print("process file_path", file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f'{basepath}/history/device_env.pkl', 'wb') as f:
        pickle.dump(device_eval, f)

# process()
