import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import pearsonr
# import re
#
# # 创建数据框
# data = pd.read_csv('/Users/alice/Desktop/python/PFL//res/FedAvg/Cifar100_clients_error.csv')
#
# # 提取distance列的数据
# distance = data['distance']
#
# # 解析error列的数据并将其转换为NumPy数组
# error = np.array([np.array(eval(x)) for x in data['error']])
#
# # 绘制散点图
# plt.scatter(distance, error)
# plt.xlabel('Distance')
# plt.ylabel('Error')
# plt.title('Scatter Plot of Distance vs Error')
# plt.show()
#
# # 计算相关性
# correlation = [pearsonr(distance, e)[0] for e in error]
# print(f'Pearson Correlation Coefficients: {correlation}')
#
#


import matplotlib.pyplot as plt

# 读取数据
data = []
data = pd.read_csv('/Users/alice/Desktop/python/PFL//res/FedAvg/Cifar100_error.csv')
print(data['error'])


# 绘制曲线
plt.figure(figsize=(10, 5))
plt.plot(data['error'].tolist(), marker='o', linestyle='-')
plt.xlabel('Group')
plt.ylabel('Error')
plt.title('Error Curve')
plt.grid(True)
plt.show()
