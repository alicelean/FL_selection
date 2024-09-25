import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vars import Programpath
# 读取txt文件
fpath=Programpath+'res/mnist/ratio_0.5/mnist_acc.csv'

def plot_one_line(x,y,label,title,picpath):
    '''
    :param x: x轴标签列表
    :param y: x轴对应的数据y1
    :param label: y曲线的名称
    picpath:图片存放位置
    title:图表的名称
    :return:
    一条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    ln1, = plt.plot(x, y, color='red', linewidth=2.0, linestyle='--')
    #plt.xticks(np.arange(0,1,0.2))
    plt.legend(handles=[ln1], labels=[label])
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    print(picpath)
    plt.savefig(picpath, format='png')
    plt.show()
def plot_listline(path,x,ylist,labellist,colist,title,value,dataset,alpha):
    '''
    :param x: x轴标签列表
    :param ylist: x轴对应的多条数据y
    :param labellist: x轴对应的多条数据y的曲线的名称
    :param title:图表的名称
    :return:
    多条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    #plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    lnlist=[]
    markers = ['o', 's', '^', 'd', 'v']  # 可自定义不同的标记样式
    num_markers = min(len(x), len(markers))
    points=[]
    for i in range(len(ylist)):
        if len(x)!=len(ylist[i]):
            print("ERROR x and y length not equal")
        if i ==0:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='--')
        else:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-.')
        lnlist.append(ln)
        for j in range(len(ylist[i])):
            if ylist[i][j] > 0.7:
                points.append((j, ylist[i][j]))
                break

    print(points)

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    # 创建散点图
    plt.scatter(x, y,color='black')


    # acc1=[0.9 for i in range(len(x))]
    # ln, = plt.plot(x, acc1, color='black', linewidth=1.0, linestyle='--')
    # lnlist.append(ln)
    #
    # acc2 = [0.8 for i in range(len(x))]
    # ln, = plt.plot(x, acc2, color='black', linewidth=1.0, linestyle='--')
    # lnlist.append(ln)

    # # 标记与ln1和ln2交点的横坐标位置
    # for i in range(len(x)):
    #     if ylist[0][i] == acc1[i]:
    #         plt.scatter(x[i], acc1[i], color='red')
    #     if ylist[0][i] == acc2[i]:
    #         plt.scatter(x[i], acc2[i], color='blue')
    # plt.yticks(np.arange(0.75,0.95,0.05))
    # plt.yticks(xyran)
    print("label is ",labellist)

    #labellist=['FedAvg','JSND','FedSGD','Center']
    plt.legend(handles=lnlist, labels=labellist)
    plt.xlabel("Rounds "+dataset+"(alpha="+alpha+")")
    plt.ylabel(value)
    #plt.ylim(0.1,0.88,0.01)
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    #plt.savefig(path,format='svg')
    print('save',path)
    plt.savefig(path, format='eps')
    plt.show()











def plot(fpath,picpath,value,dataset,alpha,length=-1):
    '''
    绘制不同列的曲线图
    Args:
        value: 列的名称
        length: 制定列的长度
        names:所有列名

    Returns:

    '''
    # 读取txt文件，指定列名
    data = pd.read_csv(fpath, delimiter=',',
                       names=['1', 'case', 'method', 'group', 'Loss', 'Accuracy', 'AUC', 'Std Test Accuracy',
                              'Std Test AUC', 'join_ratio'])
    methods= data['method'].unique()
    #['method' 'FedJS' 'FedAvg' 'FedALA_AAW' 'FedAAW' 'FedALA' 'SCAFFOLD''FedProx' 'MOON']
    print(methods,len(methods))
    # 按照method分组
    title=value+" trend of different algorithms"
    labellist = []
    ylist = []
    points=[]
    #last accuracy
    repoints=[]
    # 按照method分组
    grouped_data = data.groupby('method')
    for name, group in grouped_data:
        #all
        if name != 'method' and name != 'SCAFFOLD':

        #fix
        #if name != 'method' and name != 'FedALA_AAW' and name != 'FedALA' and name != 'SCAFFOLD':
        #if name != 'method' and name != 'FedAAW' and name != 'FedJS' and name!='SCAFFOLD':
            smoothed_loss = group[value].rolling(window=5, min_periods=1).mean()
            labellist.append(name)
            print("add",name)
            acc=None
            if length==-1:
                acc=smoothed_loss.tolist()
            else:
                acc =smoothed_loss.tolist()[:length]
            for j in range(len(acc)):
                if acc[j] > 0.5:
                    points.append((j, acc[j],name))
                    break
            ylist.append(acc)
            repoints.append((round(acc[-1],4),name))
    x = [i for i in range(len(ylist[0]))]
    print(points)
    print("accracy",repoints)
    # colist=['red','yellow','blue','green','black','pink','orange']



    # 获取预定义的颜色循环
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色循环，共有10种颜色
    # 定义要绘制的曲线数量
    num_curves = 7
    # 生成一组颜色
    colist = [cmap(i) for i in range(num_curves)]

    #colist = ['red',  'blue', 'green', 'black', 'orange']
    print(f"must equal len(ylist)={len(ylist)}, len(colist)={ len(colist)},labellist={len(labellist)}")
    plot_listline(picpath,x,ylist,labellist,colist,title,value,dataset,alpha)






# plot('Loss')
# plot('Loss',50)
# plot('Accuracy')
# plot(fpath,'Accuracy',50)
# value='Accuracy'
# length=50
# fpath=Programpath+'res/mnist/fedda/mnist_acc.csv'
# picpath=Programpath+'res/mnist/fedda/'+'mnist_'+value+str(length)+'.eps'



# fpath=Programpath+'res/ mnist_allacc_0.05.csv'
# picpath=Programpath+'res/mnist_'+'allacc_'+'0.05.eps'
# dataset='mnist'
# group=100
# alpha='0.05'
# fpath=Programpath+'res/0.05/ '+dataset+'_allacc_'+alpha+'.csv'
# picpath=Programpath+'res/0.05/'+dataset+'_'+'fixallacc_'+alpha+'.eps'
# plot(fpath,picpath,'Accuracy',dataset,alpha,group)
#




import matplotlib.pyplot as plt

# 读取数据



def plot_multlines(path,x,ylist,labellist,colist):
    '''
    :param x: x轴标签列表
    :param ylist: x轴对应的多条数据y
    :param labellist: x轴对应的多条数据y的曲线的名称
    :param title:图表的名称
    :return:
    多条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    #plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    lnlist=[]
    markers = ['o', 's', '^', 'd', 'v']  # 可自定义不同的标记样式
    for i in range(len(ylist)):
        if len(x)!=len(ylist[i]):
            print("ERROR x and y length not equal")
        if i ==0:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='--')
        else:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-.')
        lnlist.append(ln)



    print("label is ",labellist)
    plt.legend(handles=lnlist, labels=labellist)
    plt.xlabel(" Commnunication Rounds ")
    plt.ylabel(" aggregation error ")
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    print('save',path)
    plt.savefig(path+'error.svg', format='svg')
    plt.show()

def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = data[start:end]
        smoothed_value = sum(window) / len(window)
        smoothed_data.append(smoothed_value)
    return smoothed_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
def plot_big_small(x,y1,y2=None,y3=None,y4=None,y5=None,y6=None,y7=None):
    # 获取预定义的颜色循环
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色循环，共有10种颜色
    # 定义要绘制的曲线数量
    num_curves = 7
    # 生成一组颜色
    colist = [cmap(i) for i in range(num_curves)]

    # 创建大图
    fig, ax = plt.subplots(figsize=(8, 6))
    # 绘制第一条曲线
    ax.plot(x, y1, label='FedAvg (ratio=0.5)', color='g',linestyle='-', marker='^', markevery=5)
    #ax.plot(x, y1, label='FedAvg', color=colist[2],linestyle='-.')

    # 绘制第二条曲线
    if y2 is not None:
        ax.plot(x, y2, label='FedProx', color=colist[4],linestyle='-.')
        #ax.plot(x, y2, label='FedAvg (ratio=0.1)', color='g',linestyle='--', marker='s', markevery=5)
    # 绘制第二条曲线
    if y3 is not None:
        #ax.plot(x, y3, label='FedAvg (ratio=0.3)', color='g', linestyle='--', marker='o', markevery=5)
        ax.plot(x, y3, label='FedALA', color=colist[0],linestyle='--')
        #ax.plot(x, y3, label='FedAvg (ratio=0.8)', color='g')

    # 绘制第二条曲线
    if y4 is not None:
        ax.plot(x, y4, label='FedAAW (ours)', color=colist[1])
    # 绘制第二条曲线
    if y5 is not None:
        #ax.plot(x, y5, label='Center', color='pink')
        ax.plot(x, y5, label='MOON', color=colist[5],linestyle='-.')
    if y6 is not None:
        # ax.plot(x, y5, label='Center', color='pink')
        ax.plot(x, y6, label='SCAFFOLD', color=colist[6],linestyle='-.')
    if y7 is not None:
        # ax.plot(x, y5, label='Center', color='pink')
        ax.plot(x, y7, label='PerAvg', color=colist[3],linestyle='-.')
    # 添加圆圈到大图的尾部
    # 添加圆圈到大图的尾部
    tail_circle = plt.Circle((x[-1], y1[-1]), 0.1, color='b', fill=False, linestyle='dotted', zorder=10)
    ax.add_patch(tail_circle)
    # 创建小图
    # subax = plt.axes([0.5, 0.5, 0.25, 0.25])

    # 在小图上展示曲线尾部
    # subax.plot(x[-25:], y1[-25:], color='b')
    # if y2 is not None:
    #     subax.plot(x[-25:], y2[-25:], color='r')
    # if y3 is not None:
    #     subax.plot(x[-25:], y3[-25:], color='g')
    # if y4 is not None:
    #     subax.plot(x[-25:], y4[-25:], color='y')
    # if y5 is not None:
    #     subax.plot(x[-25:], y5[-25:], color='g')
    # 添加连接线
    # con = ConnectionPatch(xyA=(40, 1), coordsA=ax.transAxes, xyB=(0, 1), coordsB=subax.transData,
    #                       arrowstyle="->", shrinkA=5, shrinkB=5, color="black", linewidth=2)
    # fig.add_artist(con)

    # 设置图例
    ax.legend(loc='upper right')
    #subax.legend(['FedAvg', 'FedALA','FedALA_AAW'])

    # 设置标题和标签
    #ax.set_title('Aggregation Error Trend with dfferent Method')
    ax.set_xlabel('Comunication Round')
    ax.set_ylabel('Agregation Error')
    ax.legend(loc='upper left')
    #plt.savefig('/Users/alice/Desktop/python/PFL//res/FedAvg/error_2.png')
    plt.savefig('/Users/alice/Desktop/python/PFL//res/FedAvg/allerror.png')
    #plt.savefig('/Users/alice/Desktop/python/PFL//res/FedAvg/allerror_2.png')

    # 显示图形
    plt.show()

csv=["","2","3"]
#rate=0.5
csv=["1"]
x=[]
ylist1=[]
ylist2=[]
ylist3=[]
ylist4=[]
ylist5=[]
ylist6=[]
ylist7=[]
ylist8=[]
ylist9=[]
ylist10=[]
ylist11=[]
keyname='Loss'
#keyname='error'
# colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC"]
#keyname='var',FedAvg,FedAAW,FedALA,FedALA_AAW,Center
for i in csv:
    path="/Users/alice/Desktop/python/PFL//res/FedAvg/"
    data = pd.read_csv(path+'/Cifar100_error'+i+'.csv')
    #data = pd.read_csv(path + '/mnist_acc_' + i + '.csv')
    data = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    #data = pd.read_csv(path + '/mnist_error' + i + '.csv')
    #print(len(data['error']))
    y=data[keyname].tolist()[1:]
    ylist1.append(y)

    data = pd.read_csv(path + '/Cifar100_acc_0.1.csv')
    ylist6.append(data[keyname].tolist()[1:])
    data = pd.read_csv(path + '/Cifar100_acc_0.3.csv')
    ylist7.append(data[keyname].tolist()[1:])
    data = pd.read_csv(path + '/Cifar100_acc_0.3.csv')
    ylist11.append(data[keyname].tolist()[1:])

    path = "/Users/alice/Desktop/python/PFL//res/Fedprox/"
    #data2 = pd.read_csv(path + '/Cifar100_error' + i + '.csv')
    #data2 = pd.read_csv(path + '/mnist_acc_' + i + '.csv')
    data2 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    y2 = data2[keyname].tolist()[1:]
    ylist2.append(y2)
    path = "/Users/alice/Desktop/python/PFL//res/FedALA/"
    data3 = pd.read_csv(path + '/Cifar100_error' + i + '.csv')
   # data3 = pd.read_csv(path + '/mnist_acc_' + i + '.csv')
    data3 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    y3 = data3[keyname].tolist()[1:]
    ylist3.append(y3)
    path = "/Users/alice/Desktop/python/PFL//res/FedALA_AAW/"
    data4 = pd.read_csv(path + '/Cifar100_error' + i + '.csv')
    #data4 = pd.read_csv(path + '/mnist_acc_' + i + '.csv')
    data4 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    # data = pd.read_csv(path + '/mnist_error' + i + '.csv')
    # print(len(data['error']))
    y4 = data4[keyname].tolist()[1:]
    ylist4.append(y4)

    path = "/Users/alice/Desktop/python/PFL//res/Center/"
    #data5 = pd.read_csv(path + '/Cifar100_error' + i + '.csv')
    # data4 = pd.read_csv(path + '/mnist_acc_' + i + '.csv')
    data5 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    # data = pd.read_csv(path + '/mnist_error' + i + '.csv')
    # print(len(data['error']))
    y5= data5[keyname].tolist()[1:]
    ylist5.append(y5)

    path = "/Users/alice/Desktop/python/PFL//res/MOON/"
    data2 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    y=data2[keyname].tolist()[1:]
    ylist8.append([float(j) for j in y])


    path = "/Users/alice/Desktop/python/PFL//res/SCAFFOLD/"
    data2 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    ylist9.append(data2[keyname].tolist()[1:])

    path = "/Users/alice/Desktop/python/PFL//res/PerAvg/"
    data2 = pd.read_csv(path + '/Cifar100_acc_' + i + '.csv')
    ylist10.append(data2[keyname].tolist()[1:])

y1combined = zip(*ylist1)

# 计算每组元素的平均值
y1 = [sum(values) / len(values) for values in y1combined]

x=[i+1 for i in range(len(y1))]
print(y1)
plot_one_line(x,y1,"FedAvg","",path+'errorg11.png')

y2combined = zip(*ylist2)
y2 = [sum(values) / len(values) for values in y2combined]
y3combined = zip(*ylist3)
y3 = [sum(values) / len(values) for values in y3combined]
y4combined = zip(*ylist4)
y4 = [sum(values) / len(values) for values in y4combined]
y5combined = zip(*ylist5)
y5 = [sum(values) / len(values) for values in y5combined]
print(len(y1),len(y2))
y6combined = zip(*ylist6)
y6 = [sum(values) / len(values) for values in y6combined]

y7 = [sum(values) / len(values) for values in zip(*ylist7)]

y8 = [sum(values) / len(values) for values in zip(*ylist8)]
y9 = [sum(values) / len(values) for values in zip(*ylist9)]
y10 = [sum(values) / len(values) for values in zip(*ylist10)]

ylist=[y1,y2,y3,y4,y5]
labellist=['FedAvg','FedALA_AAW','FedALA','FedAAW','Center']
colist=['red','green','black','blue','purple']
# plot_multlines(path,x,ylist,labellist,colist)
x=[i for i in range(len(y1))]
print(len(y1),len(y2),len(y3),len(x),len(y8),len(y9),len(y10))
#plot_big_small(x,y1,y2,y3,y4,y5)


for i in range(len(y5)):
    y2[i] = y2[i] /float(y5[i])
    y3[i] = y3[i] /float(y5[i])
    y4[i] = y4[i] / float(y5[i])
    y1[i] = y1[i]/float(y5[i])
    y6[i] = y6[i] / float(y5[i])
    y7[i] = y7[i] / float(y5[i])
    y8[i] = y8[i] / float(y5[i])
    y9[i] = y9[i] / float(y5[i])
    y10[i] = y10[i] / float(y5[i])

    # y1[i] = y1[i] - float(y5[i])
    # y6[i] = y6[i] -float(y5[i])
    # y7[i] = y7[i] -float(y5[i])
    # y2[i] = y2[i] -float(y5[i])
    # y3[i] = y3[i]-float(y5[i])
    # y4[i] = y4[i] - float(y5[i])
    # y8[i] = y8[i] - float(y5[i])
    # y9[i] = y9[i] -float(y5[i])
    # y10[i] = y10[i] -float(y5[i])
#plot_big_small(x, y1,y6,y7)
#plot_big_small(x, y1,y2,y3,y4,y8,y9,y10)





from PIL import Image
import os

def convert_png_to_eps(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的PNG文件
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for png_file in png_files:
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_folder, png_file)
        output_file = os.path.splitext(png_file)[0] + '.eps'
        output_path = os.path.join(output_folder, output_file)

        # 打开PNG文件并将其转换为RGB模式
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            # 保存为EPS格式
            img.save(output_path, 'EPS')


if __name__ == "__main__":
    # 指定输入和输出文件夹
    input_folder = "/Users/alice/Desktop/python/PFL/res/picture/" # 替换为实际的输入文件夹路径
    output_folder = "/Users/alice/Desktop/python/PFL/res/picture2/" # 替换为实际的输出文件夹路径
    convert_png_to_eps(input_folder, output_folder)













