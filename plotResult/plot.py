import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#---------client selectiion----------------------
def plot_multiline(path,x,ylist,labellist,colist,title,value,dataset,alpha,xlabel):
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
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+', '*']
    markerslist = markers[:len(ylist)]
    lnlist=[]
    num_markers = min(len(x), len(markers))
   # ['FedALA', 'FedAAW (Ours)', 'FedAvg', 'FedProx', 'MOON', 'perFedAvg', 'SCAFFOLD']

    points=[]
    print(f"----------------all data is:len(ylist) is {len(ylist)}")
    for i in range(len(ylist)):
        if len(x)!=len(ylist[i]):
            print("ERROR x and y length not equal")
        ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='-',marker=markerslist[i], markevery=10)
        # if i ==0:
        #     ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='--')
        # elif i==1:
        #     ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-')
        # else:
        #     ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-.')
        lnlist.append(ln)
        for j in range(len(ylist[i])):
            if ylist[i][j] > 0.8:
                points.append((j, ylist[i][j]))
                break
    print("label is ",labellist)

    if len(x) > 10:
        legend_loc = 'upper right'
    else:
        legend_loc = 'best'

    plt.legend(handles=lnlist, labels=labellist, loc=legend_loc ,bbox_to_anchor=(1.05, 1))  # 将图例放置于右上角或最佳位置
    plt.xlabel(dataset+"(alpha="+alpha+")")
    plt.xlabel(xlabel)
    plt.ylabel(value)
    plt.xlim(-5, 135)
    plt.xticks(np.arange(-5, 140, 5))

    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    print('save',path)
    plt.savefig(path, format='png', bbox_inches='tight')
    plt.show()

def plot_select(fpath,picpath,value,dataset,alpha,xlabel,length=-1):
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
                              'Std Test AUC', 'select_mode'])
    print("read data from :",fpath)
    methods= data['select_mode'].unique()
    #['method' 'FedJS' 'FedAvg' 'FedALA_AAW' 'FedAAW' 'FedALA' 'SCAFFOLD''FedProx' 'MOON']
    print("method is",methods,len(methods))
    # 按照method分组
    title=value+" trend of different algorithms"
    labellist = []
    ylist = []
    points=[]
    #last accuracy
    repoints=[]
    # 按照method分组
    grouped_data = data.groupby('select_mode')
    for name, group in grouped_data:
        print(group[value].dtypes)
        # 转换数据类型，如果不是数值类型的话
        group[value] = pd.to_numeric(group[value], errors='coerce')

        smoothed_loss = group[value].rolling(window=5, min_periods=1).mean()

        print("add",name)
        print("value is",len(smoothed_loss))
        if length==-1:
            acc=smoothed_loss.tolist()
        else:
            acc =smoothed_loss.tolist()[:length]
        for j in range(len(acc)):
            if acc[j] > 0.8:
                print(j, acc[j],name)
                points.append((j, acc[j],name))
                break
        # print(type(acc[0]))
        if len(acc)==length:
            labellist.append(name)
            ylist.append(acc)
        # repoints.append([str(round(acc[-1],4)),name])

    # print("TTA:", points)
    #print("accracy", repoints)
    x = [i for i in range(len(ylist[0]))]

    # colist=['red','yellow','blue','green','black','pink','orange']



    # 获取预定义的颜色循环
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色循环，共有10种颜色
    # 定义要绘制的曲线数量
    num_curves = 15
    # 生成一组颜色

    colist = [cmap(i) for i in range(num_curves)][:len(ylist)]

    print("col",len(ylist))

    #colist = ['red',  'blue', 'green', 'black', 'orange']
    print(f"must equal len(ylist)={len(ylist)}, len(colist)={ len(colist)},labellist={len(labellist)}")
    plot_multiline(picpath,x,ylist,labellist,colist,title,value,dataset,alpha,xlabel)



programpath="/Users/alice/Desktop/python/FL_selection/"
dataset="mnist"
xlabel=" Commnication Round on cifar10 "
join_ratio='0.5'
alpha='0.5'
num_clients=20
alpha="0.5"
method='FedAvg'
group=50
folder_path = programpath + "/res/" + method + "/"
fpath = folder_path + dataset + "_acc.csv"
#fpath=programpath + "/res/ " + dataset + "_allacc_" + str(join_ratio) + "_" + str(num_clients) + "_" + str(alpha) + ".csv"
picpath=folder_path + dataset + "_acc.png"
plot_select(fpath,picpath,'Accuracy',dataset,alpha,xlabel,group)


