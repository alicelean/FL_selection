import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




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
    data = pd.read_csv(fpath,header=0)
    print("read data from :",fpath,data.head(5))
    # 查看列名
    print(data.columns)
    # 清理列名中的空格
    data.columns = data.columns.str.strip()

    # 检查数据类型
    for i in data.columns:
        print(i," data",data[i].tolist())

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
        print("sss",group[value].dtypes)
        # 转换数据类型，如果不是数值类型的话
        group[value] = pd.to_numeric(group[value], errors='coerce')
        smoothed_loss = group[value].rolling(window=1, min_periods=1).mean()

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
    print("accracy", ylist)
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
def normalize_list(data):
    #print("data is ",data)
    data=[float(i) for i in data]
    min_val = min(data)  # 找到最小值
    max_val = max(data)  # 找到最大值
    if max_val - min_val == 0:  # 避免分母为零
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
def plot_multi(fpath,picpath,valuelist,dataset,alpha,xlabel,length=-1):
    '''
    绘制不同列的曲线图
    Args:
        value: 列的名称
        length: 制定列的长度
        names:所有列名

    Returns:

    '''
    # 读取txt文件，指定列名
    data = pd.read_csv(fpath,header=0)
    #print("read data from :",fpath,data.head(5))
    # 查看列名
    print("data.columns is : ",data.columns)
    # 清理列名中的空格
    data.columns = data.columns.str.strip()
    # 检查数据类型
    # for i in data.columns:
    #     print(i," data",data[i].tolist())

    # 按照method分组
    title=" trend of different algorithms"
    labellist = []
    ylist = []
    points=[]
    for i in valuelist:
        #print(i,data[i].tolist())
        ll=[float(j) for j in data[i].tolist()]
        #print("ll",i ,ll)
        # newll=[]
        # if i =='Accurancy':
        #     for j in range(len(ll)-1):
        #         print("j",j,len(ll))
        #         newll.append(ll[j+1]-ll[j])
        #     ll=newll


        #ylist.append(normalize_list(data[i].tolist()[:length]))
        ylist.append(normalize_list(ll)[:length])
        labellist.append(i)
    correlation = np.corrcoef(ylist[0], ylist[1])[0, 1]
    from scipy.stats import spearmanr
    print("两列数据的相关性（皮尔逊相关系数）：", correlation)
    correlation, p_value = spearmanr(ylist[0], ylist[1])

    print("两列数据的相关性（斯皮尔曼相关系数）：", correlation)
    print("accracy", ylist)
    x = [i for i in range(len(ylist[0]))]

    # colist=['red','yellow','blue','green','black','pink','orange']
    # necolist=[]
    # for i in range(len(ylist)):
    #     necolist.append(colist[i])
    # 获取预定义的颜色循环
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色循环，共有10种颜色
    # 定义要绘制的曲线数量
    num_curves = 15
    # 生成一组颜色
    colist = [cmap(i) for i in range(num_curves)][:len(ylist)]
    print("col",len(ylist))
    #colist = ['red',  'blue', 'green', 'black', 'orange']
    print(f"must equal len(ylist)={len(ylist)}, len(colist)={ len(colist)},labellist={len(labellist)}")
    plot_multiline(picpath,x,ylist,labellist,colist,title,"value",dataset,alpha,xlabel)



def plotoneline(x,y):
    # 创建绘图
    plt.figure(figsize=(8, 5))

    # 绘制曲线
    plt.plot(x, y, marker='o', color='blue', label='y = x^2')

    # 设置轴标签和标题
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of y = x^2')

    # 添加网格
    plt.grid()

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()
def normalize(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
    return normalized_data
def plot_distance(path):
    # 读取CSV文件
    df = pd.read_csv(path)
    # 选择特定的method，例如 'FedPyramid'
    selected_method = 'FedPyramid'
    df_filtered = df[df['method'] == selected_method]
    df_filtered2 = df[df['method'] == 'FedVOI']
    print(df_filtered.head())
    # 排序数据以便于绘制曲线

    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制 Distance 曲线
   # plt.plot(x, df_filtered[''], marker='o', label='Distance', color='blue')

    # 绘制 Accuracy 曲线
    x1 = [float(i) for i in df_filtered2['distance'].tolist()]
    # 进行归一化
    # x = normalize(x)
    y1 = [float(i) for i in df_filtered2['Accurancy'].tolist()]


    x=[float(i) for i in df_filtered['distance'].tolist()]
    # 进行归一化
    # x = normalize(x)
    y = [float(i) for i in df_filtered['Accurancy'].tolist()]
    print(f"x is {x},y is {y}")
    # 计算皮尔逊相关系数
    correlation_coefficient, p_value = pearsonr(x1, y1)
    from scipy.stats import spearmanr, kendalltau

    spearman_corr, spearman_p = spearmanr(x1, y1)
    kendall_corr, kendall_p = kendalltau(x1, y1)

    print(f"Spearman 相关系数: {spearman_corr}, p值: {spearman_p}")
    print(f"Kendall 相关系数: {kendall_corr}, p值: {kendall_p}")
    # 输出结果
    print(f"皮尔逊相关系数: {correlation_coefficient}")
    print(f"p值: {p_value}")
    # plotoneline(x,y_accuracy)

    #继续分析

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # 将数据转换为numpy数组
    x_np = np.array(x).reshape(-1, 1)
    y_np = np.array(y).astype(float)

    # 多项式特征
    poly = PolynomialFeatures(degree=2)  # 选择合适的多项式度数
    x_poly = poly.fit_transform(x_np)

    # 线性回归模型
    model = LinearRegression()
    model.fit(x_poly, y_np)

    # 绘制结果
    plt.scatter(x1, y1, color='red', alpha=0.6)
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.plot(x, model.predict(x_poly), color='red')
    plt.title('Polynomial Regression Fit')
    plt.xlabel('Distance')
    plt.ylabel('Accuracy')
    plt.show()
def find(path):
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    # 读取数据
    data = pd.read_csv(path)  # 假设数据已保存
    distance = data['distance']
    accuracy = data['Accurancy']
    distance = pd.to_numeric(distance, errors='coerce')
    accuracy = pd.to_numeric(accuracy, errors='coerce')

    # 绘制散点图
    print(f"distance is {distance.tolist()},accuracy is {accuracy.tolist()}")
    # plt.scatter(distance, accuracy, alpha=0.6, label='Data Points')
    # plt.xlabel('Distance')
    # plt.ylabel('Accuracy')
    # plt.title('Distance vs Accuracy')
    #
    # # 线性回归分析
    # slope, intercept, r_value, p_value, std_err = linregress(distance, accuracy)
    # plt.plot(distance, intercept + slope * distance, 'r', label=f'Linear Fit: R²={r_value ** 2:.3f}')
    # plt.legend()
    # plt.show()


programpath="/Users/alice/Desktop/python/FL_selection/"
dataset="mnist"
xlabel=" Commnication Round on mnist "
join_ratio='0.5'
alpha='0.05'
num_clients=20
method='FedAvg'
group=50
folder_path = programpath + "/res/"
# fpath = folder_path + dataset + "_allacc_0.5_20_0.05.csv"
# fpath = dataset + "_allacc_" + str(join_ratio) + "_" + str(
#             num_clients) + "_" + str(alpha) + ".csv"
fpath=programpath+'plotResult/mnist_all.csv'
#fpath=programpath+'plotResult/dongji.csv'
picpath=folder_path + dataset + "_acc.png"

#----------------------
tex="motivation"
group=200
fpath=programpath+"plotResult/"+tex+".csv"
picpath=programpath+"plotResult/"+tex+".png"
value='Accurancy'
plot_select(fpath,picpath,value,dataset,alpha,xlabel,group)


#fpath=programpath + "/res/ " + dataset + "_allacc_" + str(join_ratio) + "_" + str(num_clients) + "_" + str(alpha) + ".csv"
#绘制不同算法的准确率曲线图
# plot_select(fpath,picpath,'Accurancy',dataset,alpha,xlabel,group)
# plot_select(fpath,picpath,'round_time',dataset,alpha,xlabel,group)
# plot_select(fpath,picpath,'highResourceClientNum',dataset,alpha,xlabel,group)
# plot_select(fpath,picpath,'distance',dataset,alpha,xlabel,group)
v=['Accurancy','round_time','distance','highResourceClientNum']
#'Kl', 'JS', 'EMD', 'round_time'
#plot_multi(fpath,picpath,['Accurancy','highResourceClientNum'],dataset,alpha,xlabel,group)
#数据分布距离图
# plot_select(fpath,picpath,'Accurancy',dataset,alpha,xlabel,group)
# find(fpath)
# from scipy.stats import pearsonr
# plot_distance(fpath)


