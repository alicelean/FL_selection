import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id,traindata,testsdata, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        #记录聚合误差
        self.localloss = []
        self.globalloss = []
        self.global_model=None
        self.selected=False
        self.error=[]
        self.distance = 0
        self.alldistance = 0
        self.alllabel = None
        self.localmodel=copy.deepcopy(args.model)
        self.isselected = False#上一轮被选择

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.sizerate = 0
        #记录迭代的次数
        self.current_round=0
        self.timecost = []
        self.losslist=[]
        self.prev_weight=None
        self.layer=1
        print(args.num_classes,"args.num_classes")

        self.traindata = traindata
        self.testsdata = testsdata
        if self.dataset == 'agnews':
            self.labellenght =4
        if self.dataset=='Cifar100':
            self.labellenght =100
        if self.dataset == 'Cifar10'or self.dataset == 'fmnist' or self.dataset == 'mnist':
            self.labellenght =10
        if args.dataset == 'Tiny-imagenet':
            self.labellenght = 200

        self.label = [0 for i in range( self.labellenght)]
        self.jsweight=0

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.globallosess= nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.local_losss = nn.CrossEntropyLoss()
        self.localoptimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.locallearning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

    def load_train_data(self, batch_size=None):
        #print("training batch_size is :",batch_size)
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)

        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def set_parameters(self, model):
        '''
        将model 复制给本地模型
        Args:
            model:

        Returns:

        '''
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        if testloaderfull is None:
            print("client test_metrics Error: Failed to load test data.")
            return None

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        # 记录NaN值的数量
        nan_x=0
        nan_y=0
        nan_output=0
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 检查输入数据中是否存在NaN值
                if self.dataset!='agnews':
                    if torch.isnan(x).any() :
                        nan_x += 1
                        continue
                    if torch.isnan(y).any():
                        nan_y += 1

                output = self.model(x)

                # 检查模型输出中是否存在NaN值
                if self.dataset != 'agnews':
                    if torch.isnan(output).any():
                        nan_output += 1
                        continue

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        # self.model.cpu()
        # self.save_model(self.model, 'model')
        nan_count=nan_x+nan_y+nan_output
        if nan_count > 0:
            nan_ratio = nan_count / len(testloaderfull)  # 计算NaN值在测试数据中的比例
            print(f"client {self.id} ,nan_x {nan_x},nan_y {nan_y},nan_output {nan_output},total NaN value ratio in test data: {nan_ratio:.2%}")
        if nan_count!=len(testloaderfull) :
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            return test_acc, test_num, auc
        else:
            print(f"ERROR:testloaderfull {len(testloaderfull)},nan_count {nan_count}, test_num {test_num}")
            return 0, test_num, 0



    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                #print("model print",output ,y)
                loss = self.loss(output, y)
                #print("test:",loss.shape)
                loss=loss.mean()
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def train_metrics_last(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()
        truelabel=[]
        predictlabel=[]
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                truelabel.append(y)
                predictlabel.append(torch.argmax(output, dim=1))
                # print("model print",output ,y)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return train_num,truelabel,predictlabel

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def setlabel(self):
        # train_data = [(x, y) for x, y in zip(X_train, y_train)]
        label = []
        total_num=0
        for data in self.traindata:
            label.append(data[1].tolist())
        for data in self.testsdata:
            label.append(data[1].tolist())
        for i in label:
            self.label[i] += 1
            total_num+=1

        print(f"client base client  {self.id}, total_num is {total_num},label is {self.label}")


    def local_initialization(self, received_global_model, round):
        #self.AAW.adaptive_aggregation_weight(received_global_model, self.model, round)
        self.AAW.adaptive_aggregation_weight_update(received_global_model, self.model, round)


    def init_local_model_1(self,
                                 global_model: nn.Module,
                                 local_model: nn.Module) -> None:

        #初始化全局模型和局部模型
        self.layer_idx=1
        a=0.9
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())
        model_lt = copy.deepcopy(local_model)
        params_lt = list(model_lt.parameters())
        # preserve all the updates in the lower layers,将底层全局模型信息保留
        for param, param_g in zip(params_l[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning


        # only consider higher layers
        params_lp = params_l[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_ltp = params_lt[-self.layer_idx:]
        # initialize the higher layers in the temp local model
        for param_ltp, param, param_g in zip(params_ltp, params_lp, params_gp):
            param_ltp.data = param*a + param_g *(1-a)

        # obtain initialized local model
        for param, param_t in zip(params_lp, params_ltp):
            param.data = param_t.data.clone()
    def init_local_model(self, global_model: nn.Module,local_model: nn.Module) -> None:

        #初始化全局模型和局部模型

        a=0.5
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())
        model_lt = copy.deepcopy(local_model)
        params_lt = list(model_lt.parameters())

        # initialize the higher layers in the temp local model
        for param_ltp, param, param_g in zip(params_lt, params_l, params_g):
            param_ltp.data = param*a + param_g *(1-a)

        # obtain initialized local model
        for param, param_t in zip(params_l, params_lt):
            param.data = param_t.data.clone()
    def calculate_local_loss(self,global_model=None):
        #self.model_is_same(global_model)
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        num_samples = 0
        data_loader = self.load_train_data()
        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                # Calculate the loss for this batch
                loss =self.loss(outputs, labels)
                # Update the total loss and the number of samples
                total_loss += loss.item()
                num_samples += len(inputs)
        # Calculate the average loss over all batches,total_loss / num_samples

        self.localloss.append(total_loss / num_samples)


        #print(f"client {self.id},Local loss:samples is  {num_samples },loss is{total_loss:.4f}")

    def calculate_gobal_loss(self,testmodel):
        #print("''''''''''calculate_loss''''''''''''''''")
        #self.model_is_same(testmodel)
        testmodel.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        num_samples = 0
        data_loader = self.load_train_data()
        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = testmodel(inputs)
                # Calculate the loss for this batch
                loss = self.loss(outputs, labels)
                # Update the total loss and the number of samples
                total_loss += loss.item()

                num_samples += len(inputs)
        # Calculate the average loss over all batches,total_loss / num_samples
        avgloss=total_loss / num_samples
       # print(f" islocal {IsLocal},loss is:{avgloss}")

        self.globalloss.append(avgloss)

        # print(f"client {self.id},Local loss:samples is  {num_samples },loss is{total_loss:.4f}")

    def calculate_global_loss_aaw(self, testmodel):
        self.model.eval()  # Set the original model to evaluation mode
        total_loss = 0.0
        num_samples = 0
        data_loader = self.load_train_data()

        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass using the testmodel
                test_outputs = testmodel(inputs)

                # Calculate the loss for this batch using the testmodel
                loss = self.loss(test_outputs, labels)

                # Update the total loss and the number of samples
                total_loss += loss.item()
                num_samples += len(inputs)

        # Calculate the average loss over all batches
        avgloss = total_loss / num_samples

        self.globalloss.append(avgloss)

        # print(f"client {self.id}, Local loss: samples is {num_samples}, loss is {total_loss:.4f}")

    def model_is_same(self,model1):
        '''
        判定两个模型是否一致
        Args:
            model1:

        Returns:

        '''
        import torch
        # 检查模型结构是否一致
        if str(model1) == str(self.model):
            print("模型结构一致")
        else:
            print("模型结构不一致")

        # 检查每个层的参数是否一致
        parameters1 = list(model1.parameters())
        parameters2 = list(self.model.parameters())

        parameters_equal = all(
            [torch.equal(p1, p2) for p1, p2 in zip(parameters1, parameters2)]
        )

        if parameters_equal:
            print("模型参数一致")
        else:
            print("模型参数不一致")

    def load_train_data_center(self, batch_size=None):

        if batch_size == None:
            batch_size = self.batch_size
        #train_data = read_client_data(self.dataset, self.id, is_train=True)
        train_data=self.traindata
        print("training batch_size is :", batch_size,len(train_data))
        return DataLoader(train_data, 77, drop_last=True, shuffle=False)


