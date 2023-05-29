import copy
import gc
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count

from torch import floor
from numpy import ceil
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .models import *
from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning

    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    首先，中心服务器将模型骨架分发给所有参与配置的客户端。
    在进行联邦学习的过程中，中心服务器对一部分客户端进行采样，
    接收局部更新的参数，将其平均为全局参数(模型)，并将其应用于全局模型。
    在下一轮中，新选择的客户端将接收更新后的全局模型作为其本地模型
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        属性:
        clients:包含参与联邦学习的Client实例的列表。
        __round: Int，表示当前的联邦轮。
        writer: SummaryWriter实例，用于跟踪度量和全局模型的丢失。
        模型:一个全局模型的实例。
        seed: Int表示随机种子。
        device: Training machine indicator (e.g. "cpu", "cuda").
        设备:训练机指示器(例如:“cpu”、“cuda”)。 CPU还是GPU
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        mp_flag:用于“client_update”和“client_evaluate”方法的多处理使用的布尔指示符。
        data_path: Path to read data.
        data_path:读取数据的路径。
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        num_shards:模拟非iid数据分裂的分片数(仅当“iid = False”时有效)。
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        iid: Boolean指示如何分割数据集(iid或非iid)。
        init_config: kwargs for the initialization of the model.
        Init_config:用于模型初始化的kwargs。
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.u
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    #Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.final_set = [0,1,2,3,4]
        self.model = eval(model_config["name"])(**model_config)
        
        self.seed = global_config["seed"]
        """  seed: 5959
            device: "cuda"
            is_mp: True"""
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]

        self.init_config = init_config

        self.fraction = fed_config["C"]  #在每个联邦回合中选择的客户端数量的比率。
        self.num_clients = fed_config["K"] #客户端数量
        self.num_rounds = fed_config["R"]  #整体次数
        self.local_epochs = fed_config["E"] #本地客户端迭代次数
        self.batch_size = fed_config["B"]   #随机梯度下降算法的分片数量
        self.douselect_P = fed_config["P"]  #选择平衡参数的用户设备比例

        self.criterion = fed_config["criterion"]   #代价函数
        self.optimizer = fed_config["optimizer"]   #随机梯度下降
        self.optim_config = optim_config
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning.为联邦学习设置所有配置"""
        # valid only before the very first round
        assert self._round == 0  #round如果等于0程序才往下执行  联邦学习执行的轮次

        # initialize weights of the model
        torch.manual_seed(self.seed)   #初始化模型的权重 设置CPU生成随机数的种子，方便下次复现实验结果
        init_net(self.model, **self.init_config)
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client 为每个客户端分割本地数据集
        local_datasets,local_datasets_spare, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)
        
        # assign dataset to each client 为每个客户端分配数据集
        self.clients = self.create_clients(local_datasets, local_datasets_spare)
        #print(f"len(self.clients{len(self.clients)})")
        # prepare hold-out dataset for evaluation 准备用于评估的保留数据集
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        #提供给定数据集上的可迭代对象
        # configure detailed settings for client upate and  配置客户端更新的详细设置
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets, local_datasets_spare):
        """Initialize each Client instance.初始化每个Client实例"""
        # print(len(local_datasets),len(local_datasets_spare))
        # print(f"number{number}")
        # print(f"len(clients){len(clients)}")
        clients = []
        number = random.sample(range(0, self.num_clients), 100)
        #前a个设备从随机数据集中获取，后从切分中随机获取self.clients-a个
        if self.iid == "Hybrid_One" or self.iid == "Hybrid_Two":
            a = int(self.douselect_P * self.num_clients)  #分布均匀的设备数量
            cnt = 0
            for k, dataset in tqdm(enumerate(local_datasets), leave=False):
                cnt=cnt+1
                if cnt > a:
                    break
                print(f"均匀的设备编号{k}")
                client = Client(client_id=k, local_data=dataset, device=self.device)
                clients.append(client)

            while cnt <= self.num_clients:
                cnt = cnt + 1;
                data_list_spare = list(enumerate(local_datasets_spare))
                client = Client(client_id=cnt-2, local_data=data_list_spare[number[cnt-a-2]][1], device=self.device)
                clients.append(client)

        else:
            for k, dataset in tqdm(enumerate(local_datasets), leave=False):
                client = Client(client_id=k, local_data=dataset, device=self.device)
                clients.append(client)

        client_assist = Client(client_id=self.num_clients, local_data=list(enumerate(local_datasets))[number[0]][1], device=self.device)
        clients.append(client_assist)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """将更新后的全局模型发送到选定的/所有客户端"""
        if sampled_client_indices is None:
            #在第一轮联邦轮之前和最后一轮联邦轮之后，将全局模型发送给所有客户端
            assert (self._round == 0) or (self._round == self.num_rounds)
            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)
                #防止两个量地址相同，造成值同步改变

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            #f字母表示格式化大括号中的内容，即用值输出，其余的照常输出
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients and assist client
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False): #leave参数表示是否保持进度条
                self.clients[idx].model = copy.deepcopy(self.model)
            self.clients[self.num_clients].model = copy.deepcopy(self.model) #给辅助设备分发全局模型

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """随机选择客户端"""
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()


        if len(self.final_set) < ceil(self.num_clients * self.douselect_P):
            num_sampled_clients = min(max(int(self.fraction * self.num_clients), 1) * 2, self.num_clients)
        else:
            num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        #从一维数组或者列表元组a中随机抽取size个数据
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0  #最终确定平衡系数有关
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])


        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size


    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        #print(f"selected_index{selected_index}")
        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        #coefficients权重的平均系数
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        #enumerate同时显示下标和列表中的值
        for it, idx in tqdm(enumerate(sampled_client_indices + self.final_set), leave=False):
            local_weights = self.clients[idx].model.state_dict() # state_dict()字典存放了模型参数
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices+self.final_set)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """ 调用每个选定客户端的"client_evaluate"函数"""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices+self.final_set))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def count_dis(self, sampled_client_indices):
        weight_assist = self.clients[self.num_clients].model.state_dict() #辅助设备的模型权重
        weight_sum_ass = torch.zeros(1, 1).cpu() #存储模型之间的权重差异
        for weight_temp in weight_assist:
            weight_powass = torch.pow(weight_assist[weight_temp],2).cpu()
            #weight_powass.to(self.device)
            weight_sum_ass += torch.sum(weight_powass).cpu()
        weight_assist_div = torch.sqrt(weight_sum_ass).cpu()  #求出了||辅助设备权重||
        dis_list = []  #存储每个设备与辅助设备的权值差异
        for idx in sampled_client_indices:
            weight_select = self.clients[idx].model.state_dict()
            weight_sum = torch.zeros(1,1).cpu()
            for weight_temp in weight_assist:
                weight_sub = torch.sub(weight_select[weight_temp],weight_assist[weight_temp].cpu())
                weight_pow = torch.pow(weight_sub,2).cpu()
                weight_sum += torch.sum(weight_pow)
            weight_div = torch.sqrt(weight_sum).cpu()
            dis_list.append(torch.div(weight_div,weight_assist_div))

        return dis_list


    def train_federated_model(self):
        """Do federated training."""
        # 按照设置的客户端比例随机选择客户端
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients 发送全局模型到选定的客户端
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset

        pub_slement = set(self.final_set)&set(sampled_client_indices)
        list_sub = list(set(self.final_set) - pub_slement)
        len_sub = len(self.final_set) - len(list_sub)
        update_set = list_sub+sampled_client_indices

        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, update_set)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(update_set)

        selected_total_size += len(self.clients[self.num_clients]) * len_sub
        #这里为了让共同的数据集也参与模型聚合，用以平衡参数，但只能在各个客户端数据量都相同时使用
        if len(self.final_set) < self.num_clients * self.douselect_P:
            dis_list = self.count_dis(sampled_client_indices)
            sort_indic = torch.argsort(torch.tensor(dis_list), dim=0, descending=True)
            print(f"dis_list{dis_list}  torch{torch.tensor(dis_list)[sort_indic]}")
            print(f"随机编号{sampled_client_indices}")
            put_size = ceil(torch.tensor(self.num_clients * self.fraction * self.douselect_P))
            cnt=0
            while cnt < put_size and len(self.final_set) < ceil(self.num_clients * self.douselect_P):
                temp_indice = sampled_client_indices[sort_indic[cnt]]
                if temp_indice in self.final_set:
                    cnt = cnt + 1
                    continue
                self.final_set.append(sampled_client_indices[sort_indic[cnt]])
                selected_total_size += len(self.clients[sampled_client_indices[sort_indic[cnt]]])
                cnt = cnt + 1

        #print(f"len(self.final_set){len(self.final_set)}")
        # evaluate selected clients with local dataset (same as the one used for local update)
        """       if self.mp_flag:
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)
        """

        # calculate averaging coefficient of weights
        print(f"参与聚合的设备数量{len(sampled_client_indices+self.final_set)}")
        print(f"最终集合中有哪些设备{self.final_set}")
        print(f"select_total_size{selected_total_size}")
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices+self.final_set]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """使用全局保留数据集(self.data)评估全局模型"""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        # if test_accuracy >= 0.96:
        #     sys.exit()
        return test_loss, test_accuracy

    def fit(self):
        """执行联邦学习的整个过程"""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, P_{self.douselect_P}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Accuracy', 
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, P_{self.douselect_P}, IID_{self.iid}": test_accuracy},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()
