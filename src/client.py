import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server. 客户端对象由中心服务器发起"""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """用于参数聚合的局部模型getter"""
        return self.__model

    @model.setter
    def model(self, model):
        """用于传递全局聚合模型参数的局部模型setter"""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """建立各客户端的通用配置;由中心服务器调用。"""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True) #用于将数据集分批次传送给模型
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self):
        """使用本地数据集更新本地模型"""
        self.model.train() #该方法用于启用Batch Normalization层和Dropout层
        self.model.to(self.device)    #将模型放于GPU进行训练

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device) #转换标签和预测结果的类型
  
                optimizer.zero_grad()  #将优化器的梯度清零
                outputs = self.model(data)  #得到模型更替后的输出
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()  #损失值反向传播
                optimizer.step() #优化器参数更新

                if self.device == "cuda": torch.cuda.empty_cache()         #清空显存
        self.model.to("cpu")

    def client_evaluate(self):
        """使用局部数据集(与训练集相同)评估局部模型"""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad(): #requires_grad参数为false，即反向传播时不会自动求导
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                #output类型为torch.tensor,大小为(batch, class_num)
                predicted = outputs.argmax(dim=1, keepdim=True)
                #argmax返回每个维度上最大值也就是最有可能的位置的下标
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                #torch.eq返回batch维的bool值,sum()累加,item()返回张量元素的值，仅当张量元素只有一个的时候使用
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy
