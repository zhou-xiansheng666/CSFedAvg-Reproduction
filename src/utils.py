import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


##网络权重初始化、数据集处理

#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True


#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        # hasattr，确保对象中是否包含某个属性
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    初始化网络权重
    Args:
        model: A torch.nn.Module to be initialized  要初始化的torch.nn.Module
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l 初始化方法名称(normal | xavier | kaiming | orthogonal)
        init_gain: Scaling factor for (normal | xavier | orthogonal).缩放因子为(正常| xavier |正交)
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
                表示网络运行在哪个GPU上的列表或int。(例如，[0,1,2]，0)
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model


#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):   #自己的Dataset的子类
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y  #x表示Img，y表示label

    def __len__(self):
        return self.tensors[0].size(0) #返回所有数据集的数量


def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients.以IID或非IID方式拆分整个数据集，以便分发给客户端。"""
    dataset_name = dataset_name.upper()
    # 从torchvision获取数据集。数据集(如果存在)
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset 对每个数据集设置不同的转换
        if dataset_name in ["CIFAR10"]:
            # torchvision是pytorch深度学习框架下得图形库，transforms是常用图形变换操作库，如旋转、裁剪
            transform = torchvision.transforms.Compose(
                [

                    torchvision.transforms.ToTensor(),  # 将图片数值映射到0-1
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()

        elif dataset_name in ["EMNIST"]:
            transform = torchvision.transforms.ToTensor()

        # prepare raw training & test datasets

        DownloadDataset = True

        if dataset_name in ["EMNIST"]:
            training_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=True,
                download=DownloadDataset,
                transform=transform,
                split='letters'
            )
            test_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=False,
                download=DownloadDataset,
                transform=transform,
                split='letters'
            )

        else:
            training_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=True,
                download=DownloadDataset,
                transform=transform
            )
            test_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=False,
                download=DownloadDataset,
                transform=transform
            )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets 为灰度图像数据集解压缩通道维度
    if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]
    #np.unique函数去重并从小到大排序
    #print(f"num_categories:{num_categories}___{type(training_dataset.targets)}")
    #int 10 、torch.tensor
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
    training_dataset_spare = training_dataset
    local_datasets_spare = []
    # split dataset according to iid flag 根据iid标志拆分数据集
    if iid == True:
        # shuffle data 混乱的数据
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients 将数据分区到num_clients中
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        #成本地数据集
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]

    elif iid == "Hybrid_One":

        #100个分片大小为600的数据集合
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients 将数据分区到num_clients中
        split_size = len(training_dataset) // num_clients
        #print(f"split_size:{split_size}") #600
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )
        #print(f"split_datasets{len(split_datasets)}") #100
        # finalize bunches of local datasets  完成本地数据集
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]
        #print(f"local_datasets的长度是{len(local_datasets_spare)}") 100
        #100个随机，分片大小为600
        sorted_indices = torch.argsort(torch.Tensor(training_dataset_spare.targets))
        #torch.argsort返回排序后的下标
        """
        print("tensor_size:"+str(sorted_indices.size()))
        cnt=0
        for i in range(60000):

            if(torch.Tensor(training_dataset.targets)[i]==torch.tensor(1)):
                cnt=cnt+1
        print(f"cnt的值是{cnt}")
        """
        training_inputs_spare = training_dataset_spare.data[sorted_indices]
        training_labels_spare = torch.Tensor(training_dataset_spare.targets)[sorted_indices]
        # 按标签对数据排序
        # 先将数据划分为分片
        num_shards_one = (num_shards // 2) #每个客户端只有一类数据集时的分片数
        shard_size = len(training_dataset_spare) // num_shards_one  # 一个分片600
        shard_inputs = list(torch.split(torch.Tensor(training_inputs_spare), shard_size))
        shard_labels = list(torch.split(torch.Tensor(training_labels_spare), shard_size))

        # 对列表进行排序，以便方便地将来自一个类的样本分配给每个客户机
        shard_inputs_sorted, shard_labels_sorted = [], []
        for i in range(num_shards_one // num_categories):
            for j in range(0, ((num_shards_one // num_categories) * num_categories), (num_shards_one // num_categories)):
                shard_inputs_sorted.append(shard_inputs[i + j])
                shard_labels_sorted.append(shard_labels[i + j])

        # 通过为每个客户端分配碎片来完成本地数据集
        shards_per_clients = num_shards_one // num_clients
        local_datasets_spare = [
            CustomTensorDataset(
                (
                    torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                    torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                ),
                transform=transform
            )
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ]

    else:
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients 将数据分区到num_clients中
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        # finalize bunches of local datasets  完成本地数据集
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]

        sorted_indices = torch.argsort(torch.Tensor(training_dataset_spare.targets))
        training_inputs_spare = training_dataset_spare.data[sorted_indices] #train_inputs:<class 'numpy.ndarray'>
        training_labels_spare = torch.Tensor(training_dataset_spare.targets)[sorted_indices] #training_labels:<class 'torch.Tensor'>
        #print(f"training_labels:{type(training_labels)}")
        # 先将数据划分为分片
        #//除法且向下取整
        #print(f"train_labels{training_labels[3000]} {training_labels[9000]}")
        shard_size = len(training_dataset_spare) // num_shards  # 300 ，平均每人600条数据
        shard_inputs = list(torch.split(torch.Tensor(training_inputs_spare), shard_size))
        shard_labels = list(torch.split(torch.Tensor(training_labels_spare), shard_size))
        #print(f"{shard_labels}")
        #print(f"len(shard_inputs):{len(shard_inputs)}") 200
        # 对列表进行排序，以便方便地将来自至少两个类的样本分配给每个客户机
        shard_inputs_sorted, shard_labels_sorted = [], []
        for i in range(num_shards // num_categories):
            for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                #print("i+j:"+str(((i+j)*300)//6000)) #range不取上界，
                shard_inputs_sorted.append(shard_inputs[i + j])
                shard_labels_sorted.append(shard_labels[i + j])

        # 通过为每个客户端分配碎片来完成本地数据集
        shards_per_clients = num_shards // num_clients
        local_datasets_spare = [
            CustomTensorDataset(
                (
                    torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                    torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                ),
                transform=transform
            )
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ]

    return local_datasets, local_datasets_spare, test_dataset
