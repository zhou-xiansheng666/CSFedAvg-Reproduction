import os
import time

import pickle
import yaml
import threading
import logging

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board

if __name__ == "__main__":
    # read configuration file
    with open('./config.yaml', encoding='utf-8') as c: #with 关键字会自动执行close()程序，防止资源被异常程序持续占用
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]  #嵌套的字典
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]

    # modify log_path +6+to contain current time
    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=(log_config["log_path"]+time.strftime('%Y%m%d_%H')),filename_suffix='FL')
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
        ).start()
    time.sleep(3.0)

    # set the configuration of global logger 设置全局记录器的配置
    logger = logging.getLogger(__name__)  #内置的变量 __name__ logger:日志对象
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration 设置全局记录器的配置
    message = "\n[WELCOME] Unfolding configurations...! 展开配置"
    print(message); logging.info(message)

    for config in configs:
        print(config); logging.info(config)
    print()

    # initialize federated learning  联邦学习初始化
    central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()

