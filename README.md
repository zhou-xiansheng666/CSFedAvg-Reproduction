#论文复现
"Client Selection for Federated Learning with non-IID Data in Mobile Edge Computing"对该篇论文算法的复现.

## Requirements
* See `requirements.txt`

## Configurations
* See `config.yaml`

## Run
* `python3 main.py`

## Results
### MNIST
* Number of clients: 100 (K = 100)
* Fraction of sampled clients: 0.1 (C = 0.1)
* Number of rounds: 500 (R = 500)
* Number of local epochs: 10 (E = 10)
* Batch size: 10 (B = 10)
* Optimizer: `torch.optim.SGD`
* Criterion: `torch.nn.CrossEntropyLoss`
* Learning rate: 0.01
* Momentum: 0.9
* Initialization: Xavier

#项目框架与指导




