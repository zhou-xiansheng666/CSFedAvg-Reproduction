import torch
import torch.nn as nn
from tqdm import tqdm

from models import CNN
# tor1 = torch.randn(2,2)
# tor2 = torch.randn(2,2)
# tor3 = torch.zeros(1,1)
#
# print(tor3)
# print(tor1)
# print(torch.pow(tor1,2))
# print(tor2)
# print(torch.abs(torch.sub(tor1,tor2)))
# print(torch.div(torch.abs(torch.sub(tor1,tor2)),torch.abs(tor1)))
# model = CNN(CNN, 1, 32, 512, 10)
# ten1 = torch.randn(1,1,28,28)
# print(torch.sum(ten1))
# ten2 = torch.randn(2,1,28,28)
#
# model(ten1)
# parma = list(model.parameters())
# par1 = model.state_dict()
#
# model(ten2)
# par2 = model.state_dict()
#
# #
# for parcnt in par1:
#     print(parcnt,par1[parcnt])
#     print(parcnt,par2[parcnt])
#     print(torch.sub(par1[parcnt], par2[parcnt]))
"""
print(type(par))
k = 0
for i in parma:
    l = 1
    print("该层的结构" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和" + str(l))
    k = k + l;
print("参数综合" + str(k))"""
# a = "Hybrid"
# if a == "Hybrid":
#     print(ten[2])
#     if ten[2]== torch.tensor(3):
#         print("yes")
# else:
#     print("pig")

# print([i for i in range(60)])
# print(5 //2 *2)
# a = [0,1,2,3,4,5,6,7,8,9,10]
# for i in range(10):
#     print(a[i:i+1])
# cnt=0
# for m,n in tqdm(enumerate(a),leave=False):
#     cnt=cnt+1
#     if cnt>5:
#         break
#     print(f"m n{m},{n}")
# print(len(list(enumerate(a))))
# print(list(range(0,99)))

a = [1,5,6,7,2,10,4,6,9]
c = [1,6,6]

# d=set(a)&set(c)
# print(d)
# print(list(set(c)-d))
# print(len(set(c)))
indice = torch.argsort(torch.tensor(a),dim=0,descending=True)
print(indice)
print(1 in a)