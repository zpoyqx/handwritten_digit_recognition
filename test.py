"""
进行模型的评估
"""

"""
进行模型的训练
"""
from dataset import get_dataloader
from models import  MnistModel
from torch import optim
import torch.nn.functional as F
import conf
from tqdm import tqdm
import numpy as np
import torch
import os



# 1. 实例化模型，优化器，损失函数
model = MnistModel().to(conf.device)

if os.path.exists("./models/model.pkl"):
    model.load_state_dict(torch.load("./models/model.pkl"))

test_dataloader = get_dataloader(train=False)
total_loss = []
total_acc = []
# 禁止梯度计算
with torch.no_grad():
    for input,target in test_dataloader: #2. 进行循环，进行训练
        input = input.to(conf.device)
        target = target.to(conf.device)
        #计算得到预测值
        output = model(input)
        #得到损失
        loss = F.nll_loss(output,target)
        #反向传播，计算损失
        total_loss.append(loss.item())

        #计算准确率
        ###计算预测值
        # output越大，正确率越高
        pred = output.max(dim=-1)[-1] # 获取每一行准确率的位置
        total_acc.append(pred.eq(target).float().mean().item())
print("test loss:{},test acc:{}".format(np.mean(total_loss),np.mean(total_acc)))



