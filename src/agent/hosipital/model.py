# coding: utf-8 -*- coding: utf-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorflow.python.training import momentum
from torch import optim
from hospital import Hospital
from src import utils
import numpy as np
import random


class Model(nn.Module):
    '''
    model for hospital
    '''
    def __init__(self, name, coefficient, path):
        super(Model, self).__init__()
        self.hospital = Hospital(name, coefficient, path)
        data_shape = self.get_data().shape[1]
        self.fc1 = nn.Linear(data_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        # TODO
        self.fc3 = nn.Linear(512, 10000)

    def get_data(self):
        '''
        get input data
        :return: data :ndarray 100*3784
        '''
        return utils.get_input(self.hospital.path, self.hospital.coefficient)

    def forward(self, x):
        '''
        network
        三层：
        第一层：
        3784 * 512
        第二层：
        512 *512
        第三层
        512*10000  （目前是随机产生10000随机决策）
        :param x:
        :return:
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model("市二", 0.75, 'D:\work_software\zj_code\input_data.csv')
# 优化函数
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
train_loader = model.get_data()
train_loss=[]
for epoch in range(epochs):
    for idx in range(20):
        # 5条记录一批
        x = torch.from_numpy(train_loader[idx * 5:(idx + 1) * 5, :].astype(np.float32))
        out = model(x)
        # 临时使用随机的10种作为决策 TODO
        # print("out:",out.shape)
        print(out)
        y = torch.zeros(5,10000)
        for i in range(5):
            for k in range(10):
               j = random.randint(0,9999)
               y[i][j] = 1

        # loss = F.cross_entropy(out,y)
        # 计算损失函数
        # loss = F.mse_loss(out,y)
        loss = F.multilabel_soft_margin_loss(out,y)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
        # 存储loss
        train_loss.append(loss.item())
utils.plot_curve(train_loss)
print(train_loss)

