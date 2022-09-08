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
    def __init__(self, name, coefficient, path):
        super(Model, self).__init__()
        self.hospital = Hospital(name, coefficient, path)
        data_shape = self.get_data().shape[1]
        self.fc1 = nn.Linear(data_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        # TODO
        self.fc3 = nn.Linear(512, 10000)

    def get_data(self):
        return utils.get_input(self.hospital.path, self.hospital.coefficient)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model("市二", 0.75, 'D:\work_software\zj_code\input_data.csv')
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1
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
        loss = F.mse_loss(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
utils.plot_curve(train_loss)
print(train_loss)

