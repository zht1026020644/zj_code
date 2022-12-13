import numpy as np
import pandas as pd
import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MIAgent(object):
    # 随机选择抽取下一个动作的概率为0.2
    epilon = 0.2
    # 训练期数
    EPISODES = 300000
    SHOW_EVERY = 3000
    # 衰减
    EPS_DECAY = 0.999998
    # 折扣
    DISCOUNT = 0.95
    # 学习率
    learning_rate = 0.001


    def __init__(self, hospital_coefficient, init_coefficient,delta = 1, size=1000):
        # 点数范围大小
        self.size = size
        # 医院系数
        self.hospital_coefficient = hospital_coefficient
        # 当前周期点数值
        self.init_coefficient = init_coefficient
        # 状态空间
        self.observation_space_values = (init_coefficient.size,)
        # 动作个数
        self.action_space_values = 3**init_coefficient.size

    def reset(self):
        '''
        重置
        :return: 返回一个observation
        '''
        # 决策者
        self.player = Cube(self.init_coefficient)
        observation = tuple(self.player.init_coefficient.tolist())
        self.episode_step = 0
        return observation
    def step(self,action):
        self.episode_step += 1
        self.player.action(action)
        new_observation = self.player.coefficient
        # print(f"action:{action}")
        # print(f'{self.episode_step}:{self.player.coefficient}')
        reward = self.cal_reward()
        done = False
        # 超出范围或者调整了1000次停止
        if (np.sum(self.player.coefficient>self.size)+np.sum(self.player.coefficient<0)) >1 or self.episode_step>=1000:
            done = True
        # TODO
        info = {}
        return new_observation,reward,done,info
    # def get_qtable(self,qtable_name=None):
    #     qtable = pd.DataFrame()

    def render(self):
        '''
        可视化
        '''
        # TODO
        pass



    def cal_hospital_factor(self):
        '''
        计算系数
        '''
        # TODO
        pass

    def cal_reward(self):
        '''
        计算奖励值
        '''
        # 读取医院患者数据 分组后计算奖励函数


        return np.random.randint(100)


class Cube(object):
    def __init__(self,init_coefficient):
        # 初始值
        self.init_coefficient = init_coefficient
        # print(f'初始值： {self.init_coefficient} -------')
        # 决策中的值
        self.coefficient = init_coefficient
    def __str__(self):
        return f'{self.init_coefficient}'
    def action(self,choice):
        choice_item = choice.item()
        choice_transform = Cube.decimal_conversion(choice_item,3)
        while len(choice_transform)<self.coefficient.size:
            choice_transform.append(0)
        # print(f'动作：  {choice}   动作编码：  {choice_transform}  ------')
        for i in range(len(choice_transform)):
            if choice_transform[i]==0:
                self.coefficient[i] += -1
            elif choice_transform[i] == 2:
                self.coefficient[i] += 1
        # print(f'变化值： {self.coefficient} ------')


    @staticmethod
    def decimal_conversion(n, x):
        '''
        进制转换
        '''
        # n为待转换的十进制数，x为机制
        b = []
        while True:
            s = n // x  # 商
            y = n % x  # 余数
            b = b + [y]
            if s == 0:
                break
            n = s
        b.reverse()
        return b









