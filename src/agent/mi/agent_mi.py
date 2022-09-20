import numpy as np
import pandas as pd
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

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
        self.observation_space_values = (init_coefficient.size,self.size)
        self.action_space_values = (init_coefficient.size,3)

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
    def setp(self,action):
        self.episode_step += 1
        self.player.action(action)
        new_observation = self.player.coefficient
        reward = self.cal_reward()
        done = False
        # 超出范围或者调整了1000次停止
        if np.sum(self.player.coefficient>self.size or self.player.coefficient<= self.size) >1 or self.episode_step>=1000:
            done = True
        # TODO
        info = {}
        return new_observation,reward,done,info
    # def get_qtable(self,qtable_name=None):
    #     qtable = pd.DataFrame()



    def cal_hospital_factor(self):
        pass

    def cal_reward(self):
        return np.random.randint(100)


class Cube(object):
    def __init__(self,init_coefficient):
        # 初始值
        self.init_coefficient = init_coefficient
        # 决策中的值
        self.coefficient = init_coefficient
    def __str__(self):
        return f'{self.init_coefficient}'
    def action(self,choice):
        self.coefficient += choice


def build_model(status, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + status))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    # window_length:mini_batch
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # nb_steps_warmup:热身步数
    # target_model_update 解决bootstrap问题
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    return dqn

if __name__ == "__main__":
    coefficient = np.random.randint(100,[100,1])
    hospital_coefficient = np.random.rand(1,3)
    env = MIAgent(hospital_coefficient,coefficient,size=10)
    model = build_model(env.observation_space_values,env.ACTION_SPACE_VALUES)
    dqn = build_agent(model,env.ACTION_SPACE_VALUES)





