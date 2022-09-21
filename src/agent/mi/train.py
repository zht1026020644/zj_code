
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from src.agent.mi.agent_mi import MIAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
coefficient = np.random.randint(10,100,10)
hospital_coefficient = np.random.rand(1,3)
env = MIAgent(hospital_coefficient,coefficient,size=100)
# 根据状态和动作创建Q的神经网络
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

model = build_model(env.observation_space_values,env.action_space_values)
dqn = build_agent(model,env.action_space_values)
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)
dqn.save_weights('dqn_hospital_reward.h5f',overwrite=True)

