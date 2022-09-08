# coding=utf-8
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt


def age_code(x):
    if x < 100 and x >= 90:
        x = 9
    elif x < 90 and x >= 80:
        x = 8
    elif x < 80 and x >= 70:
        x = 7
    elif x < 70 and x >= 60:
        x = 6
    elif x < 60 and x >= 50:
        x = 5
    elif x < 50 and x >= 40:
        x = 4
    elif x < 40 and x >= 30:
        x = 3
    elif x < 30 and x >= 20:
        x = 2
    elif x < 20 and x >= 10:
        x = 1
    else:
        x = 0
    return x


def code_temperature(x):
    if x <= 37:
        x = 0
    elif x > 37 and x <= 38:
        x = 1
    elif x > 38 and x <= 39:
        x = 2
    else:
        x = 3
    return x


def get_input(path, hospital_coefficient):
    '''
    读取csv文件进行特征工程作为医院端智能体输入
    :param path:
    :param hospital_coefficient:
    :return:
    '''
    df_data = pd.read_csv(path, encoding='utf-8')
    hostpital = np.ones(df_data.shape[0]) * hospital_coefficient
    df_hostpital = pd.DataFrame(hostpital, columns=['hostpital'])
    df_gender = df_data['gender_source_value'].replace('男', 1).replace('女', 0).replace(np.nan,1)
    df_age = pd.DataFrame((2022.0 - df_data['year_of_birth']).apply(age_code))
    df_temperature = pd.DataFrame(df_data["体温"].replace(np.nan, 36.5).apply(code_temperature))
    df_breath = pd.DataFrame((df_data["呼吸"] / (df_data["呼吸"].max())).replace(np.nan,0.5))
    df_systolic_pressure = pd.DataFrame((df_data["收缩压"] / (df_data["收缩压"].max())).replace(np.nan,0.5))
    df_pulse = pd.DataFrame((df_data["脉搏"] / (df_data["脉搏"].max())).replace(np.nan,0.5))
    df_diastolic_pressure = pd.DataFrame((df_data["舒张压"] / (df_data["舒张压"].max())).replace(np.nan,0.5))
    df_diagnosis = df_data.iloc[:, 20:df_data.shape[1]].fillna(value=0)
    df_merge = [df_hostpital, df_gender, df_age, df_temperature, df_breath, df_systolic_pressure, df_pulse,
                df_diastolic_pressure, df_diagnosis]
    df_input = pd.concat(df_merge, axis=1)
    return df_input.values

def plot_curve(data):
    '''
    绘制损失函数
    :param data:
    :return:
    '''
    fig = plt.figure()
    plt.plot(range(len(data)), data,color='blue')
    plt.legend(['values'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    input = get_input('D:\work_software\zj_code\input_data.csv', 0.6)
    print(np.argwhere(np.isnan(input)))
