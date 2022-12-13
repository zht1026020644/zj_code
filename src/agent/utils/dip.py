import re
import pickle
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
from pyglet.window.key import SELECT
from tqdm.notebook import tqdm
import pandas as pd
import cx_Oracle

DIPDict = pickle.load(open('D:\\下载\\DIPDict', 'rb'))
DIPMatchDict = pickle.load(open('D:\\下载\\DIPMatchDict', 'rb'))

def disease_match(df):
    df['disease_code'] = None
    for i in tqdm(df.index):
        diag = df.loc[i,'principal_condition_code']
        if pd.isna(diag):
            df.loc[i,'disease_code'] = '缺少诊断'
            continue
        diag = diag[:5]
        diag = re.sub('x', 'X', diag)
        if diag not in DIPMatchDict.keys():
            df.loc[i,'disease_code'] = 'DIP目录库不包含该诊断'
            continue

        oper_list = df.loc[i,'procedure_code']
        if oper_list != oper_list:
            oper_list = set()

        # 筛选所有包含在oper_list的oper_comb
        oper_comb_list = list(DIPMatchDict[diag].keys())
        selected_oper_comb = []
        for oper_comb in oper_comb_list:
            if set(oper_comb).issubset(oper_list): # set(())=空集，是任何一个集合的子集
                selected_oper_comb.append(oper_comb)
        if len(selected_oper_comb) == 0:
            df.loc[i,'disease_code'] = 'DIP目录库不包含该操作'
            continue

        # 选择最长oper_comb的病种编码
        max_lenth_oper_comb = max(selected_oper_comb, key=len) # 空集长度为0
        disease_code = DIPMatchDict[diag][max_lenth_oper_comb]
        df.loc[i,'disease_code'] = disease_code
def id_to_code_mapping(conn,df):
    # df['principal_condition_code'] = None
    # df.insert(loc=17, column='principal_condition_code', value='')
    # print(df)
    principal_condition_code = []
    cursor = conn.cursor()
    for i in tqdm(df.index):
        id = df.loc[i, 'CONDITION_CONCEPT_ID']
        sql = "SELECT * FROM OMOP.CONCEPT c WHERE c.CONCEPT_ID = {} ".format(id)
        # print(sql)
        cursor.execute(sql)
        all = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        data = pd.DataFrame(all, columns=cols)
        # print(cols)
        # for row in all:
        #     one_text = dict(zip(cols,row))
        #     df[i,'principal_condition_code'] = one_text['CONCEPT_CODE']
        # df.to_csv('result.scv',index = False)
        # df[i,'principal_condition_code'] = data['CONCEPT_CODE'][0]
        principal_condition_code.append(data['CONCEPT_CODE'][0])
    new_columns = pd.DataFrame(np.array(principal_condition_code), index=df.index,columns=['principal_condition_code'])
    df = pd.concat([df,new_columns],axis = 1)
    df.to_csv('result.csv', index=False)

    return df

        # print()
def get_concept_code(conn,df):
    concept_code = []
    concept_id = []
    cursor = conn.cursor()
    for i in tqdm(df.index):
        id = df.loc[i, 'CONCEPT_ID']
        sql = "SELECT * FROM GOVERN.PROCEDURE_OCCURRENCE_NEW pon WHERE pon.PROCEDURE_SOURCE_CONCEPT_ID = {} ".format(id)
        cursor.execute(sql)
        all = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        data = pd.DataFrame(all, columns=cols)
        temp_id = str(data['PROCEDURE_CONCEPT_ID'][0])
        if temp_id.startswith("112"):
            concept_id.append(temp_id)
        else:
            concept_id.append("nan")
    print(concept_id)
    print("=====================================id完成=============================================\n\n\n")
    for id in concept_id:
        if id == "nan":
            concept_code.append("nan")
        else:
            sql = "SELECT * FROM OMOP.CONCEPT c  WHERE CONCEPT_ID  = {}".format(id)
            cursor.execute(sql)
            all = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            data = pd.DataFrame(all, columns=cols)
            concept_code.append(data['CONCEPT_CODE'][0])
    new_columns = pd.DataFrame(np.array(concept_code), index=df.index, columns=['procedure_code'])
    new_columns.to_csv("procedure_code.csv",index=False)
    print("=====================================code完成=============================================\n\n\n")








if __name__ == "__main__":
    df1 = pd.read_excel("C:\\Users\\ZHT\\Desktop\\condition_surgey_price_sum.xlsx")
    df2 = pd.read_csv("result.csv")
    # print(df2['principal_condition_code'])
    df2 = pd.DataFrame(df2['principal_condition_code'],index=df2.index,columns=['principal_condition_code'])
    # df2.columns = ['principal_condition_code']
    df3 = pd.read_csv("procedure_code.csv")
    df = pd.concat([df1,df2,df3],axis=1)
    # print(df)



    # conn = cx_Oracle.connect('datagroup','datagroup','10.11.96.93:8888/ORCL')
    # get_concept_code(conn,df)




    # df = id_to_code_mapping(conn,df)




    disease_match(df)

    # df['disease_code'].to_csv("final_result.csv")
    # df_group = df.groupby('disease_code')
    #
    # count_list = df['disease_code'].value_counts().tolist()

    # print(len(count_list))

    # 随机产生5w个患者
    df_patient = pd.DataFrame(columns=df.columns)

    np.random.seed(21)
    index = 0
    for i in range(50000):
        random_num = np.random.randint(0,19595)
        df_temp = df.iloc[random_num]
        if df_temp['disease_code'] == '缺少诊断' or df_temp['disease_code'] == 'DIP目录库不包含该诊断':
            continue
        else:
            df_patient.loc[index] = df_temp
            index = index+1
    df_patient.to_csv('patient2.csv')













