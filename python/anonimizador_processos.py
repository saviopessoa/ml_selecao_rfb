# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:33:45 2021

@author: pessoa
"""

import pandas as pd
import random

processos = pd.read_excel('../dataset/processos_2014_2015.xlsx', sheet_name=0)
operacoes = pd.read_excel('../dataset/operacoes_2014_2015.xlsx', sheet_name=0)

keys_to_anonymize = ['num_processo', 'cod_receita']
cols_to_anonymize = ['cod_oper_fiscal', 'cod_ua_exec', 
                      'cod_ua_prog', 'cod_ativ_fiscal', 
                      'cod_motivacao', 'cod_cnae_emp', 'uf'] + keys_to_anonymize 

for ncol,col in enumerate(processos.columns):
    if (col in cols_to_anonymize):
        unique_values = processos.loc[:,col].unique()
        print(f'Coluna {col} com {len(unique_values)} valores únicos.')
        if any(x in unique_values for x in range(1,len(unique_values)+1)):
            processos[col] = processos[col] + len(unique_values)+1
            unique_values = processos.loc[:,col].unique()                        
            print(f'Novos valores presentes na coluna {col}.')
            print(processos.loc[:,col])
            if (col in keys_to_anonymize):
                operacoes[col] = operacoes[col] + len(unique_values)+1
        random.shuffle(unique_values)
        for i, value in enumerate(unique_values):
            processos.loc[processos[col] == value, col] = i + 1
            if (col in keys_to_anonymize):
                operacoes.loc[operacoes[col] == value, col] = i + 1

col = 'cod_oper_fiscal'
unique_values = operacoes.loc[:,col].unique()
print(f'Coluna {col} com {len(unique_values)} valores únicos.')
random.shuffle(unique_values)
for i, value in enumerate(unique_values):
    operacoes.loc[operacoes[col] == value, col] = i + 1            

ct_max = processos['ct_total'].max()
processos['ct_total'] /= ct_max
    
processos.to_excel('../dataset/processos_2014_2015_anon.xlsx')
operacoes.to_excel('../dataset/operacoes_2014_2015_anon.xlsx')