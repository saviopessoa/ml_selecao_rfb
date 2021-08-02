# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:51:21 2021

Concatena as operacoes fiscais como atributos com 0s e 1s. 
O processo é custoso, mas após salvar as informações, não
é mais necessário executar.

@author: pessoa
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer


# colunas iniciais do dataset
CAT_COLS = ['cod_receita', 
            'cod_ua_prog', 
            'cod_ativ_fiscal', 'cod_motivacao',
            'cod_cnae_emp', 'uf']
CONT_COLS = ['diferenciado', 'especial', 'ct_total', 'situacao']

PROC_FILE_PATH = '../dataset/processos_2014_2015_anon.xlsx'
OPER_FILE_PATH = '../dataset/operacoes_2014_2015_anon.xlsx'
OUT_FILE_PATH = '../dataset/processos_2014_2015_anon_opf.csv'


"""
import os
os.chdir('c:\puc\python')
"""

start = timer()

# Carregar operacoes processos e operacoes
processos = pd.read_excel(PROC_FILE_PATH, sheet_name=0) 
processos = processos.loc[:, ['num_processo'] + CAT_COLS + CONT_COLS] 
operacoes = pd.read_excel(OPER_FILE_PATH, sheet_name=0)
operacoes = operacoes.loc[:,['num_processo','cod_receita','cod_oper_fiscal']]

# Incluir coluna de índice
operacoes['proc_rec'] = operacoes['num_processo'].astype(str) + "_" + operacoes['cod_receita'].astype(str)
operacoes = operacoes.drop(columns=['num_processo','cod_receita'])

# Gerar dataframe zerado com as colunas dos 0s e 1s
op_dummies = pd.DataFrame(int(0), index=np.arange(processos.shape[0]), 
                          columns=pd.get_dummies(operacoes, columns=['cod_oper_fiscal']).columns)
op_dummies['proc_rec'] = processos['num_processo'].astype(str) + "_" + processos['cod_receita'].astype(str)

# Setar os uns do dataframe 
keys = set(op_dummies['proc_rec'])
not_found = 0
for index, row in operacoes.iterrows():
    proc_rec = row['proc_rec']
    if proc_rec in keys:
        col_dummy = 'cod_oper_fiscal_' + str(row['cod_oper_fiscal'])    
        op_dummies.loc[op_dummies['proc_rec']==proc_rec, col_dummy] = int(1)
    else:
        not_found += 1
    if index % 1000 == 0:
        print(index, end="\t")
print(f'{not_found} registros das operações não foram encontrados nos processos')

# Concatenar e salvar o dataframe
proc_opf = pd.concat([op_dummies, processos], axis=1)
proc_opf.to_csv(OUT_FILE_PATH)
print(f"Processos salvos com as operações fiscais em {OUT_FILE_PATH}.")

end = timer()
elapsed_time = end - start
print(f'\nTempo de execução {elapsed_time//60:.0f} minutos e {elapsed_time%60:.0f} segundos.')

