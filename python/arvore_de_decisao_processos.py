# -*- coding: utf-8 -*-
"""
arvore_de_decisao-processos.py

Sávio Pessôa

3 classes:
    MAJORITARIAMENTE AGUARDANDO CONTENCIOSO
    MAJORITARIAMENTE EXONERADO    
    MAJORITARIAMENTE MANTIDO
"""

"""
!pip install pydotplus
!pip install dtreeviz
!pip install openpyxl
"""

"""
import os
os.chdir('c:\puc\python')
"""

#TODO: https://learn.co/lessons/dsc-decision-trees-with-sklearn-codealong
#TODO: https://www.google.com.br/search?q=python+example+DecisionTreeClassifier+with+categorical+attributes+OneHotEncoder
#TODO: https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# constantes
UP_ARROW = '\u2191'
DOWN_ARROW = '\u2193'
RIGHT_ARROW = '\u2192'

# colunas catégoricas que precisam de conversão
CAT_COLS = [#'cod_receita','cod_ua_prog', 
            'cod_ativ_fiscal', 'cod_motivacao',
            'cod_cnae_emp', 'uf']
COLS_TO_DROP = ['cod_receita', 'cod_ua_prog']

# parâmetros
CRITERION = 'entropy'
#CRITERION = 'gini'
START_CCP_ALPHA = 0
STEP_CCP_ALPHA = 100
MAX_BAD_CCP_ALPHAS = 10


def plotEfectiveAlphaImpurityOfLeaves(ccp_alphas, impurities, criterion):    
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas, impurities)
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title(f"Total Impurity vs effective alpha for training set (criterion={criterion})")
    plt.show()
    
    
def plot_nodes_depth_versus_alpha(clfs, tested_ccp_alphas):    
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(tested_ccp_alphas, node_counts)
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title(f"Number of nodes vs alpha (criterion={CRITERION})")
    ax[1].plot(tested_ccp_alphas, depth)
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title(f"Depth vs alpha (criterion={CRITERION})")
    fig.tight_layout()
    
    test_scores = [clf.score(X_test, y_test) for clf in clfs]    
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("acurácia")
    ax.set_title(f"Acurácia vs alpha (criterion={CRITERION})")
    ax.plot(tested_ccp_alphas, test_scores, label="test")
    ax.legend()
    plt.show()
    
    
"""
Poda da árvore de decisão usando ccp_alpha.
https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
"""    
def improve_accuracy(decision_tree):       
    acuracy_score = decision_tree.score(X_test, y_test)
    print(f"\nIniciando post pruning. Valor inicial da acurácia {acuracy_score:.4f}.")
    
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    plotEfectiveAlphaImpurityOfLeaves(ccp_alphas[:-1], impurities[:-1], CRITERION)
    plotEfectiveAlphaImpurityOfLeaves(ccp_alphas[:2000], impurities[:2000], CRITERION)
    
    clfs = [decision_tree]
    best_alpha = None # identificar melhor ccp_alpha
    best_score = acuracy_score # identificar melhor acurácia
    prev_ccp_alpha = 0 # evitar repetição de testes com mesmo ccp_alpha 
    tested_ccp_alphas = [0]
    bad_results = 0 # parar com acurácias em queda seguidas
    prev_score = 0
    below_start = False
    
    for ccp_alpha in ccp_alphas[START_CCP_ALPHA::STEP_CCP_ALPHA]:
        if ccp_alpha > prev_ccp_alpha:
            prev_ccp_alpha = ccp_alpha
            tested_ccp_alphas.append(ccp_alpha)
            clf = DecisionTreeClassifier(random_state=0, criterion=CRITERION, 
                                         ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
            score = clf.score(X_test, y_test)
            print(UP_ARROW if prev_score < score else 
                  DOWN_ARROW if prev_score > score else RIGHT_ARROW, end='')
            if (score >= best_score):
                best_score = score
                best_alpha = ccp_alpha
                bad_results = 0
                print(f' melhor acurácia={best_score:.4f} com alpha_min={best_alpha}')
            elif score != prev_score:
                if score >= acuracy_score:
                    bad_results = 0
                    if below_start:
                        print('(acima do ponto inicial)')
                        below_start = False
                elif (score < prev_score):
                    bad_results += 1
                    if not below_start:
                        print('(abaixo do ponto inicial)')
                        below_start = True
                    if (bad_results >= MAX_BAD_CCP_ALPHAS):
                        break
            prev_score = score
    print(f'\nA melhor poda é com alpha_min={best_alpha}, que produz acurácia de {(best_score):.4f}.')
    
    plot_nodes_depth_versus_alpha(clfs, tested_ccp_alphas)
    
    return best_alpha
    
        
def classify():
    print(f"\nIniciando classificação com árvore de decisão e criterion={CRITERION}.\n")
    decision_tree = DecisionTreeClassifier(random_state=0, criterion=CRITERION)
    decision_tree = decision_tree.fit(X_train, y_train)
    print("Acurácia na base de treinamento:", decision_tree.score(X_train, y_train))
    
    y_pred = decision_tree.predict(X_test)
    
    target_names=["MAJ. AGUARDANDO CONTENCIOSO", "MAJORITARIAMENTE EXONERADO", "MAJORITARIAMENTE MANTIDO"]
    print("Acurácia de previsão:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
          
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_table = pd.DataFrame(data=cnf_matrix, 
                             index=["MAJ. AG. CONT.", "MAJ. EXON.", "MAJ. MANT."], 
                             columns=["MAJ. AG. CONT. (prev)", "MAJ. EXON. (prev)", "MAJ. MANT. (prev)"])
    pd.options.display.max_columns = 4
    print(cnf_table)
    
    improve_accuracy(decision_tree)
    
    # Criar grafo resumido com 4 camadas 
    file_name = f'c:/PUC/DecisionTree-{CRITERION}-4-camadas.dot'
    tree.export_graphviz(decision_tree, 
                         out_file=file_name,
                         feature_names=X_train.columns,
                         proportion=False,
                         rounded =True,
                         filled=True, 
                         special_characters=True,
                         class_names=['Maj. Ag. Cont.', 'Maj. Exonerado', 'Maj. Mantido'], 
                         max_depth=4)    
    """
    # Draw graph
    !dot -x -Tsvg decisiontree.dot > decision_tree.svg
    """     
    return decision_tree


start = timer()

processos = pd.read_excel('../dataset/processos_2014_2015_anon_opf.xlsx', sheet_name=0) 
if (len(COLS_TO_DROP) > 0):
    processos.drop(columns=COLS_TO_DROP)
    print(f"\nTestes sem a presença da(s) coluna(s) {COLS_TO_DROP}")
print("\nDimensões do dataset: {0}".format(processos.shape))


X = processos.iloc[:,0:(processos.shape[1] - 1)]
X_dummies = pd.get_dummies(X, columns=CAT_COLS)

le = LabelEncoder()
y = le.fit_transform(processos.iloc[:,(processos.shape[1] - 1)])

# Particionar a base de dados
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, random_state=4000, test_size=0.3, stratify=y)
print(f"\nDimensões após conversão dos atributos categóricos: treinamento {X_train.shape} e teste {X_test.shape}")

decision_tree = classify()

end = timer()
elapsed_time = end - start
print(f'\nTempo de execução {elapsed_time//60:.0f} minutos e {elapsed_time%60:.0f} segundos.')

