import pandas as pd
from collections import Counter
from sklearn.model_selection import cross_val_score
import numpy as np


df = pd.read_csv('situacao_do_cliente.csv')
X_df = df[['recencia','frequencia','semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
# tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

def fit_and_predict(modelo, treino_dados, treino_marcacoes):
    k = 3

    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)

    name = modelo.__class__.__name__
    print("Taxa de acerto do algoritmo {0}: {1}".format(name, taxa_de_acerto))
    return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))

resultadoOneVsRest = fit_and_predict(modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))

resultadoOneVsOne = fit_and_predict(modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()

resultadoMultinomial = fit_and_predict(modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()

resultadoAdaBoost = fit_and_predict(modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    name = modelo.__class__.__name__
    print("Taxa de acerto do algoritmo ({0}) vencedor no mundo real: {1}".format(name, taxa_de_acerto))


maximo = max(resultados)
vencedor = resultados[maximo]

vencedor.fit(treino_dados, treino_marcacoes)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % (taxa_de_acerto_base))

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" %(total_de_elementos))

