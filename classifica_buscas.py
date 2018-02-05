import pandas as pd

df = pd.read_csv('buscas.csv')
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

acerto_de_um = len(Y[Y=='sim'])
acerto_de_zero = len(X[Y=='nao'])
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(Y)
print("Taxa de acerto base: %f" % (taxa_de_acerto_base))

porcentagem_treino = 0.9

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

acertos = resultado == teste_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
 
print(taxa_de_acerto)
print(total_de_elementos) 