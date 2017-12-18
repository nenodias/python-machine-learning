from sklearn.naive_bayes import MultinomialNB
import numpy as np

# [é gordinho, tem perna curta, faz auaua]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]

# 1 é porco e -1 é cachcorro
marcacoes = [1, 1, 1, -1, -1, -1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]

teste = [misterioso1, misterioso2]

print(modelo.predict(teste))
