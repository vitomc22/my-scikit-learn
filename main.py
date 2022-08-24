import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

# conjunto de dados de cancer de mama
breast = load_breast_cancer()

# dataframe para visualizar os dados
breast_data = pd.DataFrame(breast.data, columns=breast.feature_names)
breast_data.describe().transpose()

# criar grafico com searborn e matplotlib
plt.figure(figsize=(8, 8))
atrib_medidas = breast_data.columns[1:11]
m_corr = breast_data[atrib_medidas].corr()
sns.heatmap(m_corr, cmap='Blues', annot=True, square=True)

# Criar um DataFrame para os rótulos e associá-los com as classes.

labels = pd.DataFrame(breast.target, columns=['Tipo de cancêr'])
labels_class = labels['Tipo de cancêr'].map({0: 'maligno', 1: 'binigno'})

# A seguir, iremos observar a distribuição dos valores de classe no gráfico de barras:
plt.figure(figsize=(6, 6))
plt.tick_params(labelbottom=False)
sns.countplot(data=labels, x='Tipo de cancêr', hue=labels_class, palette=sns.color_palette("RdBu", 2))

# vamos selecionar os últimos 3 atributos do conjunto de dados e plotar os relacionamentos emparelhados do conjunto de dados

breast_data["Tipo de cancêr"] = labels
atrib_3ultimos = breast_data.columns[27:31]
plt.figure(figsize=(6, 6))
sns.pairplot(breast_data[atrib_3ultimos], hue="Tipo de cancêr", palette="husl", markers=['o', 'd'])

# realizar a normalização
# método de transformação nos dados selecionados (excetoo atributo-alvo):

stdScaler = StandardScaler()
atributos_treinamento = breast_data.columns[0:30]
breast_data[atributos_treinamento] = stdScaler.fit_transform(breast_data[atributos_treinamento])

# boas práticas dizem que você deve empregar a validação cruzada com 10 subconjuntos e 10 repetições na avaliação do modelo.
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
# Armazena os resultados de todas as iterações
accuracies = []
precisions = []
recalls = []
# verificar e experimentar os algoritmos

for train_idx, test_idx in rkf.split(breast_data):
    # Divide o conjunto em treinamento/teste
    X_train, X_test = breast_data.iloc[train_idx, :-1], breast_data.iloc[test_idx, :-1]
    y_train, y_test = breast_data.iloc[train_idx, -1], breast_data.iloc[test_idx, -1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # prediz os rótulos no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcula a acurácia, a precisão e a sensibiidade
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    prec = precision_score(y_test, y_pred)
    precisions.append(prec)

    rec = recall_score(y_test, y_pred)
    recalls.append(rec)

print("Regressão logistica")

acuracia = "The acuracia is {:.3f} +- {:.3f} percent."
print(acuracia.format(np.mean(accuracies), np.std(accuracies)))

precisao = "The precision is {:.3f} +- {:.3f} percent."
print(precisao.format(np.mean(precisions), np.std(precisions)))

sensibilidade = "The recall is {:.3f} +- {:.3f} percent.\n"
print(sensibilidade.format(np.mean(recalls), np.std(recalls)))

# Com os hiperparâmetros encontrados e definidos no modelo de Regressão Logística,

model = LogisticRegression(C=0.052, penalty='l2', solver='liblinear')

model.fit(X_train, y_train)
# prediz os rótulos no conjunto de teste
y_pred = model.predict(X_test)

# Calcula a acurácia, a precisão e a sensibiidade
acc = accuracy_score(y_test, y_pred)
accuracies.append(acc)

prec = precision_score(y_test, y_pred)
precisions.append(prec)

rec = recall_score(y_test, y_pred)
recalls.append(rec)

print("Regressão logistica otimizada")

acuracia = "The acuracia is {:.3f} +- {:.3f} percent."
print(acuracia.format(np.mean(accuracies), np.std(accuracies)))

precisao = "The precision is {:.3f} +- {:.3f} percent."
print(precisao.format(np.mean(precisions), np.std(precisions)))

sensibilidade = "The recall is {:.3f} +- {:.3f} percent."
print(sensibilidade.format(np.mean(recalls), np.std(recalls)))
