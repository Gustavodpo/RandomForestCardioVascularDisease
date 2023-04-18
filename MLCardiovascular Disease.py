import numpy
import pandas as pd
import matplotlib.pyplot as plt

#tratar o separador e ler o dataset
df_cardio = pd.read_csv("cardio_train.csv", sep=";", index_col=0)
df_cardio

# Exploração dos Dados 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#---
fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Box(x=df_cardio["age"] / 365, name="Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio["weight"], name="Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_hi"], name="Pressão sanquínea sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_lo"], name="Pressão sanquínea diastólica"), row=4, col=1)


#Ver variáveis categoricas
fig = make_subplots(rows=2, cols=3)
fig.add_trace(go.Bar(y=df_cardio["gender"].value_counts(), x=["Feminino", "Masculino"], name="Genero"), row=1, col=1)
fig.add_trace(go.Bar(y=df_cardio["cholesterol"].value_counts(), x=["Normal", "Acima do Normal", "Muito Acima do Normal"], name="cholesterol"), row=1, col=3)
fig.add_trace(go.Bar(y=df_cardio["gluc"].value_counts(), x=["Normal", "Acima do Normal", "Muito Acima do Normal"], name="Glicose"), row=1, col=2)
fig.add_trace(go.Bar(y=df_cardio["smoke"].value_counts(), x=["Não Fumante", "Fumante"], name="Fumante"), row=2, col=1)
fig.add_trace(go.Bar(y=df_cardio["alco"].value_counts(), x=["Não Alcoólatra", "Alcoolatra"], name="Alcoólatra"), row=2, col=2)
fig.add_trace(go.Bar(y=df_cardio["active"].value_counts(), x=["Não Ativo", "Ativo"], name="Ativo"), row=2, col=3)


# ver a distribuição de Cardiacos/Não-cardiacos
df_cardio["cardio"].value_counts() / df_cardio["cardio"].value_counts().sum()
# 0=5003
# 1=4997


#Calcular a correlação entre todas as variáveis
import seaborn as sns
fig, ax = plt.subplots(figsize=(30, 10))
sns.heatmap(df_cardio.corr(), annot=True, cmap="RdYlGn")


# Criação do Modelo de Aprendizagem
Y = df_cardio["cardio"]
X = df_cardio.loc[:, df_cardio.columns != 'cardio']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

ml_model = RandomForestClassifier()
ml_model.fit(x_train, y_train)
RandomForestClassifier()

#testando a previsão do modelo teste
x_test.iloc[0]
ml_model.predict(x_test.iloc[0].values.reshape(1,-1))
#array([1]) neste caso, o modelo retorna como "1", que é confirmado para problema cardiaco


#Analisando a precisão com Matriz de confusão(predictions)
from sklearn.metrics import classification_report, confusion_matrix
predictions = ml_model.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Representando o Modelo com Shap
import shap
background = shap.sample(X, 200)
explainer = shap.TreeExplainer(ml_model, feature_names=X.columns, data=background)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values[1], X)

#Aplicar o treino para um individuo especifico  
n = 2 #<numero que exemplifica um individuo com menor probabilidade de problemas cardíacos
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[0][n], x_train.iloc[n], matplotlib=True)

#Aplicar o treino para um individuo especifico 
n = 4 #<numero que exemplifica um individuo com maior probabilidade de problemas cardíacos
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[0][n], x_train.iloc[n], matplotlib=True)


# gráfico de dispersão de dependência para mostrar o efeito de uma única variável em todo o conjunto de dados
# calculate SHAP values
explainer = shap.Explainer(ml_model)
shap_values = explainer(X)

# plotar SHAP values para idade e pressão
shap_values_array = numpy.array(shap_values)
shap.plots.scatter(shap_values_array[:, 2], color=shap_values_array)  # plotando a terceira coluna (age)
shap.plots.scatter(shap_values_array[:, 3], color=shap_values_array)  # plotando a quarta coluna (ap_hi)