# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Importando dados para o estudo
data = pd.read_csv('expec_vida.csv')

# Separação das variáveis independentes (Renda per Cápita (US$), PIB (US$ bilhões), População (milhões)) e dependentes (Expectativa de Vida (anos))
x = data[['Renda per Cápita (US$)', 'PIB (US$ bilhões)', 'População (milhões)']].values
y = data['Expectativa de Vida (anos)'].values

# Salvando os países para futuramente colocar na previsão
countries = data['País'].values

# Divisao dos dados em treino e teste
x_train, x_test, y_train, y_test, countries_train, countries_test = train_test_split(x, y, countries, test_size=5, random_state=2)

# Verificando o tamanho dos conjuntos
print(f"Conjunto de treino: {len(x_train)}")
print(f"Conjunto de teste: {len(x_test)}")

# Adicionar uma coluna com o valor "1" para o termo de intercepto
x_train_intercepto = np.c_[np.ones(x_train.shape[0]), x_train]
x_test_intercepto = np.c_[np.ones(x_test.shape[0]), x_test]

# Calculo  dos coeficientes beta usando a fórmula dos mínimos quadrados
beta = np.linalg.inv(x_train_intercepto.T @ x_train_intercepto) @ x_train_intercepto.T @ y_train

# Fazendo as previsões no conjunto de teste
y_pred = x_test_intercepto @ beta

# Calculo do erro quadrático médio
mse = np.mean((y_test - y_pred) ** 2)

# Exibindo resultados
print("beta:", beta)
print("mse:", mse)

# Valores reais vs previstos
previsao = pd.DataFrame({'País': countries_test, 'Real': y_test, 'Previsto': y_pred})

print(previsao)