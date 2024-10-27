import numpy as np
from rede_neuronal import Rede_Neuronal, tanh
import json
import pickle

# Função para calcular a perda
def cross_entropy(y_true, y_pred):
    # Adicione uma pequena constante para evitar log(0)
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Para evitar log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # Média da perda

# Função para calcular a precisão
def calcular_precisao(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=1)  #Classe com a maior probabilidade
    y_true_classes = np.argmax(y_true, axis=1)  # Classe verdadeira
    acertos = np.sum(y_pred_classes == y_true_classes)  # Conta acertos
    return acertos / len(y_true)  # Acertos

# Dados utilizados para o treino da rede
dados_treino = [
    ([1, 1, 1], [1, 0, 0]),  # Equilátero
    ([2, 2, 2], [1, 0, 0]),  # Equilátero
    ([3, 3, 3], [1, 0, 0]),  # Equilátero
    ([4, 4, 4], [1, 0, 0]),  # Equilátero
    ([5, 5, 5], [1, 0, 0]),  # Equilátero
    ([6, 6, 6], [1, 0, 0]),  # Equilátero
    ([7, 7, 7], [1, 0, 0]),  # Equilátero
    ([8, 8, 8], [1, 0, 0]),  # Equilátero
    ([9, 9, 9], [1, 0, 0]),  # Equilátero
    ([10, 10, 10], [1, 0, 0]),  # Equilátero
    ([2, 2, 3], [0, 1, 0]),  # Isósceles
    ([3, 3, 4], [0, 1, 0]),  # Isósceles
    ([4, 4, 5], [0, 1, 0]),  # Isósceles
    ([5, 5, 6], [0, 1, 0]),  # Isósceles
    ([6, 6, 7], [0, 1, 0]),  # Isósceles
    ([7, 7, 8], [0, 1, 0]),  # Isósceles
    ([8, 8, 9], [0, 1, 0]),  # Isósceles
    ([9, 9, 10], [0, 1, 0]),  # Isósceles
    ([2, 3, 4], [0, 0, 1]),  # Escaleno
    ([3, 4, 5], [0, 0, 1]),  # Escaleno
    ([5, 6, 7], [0, 0, 1]),  # Escaleno
    ([6, 7, 8], [0, 0, 1]),  # Escaleno
    ([7, 8, 9], [0, 0, 1]),  # Escaleno
    ([8, 9, 10], [0, 0, 1]),  # Escaleno
    ([10, 11, 12], [0, 0, 1]),  # Escaleno
    ([3, 5, 4], [0, 0, 1]),  # Escaleno
    ([4, 6, 5], [0, 0, 1]),  # Escaleno
    ([5, 7, 6], [0, 0, 1]),  # Escaleno
    ([6, 8, 7], [0, 0, 1]),  # Escaleno
    ([7, 9, 8], [0, 0, 1]),  # Escaleno
    ([8, 10, 9], [0, 0, 1]),  # Escaleno
    ([9, 11, 10], [0, 0, 1]),  # Escaleno
    ([4, 5, 4], [0, 1, 0]),  # Isósceles
    ([5, 6, 5], [0, 1, 0]),  # Isósceles
    ([6, 7, 6], [0, 1, 0]),  # Isósceles
    ([7, 8, 7], [0, 1, 0]),  # Isósceles
    ([8, 9, 8], [0, 1, 0]),  # Isósceles
    ([9, 10, 9], [0, 1, 0]),  # Isósceles
    ([10, 11, 10], [0, 1, 0]),  # Isósceles
    ([11, 12, 11], [0, 1, 0]),  # Isósceles
    ([12, 13, 12], [0, 1, 0]),  # Isósceles
    ([5, 12, 13], [0, 0, 1]),  # Escaleno
    ([5, 13, 12], [0, 0, 1]),  # Escaleno
    ([12, 13, 14], [0, 0, 1]),  # Escaleno
    ([14, 15, 16], [0, 0, 1]),  # Escaleno
    ([8, 8, 10], [0, 1, 0]),  # Isósceles
    ([10, 10, 8], [0, 1, 0]),  # Isósceles
    ([9, 9, 6], [0, 1, 0]),  # Isósceles
    ([8, 8, 5], [0, 1, 0]),  # Isósceles
    ([7, 7, 3], [0, 1, 0]),  # Isósceles
    ([6, 6, 4], [0, 1, 0]),  # Isósceles
    ([5, 5, 1], [0, 1, 0]),  # Isósceles
    ([11, 12, 14], [0, 0, 1]),  # Escaleno
    ([3, 4, 6], [0, 0, 1]),  # Escaleno
    ([5, 9, 10], [0, 0, 1]),  # Escaleno
    ([4, 5, 10], [0, 0, 1]),  # Escaleno
    ([1, 2, 2], [0, 1, 0]),  # Isósceles
    ([2, 2, 3], [0, 1, 0]),  # Isósceles
    ([5, 5, 7], [0, 1, 0]),  # Isósceles
    ([3, 3, 5], [0, 1, 0]),  # Isósceles
    ([7, 7, 8], [0, 1, 0]),  # Isósceles
    ([1, 1, 2], [0, 1, 0]),  # Isósceles
    ([6, 8, 10], [0, 0, 1]),  # Escaleno
    ([9, 12, 14], [0, 0, 1]),  # Escaleno
    ([5, 8, 11], [0, 0, 1]),  # Escaleno
    ([10, 11, 15], [0, 0, 1]),  # Escaleno
    ([7, 8, 15], [0, 0, 1]),  # Escaleno
    ([4, 4, 5], [0, 1, 0]),  # Isósceles
    ([3, 3, 5], [0, 1, 0]),  # Isósceles
    ([4, 4, 6], [0, 1, 0]),  # Isósceles
    ([1, 2, 3], [0, 0, 1]),  # Escaleno
    ([2, 3, 4], [0, 0, 1]),  # Escaleno
    ([1, 1, 1], [1, 0, 0]),  # Equilátero
    ([9, 9, 9], [1, 0, 0]),  # Equilátero
    ([2, 2, 2], [1, 0, 0]),  # Equilátero
]

X = np.array([item[0] for item in dados_treino])  # Entradas do conjunto de dados de treino
Y = np.array([item[1] for item in dados_treino])  # Saídas do conjunto de dados de treino
X = X / np.max(X)  # Normaliza os dados de entrada dividindo-os pelo valor máximo

# Definição da rede neuronal
forma = [3, 5, 3]  # 3 entradas, 5 neurônios na camada oculta, 3 saídas
rede = Rede_Neuronal(forma, tanh)  # Função de ativação tanh

# Parametros de treino
n_epocas = 30000  # Número total de épocas para o treino da rede neural
epsilon_max = 0.002  # Valor máximo de epsilon
alfa = 0.0005  # Taxa de aprendizagem
beta = 0.9  # Fator de aceleração

# Treinar a rede com os dados dos triângulos
rede.teinar(X, Y, n_epocas, epsilon_max, alfa, beta)

# Testar a rede
previsoes = rede.prever(X)

# Cálculo da perda
perda = cross_entropy(Y, previsoes)

# Calcular precisão
precisao = calcular_precisao(Y, previsoes)

# Avaliação
avaliacao = {
    "precisao": precisao,
    "perda": perda
}

# Resultados
print("Saídas previstas (classificação de triângulos):")
for i, pred in enumerate(previsoes):
    saida_binaria = [round(p) for p in pred]  # Arredonda para obter 0 ou 1
    print(f"Entrada: {X[i]} - Saída prevista: {saida_binaria}")

# Guardar a avaliação num ficheiro JSON
with open('avaliacao_rede.json', 'w') as f:
    json.dump(avaliacao, f, indent=4)

# Guardar a rede num ficheiro pkl para a usar mais tarde na aplicação do problema
with open("rede_neuronal_treinada.pkl", "wb") as f:
    pickle.dump(rede, f)