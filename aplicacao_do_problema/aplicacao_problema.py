import numpy as np
import pickle

# Lista de tipos de triângulos
tipos = ["Equilátero", "Isósceles", "Escaleno"]

# Função para classificar o tipo de triângulo
def classificar_triangulo(lados):
    # Normalizar os lados para a rede
    lados_normalizados = lados / np.max(lados)
    predicoes = rede.prever([lados_normalizados])  # Lista com a saída da camada de saída
    tipo_indice = np.argmax(predicoes[0])  # Indice da maior probabilidade
    return tipos[tipo_indice]  # Retorna o tipo de triângulo

# Carrega a rede
with open('rede_neuronal_treinada.pkl', 'rb') as f:
    rede = pickle.load(f)

while True:
    # Obtem as entradas do utilizador
    lados = []
    for i in range(3):
        lado = float(input(f"Digite o comprimento do lado {i+1}: "))
        lados.append(lado)

    # Classificar o triângulo
    tipo_triangulo = classificar_triangulo(np.array(lados))
    print(f"O triângulo é do tipo: {tipo_triangulo}\n")
