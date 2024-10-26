import numpy as np

def tanh(x):
    """
    Calcula a função tangente hiperbólica (tanh) de x.
    """
    return np.tanh(x)

def tanh_derivada(x):
    """
    Calcula a derivada da função tangente hiperbólica (tanh) de x.
    """
    return 1 - np.tanh(x) ** 2

class Neuronio:
    def __init__(self, d, phi):
        """
        Inicialização de um neuronio com parâmetros específicos.
        """
        self.d = d  # Armazena o número de entradas
        self.phi = phi  # Armazena a função de ativação
        self.w = np.random.uniform(-1, 1, d)  # Inicializa os pesos aleatoriamente entre -1 e 1
        self.b = np.random.uniform(-1, 1)  # Inicializa o bias aleatoriamente entre -1 e 1
        self.delta_w = np.zeros(d)  # Inicializa as mudanças nos pesos como zero
        self.delta_b = 0  # Inicializa a mudança no bias como zero
        self.h = 0  # Armazena a soma ponderada das entradas
        self.y = 0  # Armazena a saída do neuronio
        self.y_derivada = 0  # Armazena a derivada da saída em relação à entrada

    def propagar(self, x):
        """
        Realiza a propagação da entrada através do neuronio.

        Retorna:
        A saída do neuronio após aplicar a função de ativação.
        """
        self.h = np.dot(self.w, x) + self.b  # Calcula a soma ponderada das entradas e o bias
        self.y = self.phi(self.h)  # Aplica a função de ativação à soma ponderada
        self.y_derivada = tanh_derivada(self.h)  # Calcula a derivada da função de ativação
        return self.y  # Retorna a saída do neuronio

    def adaptar(self, delta, y_n_1, alfa, beta):
        """
        Adapta os pesos e o bias do neuronio com base no erro.
        """
        self.M_w = beta * self.delta_w  # Calcula a acelaração para os pesos
        self.delta_w = -alfa * self.y_derivada * delta * np.array(y_n_1) + self.M_w  # Calcula a nova mudança nos pesos
        self.w += self.delta_w  # Atualiza os pesos
        self.M_b = beta * self.delta_b  # Calcula a acelaração para o bias
        self.delta_b = -alfa * self.y_derivada * delta + self.M_b  # Calcula a nova mudança no bias
        self.b += self.delta_b  # Atualiza o bias

class Camada_Densa:
    def __init__(self, d_e, d_s, phi):
        """
        Inicializa uma camada densa com neuronios.
        """
        self.d_s = d_s  # Armazena o número de neuronios na camada
        self.neuronios = [Neuronio(d_e, phi) for _ in range(d_s)]  # Inicializa a lista de neuronios, criando d_s neuronios com d_e entradas e a função de ativação phi

    @property
    def y(self):
        """
        Retorna as saídas de todos os neuronios na camada. (lista)
        """
        return [neuronio.y for neuronio in self.neuronios]  # Lista com as saídas dos neuronios

    def propagar(self, x):
        """
        Realiza a propagação das entradas através da camada.

        Retorna:
        Uma lista das saídas de todos os neuronios na camada.
        """
        y = [neuronio.propagar(x) for neuronio in self.neuronios]  # Propaga a entrada x através de cada neuronio
        return y  # Retorna as saídas da camada

    def adaptar(self, delta_n, y_n_1, alfa, beta):
        """
        Adapta os pesos e o bias de todos os neuronios na camada com base no erro.
        """
        for j in range(self.d_s):
            # Adapta cada neuronio na camada com o seu respectivo erro
            self.neuronios[j].adaptar(delta_n[j], y_n_1, alfa, beta)


class Camada_Entrada:
    def __init__(self, d_s):
        """
        Inicializa uma camada de entrada.
        """
        self.d_s = d_s  # Armazena o número de saídas da camada de entrada
        self.y = [0 for _ in range(d_s)]  # Inicializa a lista de saídas com zeros

    def propagar(self, x):
        """
        Propaga a entrada através da camada de entrada.

        Retorna:
        A entrada recebida x, que agora é armazenada como saída da camada.
        """
        self.y = x  # Armazena a entrada x como saída da camada
        return self.y  # Retorna a saída da camada (que é a mesma que a entrada)


class Rede_Neuronal:
    def __init__(self, forma, phi):
        """
        Inicializa uma rede neuronal.
        """
        self.camadas = []  # Inicializa a lista de camadas da rede
        self.N = len(forma)  # Armazena o número total de camadas
        d_1_s = forma[0]  # Número de saídas da camada de entrada
        camada_1 = Camada_Entrada(d_1_s)  # Cria a camada de entrada
        self.camadas.append(camada_1)  # Adiciona a camada de entrada à lista de camadas

        # Cria as camadas densas e adiciona-as à rede
        for n in range(1, self.N):
            d_e = forma[n - 1]  # Número de entradas da camada atual
            d_s = forma[n]  # Número de saídas da camada atual
            camada = Camada_Densa(d_e, d_s, phi)  # Cria uma nova camada densa
            self.camadas.append(camada)  # Adiciona a camada à lista de camadas

    def delta_saida(self, y_n, y):
        """
        Calcula o erro entre a saída prevista e a saída real.

        Retorna:
        A lista de erros para cada neuronio na camada de saída.
        """
        return [y_n[k] - y[k] for k in range(len(y))]  # Calcula a diferença entre as saídas

    def retropropagar(self, delta_n, alfa, beta):
        """
        Realiza a retropropagação do erro para atualizar os pesos da rede.
        """
        delta = delta_n  # Inicializa o delta com o erro da saída
        for n in range(self.N - 1, 0, -1):
            y_n_1 = self.camadas[n - 1].y  # Obtém a saída da camada anterior
            self.camadas[n].adaptar(delta, y_n_1, alfa, beta)  # Adapta os pesos da camada atual
            d_n_1 = self.camadas[n - 1].d_s  # Número de saídas da camada anterior
            d_n = self.camadas[n].d_s  # Número de saídas da camada atual
            neuronios = self.camadas[n].neuronios  # Obtém os neuronios da camada atual

            # Calcula o erro da camada anterior
            delta = [
                sum(neuronios[j].w[i] * delta[j] * neuronios[j].y_derivada for j in range(d_n))
                for i in range(d_n_1)
            ]

    def adaptar(self, x, y, alfa, beta):
        """
        Adapta a rede neuronal com base na entrada e saída desejada.

        Retorna:
        O valor do erro quadrático médio.
        """
        y_n = self.propagar(x)  # Propaga a entrada x através da rede
        delta_n = self.delta_saida(y_n, y)  # Calcula o erro na saída
        self.retropropagar(delta_n, alfa, beta)  # Retropropaga o erro
        K = len(delta_n)  # Número de neuronios na camada de saída
        epsilon = (1 / K) * sum(delta_n) ** 2  # Calcula o erro quadrático médio
        return epsilon  # Retorna o erro

    def teinar(self, X, Y, n_epocas, epsilon_max, alfa, beta):
        """
        Treina a rede neuronal com um conjunto de dados.
        """
        for _ in range(n_epocas):
            epsilon = 0  # Inicializa o erro máximo para esta época
            for x, y in zip(X, Y):  # Itera sobre cada entrada e saída
                epsilon_x = self.adaptar(x, y, alfa, beta)  # Adapta a rede
                epsilon = max(epsilon, epsilon_x)  # Atualiza o erro máximo
            if epsilon <= epsilon_max:  # Para o treino se o erro for aceitável
                break

    def prever(self, X):
        """
        Faz previsões com a rede neuronal para um conjunto de entradas.

        Retorna:
        A lista de saídas previstas para cada entrada.
        """
        Y = [self.propagar(x) for x in X]  # Propaga cada entrada e obtém a saída prevista
        return Y  # Retorna as saídas previstas

    def propagar(self, x):
        """
        Propaga a entrada através de todas as camadas da rede.

        Retorna:
        A saída final da rede após a propagação.
        """
        y = x  # Inicializa y com a entrada
        for camada in self.camadas:  # Itera sobre cada camada
            y = camada.propagar(y)  # Propaga a saída da camada anterior
        return y  # Retorna a saída final da rede


"""
### TESTE XOR ###
# Definição os dados das entrada e saídas para o problema XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

# Inicialização da rede neuronal
forma = [2, 2, 1]  # 2 entradas, 2 neuronios na camada oculta, 1 saída
phi = tanh  # Função de ativação

rede = Rede_Neuronal(forma, phi)

# Treino da rede neuronal
n_epocas = 10000
epsilon_max = 0.01
alfa = 0.1  # Taxa de aprendizagem
beta = 0.9  # Acelaração

rede.teinar(X, Y, n_epocas, epsilon_max, alfa, beta)

# Teste da rede neuronal
predicoes = rede.prever(X)

# Resultados
print("Saídas previstas:")
for i, pred in enumerate(predicoes):
    print(f"Entrada: {X[i]} - Saída prevista: {pred}")
"""
