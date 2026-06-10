import numpy as np

class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    # Recebe o gradiente da camada seguinte (dA) e aplica a derivada da ReLU. 
    # Se o valor original self.Z era menor ou igual a zero, o gradiente é multiplicado por 0, se era maior que zero, é multiplicado por 1 (o sinal passa adiante).
    def backward(self, dA):
        return dA * (self.Z > 0)

# Para contemplar o requisito de comparar duas configurações diferentes, implementei a Função de Ativação sigmoide
class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        # sigmoide(X) = 1/(1+e^-X)
        # np.clip(Z, -500, 500) limita os valores de Z entre -500 e 500. Isso impede o erro de overflow do NumPy ao tentar calcular e^700.
        self.A = 1 / (1 + np.exp(-np.clip(Z, -500, 500))) 
        return self.A

    def backward(self, dA):
        return dA * (self.A * (1 - self.A))