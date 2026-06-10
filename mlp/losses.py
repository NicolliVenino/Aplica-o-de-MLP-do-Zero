import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.A = None
        self.Y_one_hot = None

    def forward(self, Z):
        # Subtração do max para estabilidade numérica
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.A

    # Implementa a fórmula matemática da Entropia Cruzada Multiclasse 
    # L = -1/m somatorio Y * log(Ŷ)
    def compute_loss(self, Y_one_hot):
        self.Y_one_hot = Y_one_hot
        m = Y_one_hot.shape[1]
        # Evita log(0) adicionando um epsilon minúsculo
        eps = 1e-15
        loss = - (1 / m) * np.sum(Y_one_hot * np.log(np.clip(self.A, eps, 1.0))) # Vale mencionar que inseri np.clip pois, antes de adicionar, a rede tentava calcular log(0), o que resultava em -inf 
        return loss

    def backward(self):
        # Inicia o processo de backpropagation.
        # A derivada conjunta da entropia cruzada com o Softmax resulta na diferença simples entre as previsões da rede (self.A) e os alvos reais (self.Y_one_hot).
        return self.A - self.Y_one_hot