import numpy as np
from .losses import SoftmaxCrossEntropy

# Arquitetura Modular

# Permite que eu crie várias camadas sem precisar inicializar variáveis de peso, vieses e derivadas indivualmente.
class DenseLayer:
   
    def __init__(self, input_dim, output_dim):
        # Inicialização He / Kaiming para ReLU
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))
        self.X = None
        self.dW = None
        self.db = None

    # Z = W * X + b
    def forward(self, X):
        self.X = X
        return np.dot(self.W, X) + self.b

    # recebe o gradiente acumulado das camadas da frente e calcula o gradiente interno 
    def backward(self, dA_or_dZ):
        # dA_or_dZ é o gradiente vindo da camada seguinte
        m = self.X.shape[1]
        self.dW = (1 / m) * np.dot(dA_or_dZ, self.X.T)
        self.db = (1 / m) * np.sum(dA_or_dZ, axis=1, keepdims=True)
        # Retorna o gradiente em relação à entrada da camada (X) para a camada anterior
        # Permite o efeito cascata do backpropagation.
        return np.dot(self.W.T, dA_or_dZ)

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


# Classe da Rede Neural

# Permite criar uma lista sequencial de objetos de camadas usando .add()
# Fiz isso inspirada no Keras
class NeuralNetwork:
  
    def __init__(self):
        self.layers = []
        self.loss_layer = SoftmaxCrossEntropy()

    def add(self, layer):
        self.layers = list(self.layers) + [layer]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return self.loss_layer.forward(out)

    def backward(self):
        # Inicia o backpropagation a partir da camada de perda
        gradient = self.loss_layer.backward()
        # Garante que o gradiente comece na última camada e vá de volta até a primeira
        for layer in reversed(self.layers):
            # gradient é atualizada a cada passo
            gradient = layer.backward(gradient)

    # Percorre todas as camadas da rede interessando as que tenham pesos e vieses
    def update_weights(self, lr):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.update(lr)

# Gradient Check
def check_gradients(network, X, Y_oh, epsilon=1e-7):
    # Gradiente Analítico
    network.forward(X)
    network.loss_layer.compute_loss(Y_oh) # Define o Y_oh interno para o backward
    network.backward()
    
    target_layer = [l for l in network.layers if isinstance(l, DenseLayer)][0]
    row, col = 0, 400 
    analytical_grad = target_layer.dW[row, col]
    original_weight = target_layer.W[row, col]

    # Gradiente Numérico
    # f(x + ε)
    target_layer.W[row, col] = original_weight + epsilon
    network.forward(X) # Atualiza as ativações internas
    loss_plus = network.loss_layer.compute_loss(Y_oh) # Usa o Y_oh CORRETO

    # f(x - ε)
    target_layer.W[row, col] = original_weight - epsilon
    network.forward(X) # Atualiza as ativações internas
    loss_minus = network.loss_layer.compute_loss(Y_oh) # Usa o Y_oh CORRETO

    # Reset do peso
    target_layer.W[row, col] = original_weight
    
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    diff = np.abs(analytical_grad - numerical_grad)
    status = "CORRETO" if diff < 1e-5 else "ERRO"
    
    print(f"\n Gradient Check:")
    print(f"Pixel {col} | Input Médio: {np.mean(X[col, :]):.4f}")
    print(f"Analítico: {analytical_grad:.10f}")
    print(f"Numérico:  {numerical_grad:.10f}")
    print(f"Diferença: {diff:.2e} -> {status}")
    return diff < 1e-5

# Treinamento e Histórico

def train_network(network, X_train, Y_train_oh, Y_train_raw, X_test, Y_test_raw, epochs=15, batch_size=64, lr=0.1):
    num_samples = X_train.shape[1]
    history = {'loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # SGD com Shuffling (para embaralhar os dados)
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[:, permutation]
        Y_shuffled_oh = Y_train_oh[:, permutation]
        
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            X_batch = X_shuffled[:, i:end]
            Y_batch = Y_shuffled_oh[:, i:end]
            
            # Forward Pass
            network.forward(X_batch)
            # Calcula perda interna
            network.loss_layer.compute_loss(Y_batch)
            # Backpropagation
            network.backward()
            # Atualização com SGD
            network.update_weights(lr)
            
        # Avaliação da época
        train_out = network.forward(X_train)
        loss = network.loss_layer.compute_loss(Y_train_oh)
        
        train_preds = np.argmax(train_out, axis=0)
        train_acc = np.sum(train_preds == Y_train_raw) / num_samples * 100
        
        test_out = network.forward(X_test)
        test_preds = np.argmax(test_out, axis=0)
        test_acc = np.sum(test_preds == Y_test_raw) / X_test.shape[1] * 100
        
        history['loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Época {epoch+1:02d} | Perda: {loss:.4f} | Acc Treino: {train_acc:.2f}% | Acc Teste: {test_acc:.2f}%")
        
    return history