# Aplicação de MLP 

## Como Rodar

**Dependências:**
- python 3.12.3
- numpy;
- matplotlib;
- tensorflow.

1) Para instalar as dependências supracitadas, rodar:
``pip install -r requirements.txt``

2) Para rodar a rede no terminal, rode:
``python main.py``

*Obs: também é possível rodar a aplicação no notebook no arquivo experimentos.py*

## Arquitetura Escolhida

Ao invés de criar uma classe única e engessada, a arquitetura foi inspirada em frameworks modernos (como o sub-módulo torch.nn do PyTorch). Essa foi uma demanda que senti ao implementar uma nova configuração de rede neural (mudando a função de ativação, o learning_rate e quantidade de camadas). Isso porque, no código passado, todas as operações matemáticas de todas as camadas estavam concentradas dentro de uma mesma classe MLP, de modo que, se houvesse a demanda de adicionar mais uma camada, seria necessário criar variáveis como as de pesos e vieses manualmente. Nessa perpectiva, modularizando o código, fiz com que cada etapa do MLP se tornasse uma classe independente com seus próprios métodos forward e backward.


Diante disso, o fluxo dinâmico dos dados configura-se em:

- **Forward Pass:** os dados de entrada entram na rede e cada camada processa o sinal e o passa para a camada seguinte. A última camada gera probabilidades e calcula o erro (loss).

- **Backward Pass:** o erro viaja no sentido oposto do supracitado. Cada camada recebe o impacto do erro da camada seguinte, calcula suas derivadas locais e repassa o sinal ajustado para a camada anterior.

## Resultados


## Decisões e Dificuldades

### Decisões e Dificuldades Matemáticas 

### He/Kaiming Initialization
Multipliquei a inicialização gaussiana por $\sqrt{2.0 / \text{input\_dim}}$ com base nos estudos que fiz, nos quais inferi que se os pesos começarem muito grandes, os sinais explodem e, se começarem muito pequenos, os neurônios morrem. Assim, essa inicialização garante que a variância das saídas de cada neurônio seja igual a 1, ideal para redes que usam ativação ReLU.

### Cache Interno (self.X)
Durante o forward, a camada obrigatoriamente armazena uma cópia da matriz de entrada X. Isso é necessário porque a derivada parcial da perda em relação aos pesos depende de $X$:$$\frac{\partial L}{\partial W} = \frac{1}{m} (dZ \cdot X^T)$$

### SoftmaxCrossEntropy

Individualmente, a derivada do Softmax é uma matriz Jacobiana complexa e pesada. No entanto, quando fundidas algebraicamente com a função de perda, as derivadas intermediárias se cancelam. Desse modo, forma-se a simplificação:
$$\frac{\partial L}{\partial Z} = A - Y$$

Implementei isso no código assim: ``return self.A - self.Y_one_hot``

Aqui, vale mencionar como dificuldade que, haja vista que a fórmula da entropia cruzada faz o cálculo de $\log(\hat{y})$, no caso da rede atribuir certeza de 0% para um número, temos que o cálculo de $\log(0)$ tende ao infinito negativo ($-\infty$). Sob essa perspectiva, para evitar que a perda virasse NaN, apliquei np.clip(self.A, 1e-15, 1.0). 

### Gradient_Check

Durante o desenvolvimento, senti dificuldade em garantir que a matemática que estava implementando no backward estava certa, haja vista que conforme estudei, uma simples troca de sinal pode prefudicar completamente o caminho da rede. Por isso, conforme conselho do professor, adicionei o gradient check.
Em contrapartida, ao implementar o gradient check, me deparei com o obstáculo do custo operacional, já que, para checar uma matriz de pesos de tamanho 128 por 784, a rede precisa fazer 128 x 784 x 2 (200.704) forward para recalcular a perda global para cada perturbação de e. Diante disso, tomei a decisão de criar uma lógica com apenas duas amostras de dados apenas para demonstração. 
Paralelamente, ao longo do desenvolvimento de testes, encontrei a dificuldade do underflow (quando o computador perde a precisão decimal) nos casos em que e foi bem pequeno por conta do expoente.

## Decisões e Dificuldades de Código

### Dimensões do NumPy 

No formato dos dados (features, exemplos), o viés b tem tamanho hidden_size igual a 1. Assim, quando o somei com a matriz de pontos, o NumPy repetiu o vetor horizontalmente para todos os exemplos. Para mitigar isso recuperando o gradiente de b, no backward, precisei somar as linhas horizontalmente e forçar o retorno da dimensão original usando keepdims=True. Dessa forma, evitei que o NumPy "achatasse" a matriz em um vetor unidimensional simples, e, consequentemente, quebrasse os cálculos futuros.

### 