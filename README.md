# Projeto MLP em C++

Este projeto implementa uma Rede Neural Perceptron Multicamadas (MLP) em C++ utilizando templates e functors, conforme os requisitos especificados. A implementação é flexível, permitindo a configuração de diferentes tipos numéricos, funções de ativação e funções de custo.

## Estrutura do Projeto

O projeto está organizado nos seguintes arquivos:

- `activation_functions.h`: Contém as implementações das funções de ativação (Sigmoid, ReLU, Tanh) como functors.
- `loss_functions.h`: Contém a implementação da função de custo (MSE - Mean Squared Error) como functor.
- `layer.h`: Define a classe `Layer`, que representa uma camada da rede neural. Utiliza templates para o tipo numérico e a função de ativação.
- `mlp.h`: Define a classe `MLP`, a classe principal da rede neural. Gerencia as camadas, o forward pass e o backpropagation. Utiliza templates para o tipo numérico, funções de ativação das camadas ocultas e de saída, e a função de custo.
- `main.cpp`: Contém o exemplo de uso da MLP para resolver o problema XOR, incluindo o treinamento e a exibição dos resultados.
- `utils.h`: (Atualmente vazio, pode ser usado para funções utilitárias futuras).

## Requisitos de Compilação

Para compilar este projeto, você precisará de um compilador C++ compatível com C++17 ou superior (ex: g++).

## Como Compilar e Executar

1. Navegue até o diretório `mlp_project` no terminal:
   ```bash
   cd mlp_project
   ```

2. Compile o projeto usando o g++:
   ```bash
   g++ -std=c++17 main.cpp -o mlp_xor
   ```

3. Execute o programa:
   ```bash
   ./mlp_xor
   ```

## Detalhes da Implementação

### Uso de Templates

Templates são amplamente utilizados para garantir a flexibilidade do projeto:

- **Tipo Numérico (`T`)**: Permite que a rede opere com `float` ou `double` (ou outros tipos numéricos) sem modificações no código principal. Isso é definido nas classes `Layer` e `MLP`, bem como nos functors de ativação e custo.
- **Funções de Ativação**: As funções de ativação são passadas como parâmetros template para a classe `Layer` e `MLP`. Isso permite que diferentes funções de ativação sejam usadas para camadas ocultas e de saída, e que novas funções de ativação sejam facilmente adicionadas.
- **Funções de Custo**: A função de custo é passada como parâmetro template para a classe `MLP`, permitindo a troca fácil entre diferentes métricas de erro.

### Uso de Functors

As funções de ativação (`Sigmoid`, `ReLU`, `Tanh`) e a função de custo (`MSE`) são implementadas como **functors**. Um functor é um objeto que pode ser chamado como uma função (através da sobrecarga do operador `()`). Isso oferece uma maneira elegante e flexível de passar funções como parâmetros para templates, permitindo que elas carreguem estado se necessário (embora neste projeto não seja o caso).

Cada functor de função de ativação também inclui um método `derivative()` para calcular a derivada da função, essencial para o algoritmo de backpropagation.

### Arquitetura da MLP

A `MLP` é construída com uma configuração de camadas (`std::vector<size_t> layers_config`), onde o primeiro elemento é o número de entradas, os elementos intermediários são o número de neurônios nas camadas ocultas, e o último elemento é o número de saídas.

- **Inicialização de Pesos**: Os pesos são inicializados aleatoriamente usando uma distribuição uniforme, com uma escala baseada na inicialização de Xavier/Glorot para ajudar na convergência do treinamento.
- **Forward Pass**: Calcula a saída da rede para uma dada entrada, propagando os valores através de cada camada e aplicando as funções de ativação.
- **Backpropagation**: Implementa o algoritmo de retropropagação para calcular os gradientes dos pesos e bias em relação à função de custo. O erro é propagado da camada de saída de volta para as camadas ocultas.
- **Atualização de Pesos**: Os pesos e bias são atualizados usando o gradiente descendente com uma taxa de aprendizado configurável.

## Exemplo (Problema XOR)

O `main.cpp` demonstra o treinamento da MLP para resolver o problema XOR, um problema clássico de classificação não linear. A rede é treinada por um número de épocas, e o erro médio é exibido periodicamente. Ao final do treinamento, as previsões da rede para as entradas XOR são mostradas, demonstrando a capacidade da MLP de aprender o padrão.

```cpp
// Exemplo de configuração da rede (2 entradas, 4 neurônios ocultos, 1 saída)
std::vector<size_t> config = {2, 4, 1};

// Instanciação da MLP com tipos e funções de ativação/custo específicos
MLP<double, Tanh<double>, Sigmoid<double>, MSE<double>> my_mlp(config);

// Dados de treinamento XOR
std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};
std::vector<std::vector<double>> targets = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
};

// Treinamento
int epochs = 10000;
double learning_rate = 0.1;
for (int epoch = 0; epoch <= epochs; ++epoch) {
    // ... (código de treinamento e cálculo de erro)
}

// Resultados
for (size_t i = 0; i < inputs.size(); ++i) {
    auto output = my_mlp.forward(inputs[i]);
    // ... (exibição de entrada, alvo e predição)
}
```

## Próximos Passos (Melhorias Opcionais)

- **Diferentes Estratégias de Inicialização**: Implementar outros métodos de inicialização de pesos (ex: He initialization).
- **Testes Unitários**: Adicionar testes unitários para as classes `Layer`, `MLP` e os functors para garantir a correção do código.
- **Otimizadores**: Implementar otimizadores como Adam, RMSprop, etc., para melhorar o processo de treinamento.
- **Mini-batches**: Modificar o treinamento para usar mini-batches em vez de gradiente descendente estocástico puro.
- **Serialização**: Adicionar funcionalidade para salvar e carregar modelos treinados.
