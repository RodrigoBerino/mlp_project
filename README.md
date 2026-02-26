# MLP Project + Interface Desktop (React + TypeScript + C++)

Este repositório mantém a implementação existente da MLP em C++ com functors de ativação e adiciona uma interface desktop em **React + TypeScript** (via Electron) para carregar CSV e executar inferência sem alterar a lógica central da rede.

## Estrutura de diretórios

```text
mlp_project/
├── activation_functions.h      # Functors (ReLU, Sigmoid, Tanh)
├── layer.h                     # Camada da rede
├── loss_functions.h            # Função de custo (MSE)
├── mlp.h                       # Implementação da MLP (inalterada)
├── CMakeLists.txt              # Build do backend C++
├── cpp/
│   ├── main.cpp                # Entrada do backend (CLI JSON)
│   ├── MainWindow.h            # Orquestração da execução
│   ├── MainWindow.cpp          # Integra CSV + MLP
│   ├── CsvReader.h             # Leitura/validação de CSV
│   └── CsvReader.cpp           # Implementação da leitura CSV
├── electron/
│   ├── main.ts                 # Processo principal Electron + integração C++
│   └── preload.ts              # Bridge segura para o renderer
├── frontend/
│   ├── index.html
│   └── src/
│       ├── App.tsx             # UI: drag-and-drop, ComboBox, resultados
│       ├── main.tsx
│       ├── styles.css
│       └── vite-env.d.ts
├── package.json
├── tsconfig.json
├── tsconfig.electron.json
└── vite.config.ts
```

## Funcionalidades entregues

- Drag-and-drop de arquivo CSV.
- ComboBox para função de ativação (`ReLU`, `Sigmoid`, `Tanh`).
- Botão **Executar MLP**.
- Exibição de:
  - dimensão dos dados carregados;
  - resultado da inferência (saída por linha).
- Validação de CSV com erros amigáveis.
- Integração com a classe `MLP` existente, sem modificá-la.

## Como compilar o backend C++ (CMake)

Pré-requisito: compilador com suporte a **C++17** e CMake >= 3.16.

```bash
cmake -S . -B build
cmake --build build
```

Isso gera o executável `build/mlp_inference`.

## Como executar interface desktop

Pré-requisito: Node.js 18+.

1. Instale dependências:

```bash
npm install
```

2. Inicie interface (Vite + Electron):

```bash
npm run dev
```

3. Fluxo de uso:
   - Arraste um arquivo `.csv` para a área de drop;
   - Selecione a ativação;
   - Clique em **Executar MLP**.

## Uso direto do backend C++ (opcional)

```bash
./build/mlp_inference --csv caminho/arquivo.csv --activation relu
```

Retorno em JSON:

```json
{ "rows": 4, "columns": 2, "outputs": [0.123456, 0.654321, 0.222222, 0.888888] }
```

## Observações de integração

- A MLP existente foi reutilizada diretamente (`mlp.h`, `layer.h`, `activation_functions.h`, `loss_functions.h`).
- A interface desktop chama o executável C++ e mostra o retorno JSON no frontend.
- Erros de CSV e argumentos inválidos retornam mensagens amigáveis para a interface.

## CI/CD (GitHub Actions)

Foi adicionada a pipeline em `.github/workflows/ci.yml` com os seguintes pontos:

- **Gatilhos**: `push` na `main` e `pull_request` para `main`.
- **C++ CI em matriz**: `ubuntu-latest` e `windows-latest`.
- **Etapas C++**:
  - checkout do código;
  - cache da pasta de build CMake;
  - configuração com CMake em `Release`;
  - build com CMake em `Release`;
  - execução de testes via `ctest` (se existirem);
  - upload de artefatos (binário compilado + logs de configuração/build/testes).
- **Frontend CI (Ubuntu)**:
  - setup de Node com cache npm;
  - instalação de dependências (`npm ci`);
  - build da interface (`npm run build`);
  - upload dos artefatos de frontend (`dist` e `dist-electron`).

### Como ativar no repositório

1. Faça commit do arquivo `.github/workflows/ci.yml` na branch principal (ou abra PR para `main`).
2. Garanta que **Actions** esteja habilitado no GitHub:
   - `Repository Settings` → `Actions` → permitir execução de workflows.
3. Após merge/push em `main`, a pipeline será disparada automaticamente.
4. Para PRs, a pipeline roda automaticamente e bloqueia merge caso haja falhas (se branch protection exigir checks obrigatórios).

## Diagrama de classes (Mermaid)

- Arquivo gerado: `docs/class_diagram.md`.
- Você pode copiar o bloco Mermaid para visualizar em editores compatíveis (GitHub, Mermaid Live Editor, etc.).

## CSVs de teste para drag-and-drop

Foram adicionados 5 arquivos em `data/` para facilitar testes da interface:

- `data/teste_01_xor.csv`
- `data/teste_02_normalizado.csv`
- `data/teste_03_negativos.csv`
- `data/teste_04_5colunas.csv`
- `data/teste_05_uma_coluna.csv`
