# Especificação Experimental e Arquitetural: Impacto de Linguagens e Paradigmas de Programação em GPU no Treinamento e Inferência de Modelos de Linguagem (Transformers)

**Data:** 02 de Novembro de 2026  
**Autora (Persona Engenheira de ML Sênior):** Chuu (Kim Ji-woo) 🍓🐧🌍  
**Disciplina:** Metodologia Científica — Trabalho 2 (UnB)  
**Extensão Alvo do Artigo Final:** 4 a 7 páginas (formato acadêmico SBC/IEEE)

---

## 1. Questão de Pesquisa & Hipóteses Científicas

### 1.1 Questão de Pesquisa
> *"Em que medida a escolha da linguagem e do nível de abstração de programação em GPU (PyTorch Eager, Triton DSL e CUDA C++ nativo) afeta o tempo de execução (wall-clock time), a vazão (throughput em tokens/s) de treinamento e inferência e a complexidade de implementação (SLOC) de um modelo de linguagem baseado na arquitetura Transformer?"*

### 1.2 Variáveis do Estudo Experimental
- **Variável Independente (Fator Experimental):** A linguagem/paradigma de programação em GPU, avaliada nos 3 principais níveis de abstração da engenharia de Machine Learning moderna:
  1. **`01_pytorch` — PyTorch Eager Mode (Python/C++ Dispatcher + Autograd Dinâmico):** Alto nível imperativo/declarativo padrão da indústria (baseline inspirada no `ng-video-lecture` de Andrej Karpathy).
  2. **`02_triton` — PyTorch + Kernels Triton Customizados (`@triton.jit`):** Nível intermediário — DSL em Python baseada em programação em blocos (*tile/block-level programming* em SRAM) para fusão manual de Atenção Causal (FlashAttention), LayerNorm e Cross-Entropy (Forward + Backward).
  3. **`03_cuda_cpp` — C++ / CUDA Nativo (`nvcc` + `cuBLAS` + Kernels `.cu`):** Baixo nível nativo em C++/CUDA sem interpretador Python no loop de treinamento ou inferência (inspirada em `llm.c`).
- **Variáveis Dependentes (Métricas de Resposta):**
  1. **Wall-Clock Execution Time ($\text{s}$):** Tempo real decorrido para (a) 5000 iterações de treinamento, (b) inferência autoregressiva de 1000 tokens com `batch_size = 1` (latência), e (c) inferência autoregressiva de 1000 tokens com `batch_size = 64` (vazão em lote).
  2. **Throughput ($\text{tokens/s}$):** Taxa de processamento de tokens no treinamento ($5000 \times 64 \times 256 / T_{\text{train}}$) e na inferência ($B \times 1000 / T_{\text{infer}}$).
  3. **Source Lines of Code (`SLOC`):** Número de linhas físicas de código-fonte (excluindo linhas em branco, comentários e docstrings), subdividido em `sloc_core` (definição matemática do modelo, forward, backward, kernels e otimizador) e `sloc_harness` (carregamento de dados, cronometragem e exportação JSON).
  4. **Métricas Secundárias de Controle:** Consumo de memória de vídeo de pico (`peak_vram_mb`), tempo de compilação/aquecimento (`warmup_time_s`) e curva de convergência (`train_loss`, `val_loss`).
- **Variáveis de Controle (Invariantes):**
  - Mesmo hardware de execução (GPU NVIDIA L4 ou A100 no Google Colab Pro, fixada para todas as 3 execuções).
  - Mesma precisão numérica: `FP32` puro (IEEE 754 `float32`, com `allow_tf32 = False` para evitar divergência numérica entre kernels CUDA manuais e Tensor Cores implícitos).
  - Mesma arquitetura e hiperparâmetros do modelo (~10,79M parâmetros).
  - Mesma semente pseudoaleatória global: `SEED = 998244353`.
  - Mesmos pesos iniciais $\theta_0$ e mesma sequência de minibatches $(X_t, Y_t)$ para $t \in [0, 4999]$.

---

## 2. Arquitetura Invariante do Modelo (GPT Character-Level)

Baseada na implementação de referência de Andrej Karpathy (`ng-video-lecture/gpt.py`), padronizada para permitir paridade matemática exata entre todas as 3 implementações:

### 2.1 Dataset e Tokenização
- **Corpus:** `tinyshakespeare` (`data/input.txt`, $1.115.394$ caracteres).
- **Tokenizador:** Nível de caractere (*character-level*). Vocabulário ordenado `chars = sorted(list(set(text)))` com `vocab_size (V) = 65`.
- **Divisão:** Primeiros $90\%$ para treino ($1.003.854$ tokens) e últimos $10\%$ para validação ($111.540$ tokens).

### 2.2 Hiperparâmetros Estruturais e de Otimização
| Hiperparâmetro | Símbolo | Valor Fixo | Justificativa Científica |
| :--- | :---: | :---: | :--- |
| Tamanho do Lote (Treino) | $B$ | `64` | Saturação adequada dos Streaming Multiprocessors (SMs) da GPU |
| Tamanho do Contexto | $T$ | `256` | $16.384$ tokens processados por passo de treinamento |
| Dimensão de Embedding | $C$ (`n_embd`) | `384` | Múltiplo de 64 e 128 (alinhamento ótimo de memória e warps) |
| Número de Cabeças | $H$ (`n_head`) | `6` | Dimensão por cabeça $d_k = C / H = 64$ |
| Número de Camadas | $L$ (`n_layer`) | `6` | Total de $10.788.929$ parâmetros treináveis |
| Dropout | $p$ | `0.0` | Desativado para garantir determinismo entre Autograd e Backprop manual |
| Passos de Treinamento | `max_iters` | `5000` | Total de $81.920.000$ tokens treinados |
| Intervalo de Avaliação | `eval_interval` | `500` | `eval_iters = 50` lotes por split (cronometrado fora do timer de treino) |
| Otimizador | `AdamW` | $\text{lr}=3\times 10^{-4}$ | $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \lambda=0.01$ |
| Semente Global | `SEED` | `998244353` | Inicialização normal $\mathcal{N}(0, 0.02)$ e sorteio determinístico de lotes |

---

## 3. Detalhamento das 3 Implementações

### 3.1 `01_pytorch` — PyTorch Eager Mode (Baseline)
- **Arquivos:** `implementations/01_pytorch/model.py`, `implementations/01_pytorch/benchmark.py`.
- **Mecanismo:** Implementa `nn.Module` padrão do PyTorch em modo imperativo (`eager`), utilizando operações explícitas de atenção matricial (`q @ k.transpose(-2, -1)`, `masked_fill`, `F.softmax`) e `loss.backward()` via motor Autograd em C++.

### 3.2 `02_triton` — PyTorch + Kernels Triton Customizados (`@triton.jit`)
- **Arquivos:** `implementations/02_triton/kernels.py`, `implementations/02_triton/model.py`, `implementations/02_triton/benchmark.py`.
- **Mecanismo:** Substitui os gargalos intensivos em largura de banda de memória (HBM) por kernels `@triton.jit` integrados ao `torch.autograd.Function` (Forward + Backward):
  1. `triton_causal_attention_fwd` / `bwd` (FlashAttention causal em SRAM sem materializar $T \times T$ na HBM).
  2. `triton_layernorm_fwd` / `bwd` (redução fundida em bloco de média/variância e gradientes $dX, d\gamma, d\beta$).
  3. `triton_cross_entropy_fwd` / `bwd` (online log-sum-exp fundido com cálculo de gradiente de logits).

### 3.3 `03_cuda_cpp` — C++ / CUDA Nativo (`nvcc` + `cuBLAS`)
- **Arquivos:** `implementations/03_cuda_cpp/gpt_cuda.cu`, `implementations/03_cuda_cpp/Makefile`.
- **Mecanismo:** Executável binário autônomo compilado com `nvcc -O3 -lcublas`.
  - Projeções lineares e atenção batched executadas via `cublasSgemm` e `cublasSgemmStridedBatched`.
  - Kernels CUDA customizados (`__global__ void`) utilizando *warp-level primitives* (`__shfl_xor_sync`) e *shared memory* para LayerNorm (fwd/bwd), Causal Softmax (fwd/bwd), ReLU (fwd/bwd), Residual Add, Embedding Lookup/Scatter-Add, Cross-Entropy Loss e AdamW.
  - Zero dependência ou overhead do interpretador Python durante todo o loop de 5000 passos e geração autoregressiva.

---

## 4. Estrutura do Repositório GitHub e Fluxo de Execução

```text
trabalho_2/
├── AGENTS.md                                    # Persona & diretrizes da Chuu
├── README.md                                    # Documentação científica do repositório
├── data/
│   └── input.txt                                # Dataset TinyShakespeare
├── common/
│   ├── generate_golden.py                       # Gera pesos iniciais e lotes (seed=998244353)
│   ├── count_sloc.py                            # Contador determinístico e auditável de SLOC
│   └── aggregate_results.py                     # Consolida os 3 JSONs em tabelas LaTeX/MD e gráficos
├── implementations/
│   ├── 01_pytorch/
│   ├── 02_triton/
│   └── 03_cuda_cpp/
└── notebooks/
    ├── 01_benchmark_pytorch.ipynb
    ├── 02_benchmark_triton.ipynb
    └── 03_benchmark_cuda_cpp.ipynb
```
