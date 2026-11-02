# 🍓🐧 Estudo Comparativo de Paradigmas de Programação em GPU para Treinamento e Inferência de Transformers (GPT)

**Universidade de Brasília (UnB) — Departamento de Ciência da Computação (CIC)**  
**Disciplina:** Metodologia Científica — Trabalho 2  

---

## 🔬 1. Questão de Pesquisa (Research Question)

> **Qual é o impacto quantitativo de diferentes níveis de abstração de programação em GPU — (1) *PyTorch Eager* de alto nível, (2) *OpenAI Triton* baseado em blocos (*tile-based programming*), e (3) *CUDA C++ Nativo com cuBLAS* — sobre a vazão de processamento (*throughput* de treinamento e inferência), tempo de execução (*wall-clock time*), consumo de memória de vídeo (VRAM) e complexidade de desenvolvimento (medida em linhas físicas de código-fonte, SLOC), mantendo rigorosamente invariantes a arquitetura do modelo Transformer, os pesos iniciais, os mini-batches de dados e a precisão aritmética IEEE 754 `FP32`?**

Este repositório implementa um desenho experimental controlado com **paridade matemática verificada bit-a-bit** entre três paradigmas de programação em GPU aplicados ao mesmo modelo **Decoder-Only Transformer (estilo GPT)**.

---

## 🏛️ 2. Invariantes Metodológicos e Arquitetura do Modelo GPT

Para isolar estritamente a **variável independente** (o paradigma de implementação e execução em GPU), todas as três implementações compartilham os seguintes invariantes matemáticos e experimentais:

| Hiperparâmetro / Invariante | Valor Fixo | Descrição Metodológica |
| :--- | :---: | :--- |
| **Dataset** | *TinyShakespeare* | `data/input.txt` ($1.115.394$ caracteres; divisão $90\%$ treino / $10\%$ validação) |
| **Tamanho do Vocabulário (`vocab_size`)** | `65` | Tokenização caractere-a-caractere ordenada deterministicamente |
| **Semente Global (`SEED`)** | `998244353` | Gera pesos iniciais $\theta_0$ e mini-batches $X_t, Y_t$ idênticos |
| **Total de Parâmetros (`num_parameters`)** | `10,788,929` | Exatamente $10.788.929$ parâmetros treináveis em `FP32` |
| **Camadas Transformer (`n_layer`)** | `6` | Blocos Pre-LayerNorm com atenção causal e MLP |
| **Cabeças de Atenção (`n_head`)** | `6` | Dimensão por cabeça `head_size = 64` |
| **Dimensão de Embedding (`n_embd`)** | `384` | Projeções QKV sem bias (`c_attn`) e projeção de saída com bias (`c_proj`) |
| **Contexto Máximo (`block_size`, $T$)** | `256` | Comprimento da sequência por amostra |
| **Tamanho do Lote (`batch_size`, $B$)** | `64` | $64 \times 256 = 16.384$ tokens processados por passo de treino |
| **Dropout** | `0.0` | Desabilitado para garantir determinismo exato entre linguagens |
| **Ativação do MLP** | `ReLU` | `Linear(384 -> 1536, bias=True) -> ReLU -> Linear(1536 -> 384, bias=True)` |
| **Otimizador** | `AdamW` | $\text{lr}=3\times 10^{-4}, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \lambda=0.01$ (decoupled weight decay em todos os parâmetros) |
| **Precisão Numérica** | `FP32` (IEEE 754) | `tf32` explicitamente desabilitado (`allow_tf32 = False`) |
| **Portão de Sanidade (*Sanity Check*)** | $< 10^{-4}$ | Valida $|L_0 - L_0^{\text{golden}}| < 10^{-4}$ e $|L_1 - L_1^{\text{golden}}| < 10^{-4}$ antes do benchmark |

---

## ⚙️ 3. As 3 Implementações em GPU (`implementations/`)

### 3.1. `01_pytorch` — Baseline de Alta Abstração (PyTorch Eager)
- **Arquivos:** `implementations/01_pytorch/model.py`, `implementations/01_pytorch/benchmark.py`
- **Descrição:** Implementação idiomática em PyTorch utilizando operadores padrão (`nn.Embedding`, `nn.LayerNorm`, `nn.Linear`, `F.softmax`, `F.relu`, `F.cross_entropy`) e o motor de diferenciação automática `torch.autograd` com `torch.optim.AdamW`.

### 3.2. `02_triton` — Programação em Blocos com Kernels OpenAI Triton (`@triton.jit`)
- **Arquivos:** `implementations/02_triton/kernels.py`, `implementations/02_triton/model.py`, `implementations/02_triton/benchmark.py`
- **Descrição:** Mantém o grafo `torch.autograd.Function` para orquestração e GEMMs densos, mas substitui as operações críticas limitadas por largura de banda de memória por kernels customizados escritos em **OpenAI Triton (`@triton.jit`)** com passes de *forward* e *backward* explícitos:
  - **LayerNorm Forward & Backward** fundidos em blocos;
  - **Fused Causal FlashAttention Forward & Backward** com softmax online em SRAM (`input_precision="ieee"`);
  - **Fused Cross-Entropy Forward & Backward** em passagem única;
  - **ReLU Forward & Backward** vetorizados.

### 3.3. `03_cuda_cpp` — Implementação Nativa em C++/CUDA & cuBLAS (`gpt_cuda.cu`)
- **Arquivos:** `implementations/03_cuda_cpp/gpt_cuda.cu`, `implementations/03_cuda_cpp/benchmark.py`
- **Descrição:** Motor de treinamento e inferência autônomo escrito inteiramente em **C++/CUDA**, compilado com `nvcc -O3 -lcublas` sem dependência do interpretador Python durante o loop de execução:
  - Alocação estática contígua de VRAM (`cudaMalloc`) para pesos $\theta$, gradientes $\nabla\theta$, momentos $m, v$ do AdamW e tensores de ativação;
  - Multiplicações de matrizes via `cublasSgemm` (`CUBLAS_PEDANTIC_MATH`, IEEE `FP32`);
  - Kernels CUDA `__global__` customizados para Embeddings, LayerNorm (fwd/bwd), Causal Multi-Head Attention (fwd/bwd), ReLU (fwd/bwd), Cross-Entropy (fwd/bwd), acumulação de bias e passo completo do otimizador AdamW.

---

## 📏 4. Complexidade de Código (SLOC) Medida por `common/count_sloc.py`

A complexidade de engenharia é quantificada automaticamente pelo utilitário `common/count_sloc.py`, que contabiliza linhas físicas de código excluindo linhas em branco, comentários (`#`, `//`, `/* ... */`) e docstrings Python (`"""..."""`), separando o código de **Núcleo do Modelo/Kernels (`sloc_core`)** do código de **Harness de Benchmark (`sloc_harness`)**:

| Implementação | Paradigma | Core SLOC (`sloc_core`) | Harness SLOC (`sloc_harness`) | Total SLOC (`sloc_total`) | Detalhamento por Arquivo |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`01_pytorch`** | PyTorch Eager | **120** | **307** | **427** | `model.py`: 120 \| `benchmark.py`: 307 |
| **`02_triton`** | PyTorch + OpenAI Triton (`@triton.jit`) | **704** | **308** | **1012** | `model.py`: 124, `kernels.py`: 580 \| `benchmark.py`: 308 |
| **`03_cuda_cpp`** | C++/CUDA Nativo + cuBLAS | **1032** | **369** | **1401** | `gpt_cuda.cu (core)`: 1032 \| `gpt_cuda.cu (harness)`: 240, `benchmark.py`: 129 |

Para reproduzir a contagem de SLOC a qualquer momento:
```bash
python3 -m common.count_sloc
```

---

## ☁️ 5. Execução dos Experimentos no Google Colab Pro (`notebooks/`)

Para evitar contenção de recursos, *throttling* térmico ou interferência de memória na GPU, cada implementação é executada em um notebook dedicado no **Google Colab Pro** (recomendado: GPU **NVIDIA A100** ou **L4/T4** com o mesmo tipo de GPU selecionado nas 3 execuções):

1. [`notebooks/01_benchmark_pytorch.ipynb`](notebooks/01_benchmark_pytorch.ipynb) — Executa o benchmark completo para `01_pytorch` e baixa automaticamente `results_01_pytorch.json`.
2. [`notebooks/02_benchmark_triton.ipynb`](notebooks/02_benchmark_triton.ipynb) — Executa o benchmark completo para `02_triton` e baixa automaticamente `results_02_triton.json`.
3. [`notebooks/03_benchmark_cuda_cpp.ipynb`](notebooks/03_benchmark_cuda_cpp.ipynb) — Compila `gpt_cuda.cu` com `nvcc`, executa o benchmark completo para `03_cuda_cpp` e baixa automaticamente `results_03_cuda_cpp.json`.

### Passo a Passo no Google Colab Pro:
1. Abra cada notebook no Google Colab e selecione **Ambiente de execução $\rightarrow$ Alterar o tipo de ambiente de execução $\rightarrow$ GPU (A100 / L4 / T4)**.
2. Na **Célula 1**, configure `REPO_URL` com a URL do seu repositório GitHub (ou execute diretamente caso os arquivos já estejam no diretório de trabalho).
3. Na **Célula 2**, gere os artefatos dourados determinísticos (`SEED = 998244353`) em `/tmp/golden_gpt`.
4. Na **Célula 3**, execute o protocolo experimental completo (Sanity Check $\rightarrow$ 50 steps de Warm-up $\rightarrow$ 5000 steps de Treino Cronometrado com avaliação a cada 500 steps $\rightarrow$ Inferência Autoregressiva de 1000 tokens com $B=1$ e $B=64$).
5. Na **Célula 4**, inspecione o resumo das métricas e receba o arquivo `results_0X_*.json` via download automático do navegador.

---

## 📊 6. Agregação dos Resultados, Tabelas Científicas e Gráficos

Após baixar os três arquivos JSON gerados nos notebooks (`results_01_pytorch.json`, `results_02_triton.json`, `results_03_cuda_cpp.json`), coloque-os no diretório `results/` da sua máquina local e execute o agregador:

```bash
python3 -m common.aggregate_results --results-dir results --output-dir results/summary
```

O script `common/aggregate_results.py` cruza automaticamente os resultados experimentais com as métricas de SLOC e produz em `results/summary/`:
- `summary_table.md`: Tabela comparativa formatada em Markdown;
- `summary_table.tex`: Tabela científica em LaTeX pronta para inclusão no artigo da disciplina;
- `throughput_comparison.png`: Gráfico de barras comparando a vazão (tokens/s) de Treinamento, Inferência $B=1$ e Inferência $B=64$;
- `pareto_sloc_vs_throughput.png`: Fronteira de Pareto relacionando Complexidade de Implementação (`sloc_core`) vs. Vazão de Treinamento;
- `loss_convergence.png`: Curvas de convergência da *Cross-Entropy Loss* de validação comprovando a equivalência matemática ao longo dos 5000 passos;
- `aggregated_summary.json`: Arquivo consolidado com todas as métricas e metadados.

---

## 🧪 7. Suite de Testes Automatizados

Para verificar localmente a paridade numérica, a geração dos artefatos dourados, os kernels Triton, o binário CUDA C++, os notebooks e a contagem de SLOC:

```bash
python3 -m pytest -v
```
