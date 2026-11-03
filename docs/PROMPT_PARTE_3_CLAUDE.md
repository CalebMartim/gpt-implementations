Ative e siga rigorosamente a skill `research-paper-writing` (instalada em `.agents/skills/research-paper-writing/SKILL.md` e suas referências em `references/abstract.md`, `references/introduction.md`, `references/method.md`, `references/experiments.md`, `references/conclusion.md` e `references/does-my-writing-flow-source.md`).

Sua tarefa é executar a Parte 3 do nosso projeto de Metodologia Científica (UnB): escrever o artigo científico completo (entre 4 e 7 páginas compiladas) em `paper/main.tex` usando a classe `IEEEtran.cls` a partir dos resultados empíricos já consolidados na GPU NVIDIA A100-SXM4-40GB.

### Passo 1: Leitura dos Dados Experimentais e Especificação
Leia os artefatos já gerados no repositório:
1. `results/summary/summary_metrics.json` e `results/summary/summary_table.tex`
2. `results/results_01_pytorch.json`, `results/results_02_triton.json` e `results/results_03_cuda_cpp.json`
3. `docs/superpowers/specs/2026-11-02-transformer-gpu-benchmark-design.md` e `AGENTS.md`

### Passo 2: Estrutura Exata do Artigo (`paper/main.tex`)
Mantenha exatamente a estrutura de seções já definida em `paper/main.tex` (sem seção separada de Discussão — integre a interpretação arquitetural diretamente na seção de Resultados):
1. **Resumo / Abstract**
2. **I. Introdução**
3. **II. Fundamentação Teórica** (Arquitetura Decoder-Only Transformer Causal, Hierarquia de Memória de GPUs HBM vs. SRAM e Paradigmas PyTorch Eager, Triton DSL e CUDA C++)
4. **III. Metodologia** (Desenho experimental, variáveis independentes/dependentes, invariantes de controle com semente `998244353` em FP32 estrito `allow_tf32=False` / `CUBLAS_PEDANTIC_MATH`, arquitetura de 10.788.929 parâmetros e detalhamento técnico dos kernels de `01_pytorch`, `02_triton` e `03_cuda_cpp`)
5. **IV. Resultados** (Análise quantitativa e arquitetural completa com tabelas `booktabs` e inclusão das 3 figuras já geradas em `../results/summary/`: `loss_convergence.png`, `throughput_comparison.png` e `pareto_sloc_vs_throughput.png`, cobrindo paridade numérica, treinamento, inferência B=1 vs. B=64, consumo de VRAM e fronteira de Pareto SLOC vs. Desempenho: 120 vs. 704 vs. 1.032 linhas no núcleo)
6. **V. Conclusão**

### Passo 3: Diretrizes Estritas de Estilo (Prosa Humana e Natural — Zero "Voz de IA")
- Siga o teste de fluxo de parágrafos de `references/does-my-writing-flow-source.md` (1 mensagem clara por parágrafo, frase-tópico direta na primeira sentença, conexão lógica de causa/contraste/consequência entre sentenças).
- Escreva em Português Acadêmico Brasileiro sóbrio, direto, técnico e natural, como um pesquisador sênior de Sistemas Computacionais / HPC escrevendo para um congresso da SBC/IEEE.
- **PROIBIDO usar clichês típicos de LLM ("voz de IA"):** Nunca use frases como *"No cenário em constante evolução"*, *"Vale ressaltar que"*, *"É crucial destacar"*, *"Nesse sentido, emerge como"*, *"Uma miríade/tapeçaria de"*, *"Não apenas X, mas também Y"* repetidamente, adjetivos hiperbólicos (*"revolucionário"*, *"impressionante"*, *"notável"*) ou excesso de travessões (`—`).
- Prefira afirmações ancoradas diretamente em mecanismos de hardware e nos dados medidos.

### Passo 4: Compilação e Verificação do PDF
- Compile `paper/main.tex` usando `pdflatex -interaction=nonstopmode main.tex` (duas vezes para resolver referências cruzadas) dentro da pasta `paper/`.
- Verifique que `paper/main.pdf` foi gerado sem erros de LaTeX, que todas as 3 figuras e as tabelas aparecem corretamente e que o artigo final possui entre **4 e 7 páginas**.
