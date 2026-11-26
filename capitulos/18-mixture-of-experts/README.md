# Capítulo 18: Mixture of Experts (MoE) em PyTorch

Aqui, a atenção causal é compartilhada. Só trocamos a FFN do bloco: uma
FFN no modelo denso, ou um roteador e quatro FFNs independentes no modelo MoE.
Para cada token no MoE, apenas as duas FFNs escolhidas recebem a representação.

## Rode o laboratório

Ambiente medido: Python 3.12, PyTorch 2.15.0+cpu, Linux x86_64.
Use um pacote PyTorch para CPU compatível com seu ambiente (veja o
[instalador do PyTorch](https://pytorch.org/get-started/locally/)):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v
python lab_moe.py
python lab_moe.py --seeds 7 --out resultado-seed7.json
python teste_combinacoes_novas.py
```

Não há downloads de dados. `resultados.json` inclui metadados e métricas por
semente. A execução também imprime métricas da avaliação.

## Do roteador ao token

1. Achate `[B,S,D]` para `[T,D]` (`T=B*S`), calcule `[T,E]` logits e
   selecione `[T,k]` índices com `topk`. A softmax separada sobre apenas
   os k logits selecionados produz pesos que somam 1 por token.
2. Para cada especialista, localize as ocorrências com `torch.where`.
   Use `index_select` para reunir só os tokens atribuídos a ele. Pule
   especialistas sem atribuições.
3. Multiplique pelo peso escolhido e devolva com `index_add`. Isso soma as
   contribuições de diferentes especialistas para o mesmo token, preservando
   a ordem do lote. A referência densa nos testes avalia todos os experts e
   mascara os não escolhidos; é só um oráculo, não o caminho executado.
4. A perda auxiliar de balanceamento é `E * sum(f_i * P_i)`: `f_i` é a
   contagem de atribuições a i dividida por `T*k`, enquanto `P_i` é a média
   das probabilidades da softmax **sobre todos E** antes do corte. Os
   índices não têm gradiente; `P_i` é diferenciável. O treino minimiza
   `cross_entropy + lambda * balance`.

Não há limite de capacidade, descarte de tokens, paralelismo por especialista
nem comunicação em rede neste exemplo. `topk` com `k=1` e softmax sobre o
único selecionado produz peso constante 1 e não manda gradiente da tarefa ao
roteador por esse peso. Não copie esse detalhe para o Switch top-1: nele o
peso do selecionado é obtido da distribuição completa.

## Experimento

Um bloco de Transformer causal, dimensão 32, quatro cabeças, FFNs sem viés
de largura 64; MoE com E=4 e k=2. A entrada é `[BOS, regra, a, b]` para
`a,b∈{0,…,7}`. Na última posição, o modelo prevê um entre oito tokens com
o resultado módulo 8 da regra escolhida: `a+b`, `a−b`, `a XOR b` ou
`2a+b`. As outras posições não entram na perda de resposta.

Foram sorteados 2.048 exemplos de treino (seed 2026) e 512 de teste
(seed 2027) da mesma gramática, com repetição. As triplas regra/a/b podem
aparecer nos dois conjuntos: 511 das 512 entradas de teste já aparecem no
treino. Portanto a acurácia mede principalmente ajuste às regras e exemplos
conhecidos, não generalização para novas combinações. Três sementes pareadas
(7, 23, 42); 12 épocas, batch 128, AdamW 0,004. Constrói-se a parte
compartilhada do bloco antes da FFN para parear a inicialização dos pesos
compartilhados. As duas variantes MoE começam com a mesma inicialização,
incluindo um viés deliberado no roteador `[2, 2, 0, 0]` para induzir mau
balanceamento; o teste observou 50/50/0/0% dos slots antes do treino.

| Modelo | Acurácia média ± DP amostral | Cross-entropy média | Fatia dos slots por especialista |
| --- | ---: | ---: | --- |
| Denso | 82,55 ± 3,44% | 0,489 | não se aplica |
| MoE λ=0 | 86,78 ± 2,16% | 0,384 | 31,24% / 44,25% / 17,20% / 7,31% |
| MoE λ=0,2 | 92,77 ± 1,74% | 0,242 | 21,57% / 24,58% / 26,38% / 27,46% |

No teste de cada semente, há 512 × 4 = 2.048 tokens e 4.096 atribuições.
Cada token conta duas vezes (uma vez por expert escolhido), de modo que no
caso 4/2 um expert pode receber no máximo 50% de todos os slots. A referência
uniforme é 25% dos slots cada. Utilização mais uniforme não comprova
especialização sem uma análise separada.

| Configuração | Parâmetros do modelo | Parâmetros totais da FFN (ou de todos os experts) | Parâmetros de FFNs selecionadas por token |
| --- | ---: | ---: | ---: |
| Denso | 9.440 | 4.096 | 4.096 |
| MoE | 21.860 | 16.384 | 8.192 |

As colunas de FFN excluem o roteador (132 parâmetros). O MoE usa metade
dos pesos de expert disponíveis **por token**, não metade dos pesos do
modelo nem menos FFN computação do que o baseline denso deste laboratório.
Comparar as acurácias contra a referência densa não é comparar com parâmetros
ou computação pareados. Nenhuma velocidade foi medida; o laço em Python e
cópias podem tornar este código mais lento do que uma FFN densa. Em produção,
medir tempo, memória, sobrecarga de despacho, desequilíbrio e comunicação.

## Suplemento: combinações totalmente novas

`teste_combinacoes_novas.py` reserva 16 dos 64 pares de operandos de cada
regra para teste (64 prompts distintos) e usa os outros 48 por regra para
sortear os 2.048 exemplos de treino. O script verifica que nenhum prompt
completo de treino reaparece no teste; orçamento e sementes de modelo
iguais aos do primeiro experimento. Com esse teste mais exigente, a
acurácia é 28,65 ± 3,25% (denso), 26,56 ± 5,63% (MoE sem auxiliar)
e 29,69 ± 3,13% (MoE com auxiliar). A auxiliar ainda distribui os slots
quase igualmente (24,67/25,33/25,52/24,48%), mas isso não garante
generalização a operandos novos em combinação. O teste não mede
generalização para regras novas.

Referências primárias: [Sparse MoE, Shazeer et al.](https://arxiv.org/abs/1701.06538),
[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html),
[Mixtral of Experts](https://arxiv.org/abs/2401.04088),
[Megatron-Core, perda auxiliar top-k](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.moe_utils.html).

## Figuras e exemplo curto

Execute também o exemplo independente com:

~~~bash
python exemplo_minimo.py
~~~

As figuras estão em figuras/. Para regenerá-las no Linux, instale Pillow e
as fontes DejaVu Sans e execute gerar_diagramas.py.
