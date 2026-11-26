# Capítulo 17: pesos estritos em −1, 0 e +1

Laboratório reproduzível em PyTorch. Compara a precisão de FP32, ternário estrito
e ternário com uma escala por neurônio/filtro. A classificação de imagens usa
dados geométricos sintéticos próprios, sem download. O exemplo de regressão
enumera todas as nove escolhas para uma camada linear de dois pesos.

[Artigo no LinkedIn](https://www.linkedin.com/pulse/e-se-cada-peso-s%25C3%25B3-pudesse-ser-1-0-ou-um-laborat%25C3%25B3rio-em-santos-daltro-btsje/) · [Post de divulgação](https://www.linkedin.com/feed/update/urn:li:activity:7531740362097373184/)

## Executar

PyTorch 2.15.0+cpu, TorchAO 0.18.0+cpu, Python 3.12 e Linux x86_64 foram usados
nesta medição. Use a combinação suportada para sua plataforma, indicada em
[PyTorch](https://pytorch.org/get-started/locally/) e
[TorchAO](https://docs.pytorch.org/ao/). No Linux x86_64, para a combinação testada:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchao==0.18.0
python -m unittest -v
python regressao_exata.py
python lab_ternario.py
python torchao_ptq.py
```

Para regenerar as figuras, instale Pillow e certifique-se de ter a família
DejaVu Sans disponível; depois rode `python gerar_figuras.py`.

## O que foi medido

Versões sem viés nas projeções. Todos os modelos veem as mesmas 1.536 imagens
de treino (seed 2026) e 512 imagens de teste (seed 2027). Dentro de cada
arquitetura e semente (7, 23, 42), os pesos mestres e a ordem dos minibatches
são idênticos na largada. Cada execução tem 8 épocas, AdamW, taxa 0,003,
minibatches de 128 e 1 thread no CPU. É uma ablação de escala, não uma busca de
hiperparâmetros nem um benchmark de desempenho em hardware.

| Modelo | FP32 | Ternário estrito | Ternário com escala |
| --- | ---: | ---: | ---: |
| MLP | 86,78 ± 0,79% | 44,86 ± 17,07% | 84,18 ± 1,03% |
| CNN | 95,25 ± 1,11% | 25,52 ± 1,63% | 90,04 ± 3,73% |
| Transformer | 67,58 ± 1,35% | 29,17 ± 1,57% | 56,32 ± 2,59% |

Acurácia média ± 1 desvio padrão amostral entre três sementes. Em
`resultados.json`: métricas por execução, média, desvio, norma RMS dos
pesos usados, magnitude inicial dos logits e frequência de zeros.

O ternário estrito usa exatamente `code` em cada projeção; o escalado usa
`scale * code`, com escala por linha de uma matriz ou filtro convolucional.
Os dois têm pesos mestres FP32 e usam STE. As ativações continuam em ponto
flutuante. No Transformer, LayerNorm e posições também ficam em FP32. Nenhum
dos dois empacota códigos ou chama um kernel dedicado: o dtype computacional
continua em float. Ganhos, normalização, arquitetura e ajuste do treino podem
mudar esses resultados. Note que a montagem geométrica deste capítulo não é
o conjunto de dados do capítulo 16.

### TorchAO real, em uma etapa separada

`torchao_ptq.py` copia o MLP FP32 (seed 7) para módulos `nn.Linear` padrão,
faz a conversão com `quantize_(copy, Int8WeightOnlyConfig(set_inductor_config=False))`
e executa inferência. O tipo do peso de cada uma das três camadas é quantizado
e o armazenamento interno é `torch.int8`; as ativações não foram quantizadas.
O tamanho serializado é o `state_dict` via `torch.save` em memória, não
memória de pico: 76.385 → 22.985 bytes. A acurácia foi 87,50% antes e depois;
a diferença absoluta máxima dos logits em 512 exemplos foi 0,04966.
Nenhuma latência foi medida. Esse fluxo não é o treino ternário.

Referências: [TorchAO INT8 weight-only](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html),
[TorchAO QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html) e
[BitNet b1.58](https://arxiv.org/abs/2402.17764).
