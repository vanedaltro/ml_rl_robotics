# Machine Learning, PyTorch e robótica

Código que acompanha a série de artigos de Vânia Daltro. Os capítulos avançam de MLPs e CNNs à atenção, Transformers e Vision Transformers, com laboratórios de visão e exemplos de simulação quando indicado.

## Comece aqui

Use um ambiente virtual com Python 3.11 ou 3.12. Os exemplos curtos dos capítulos 7 a 12 usam apenas PyTorch; os laboratórios anteriores podem precisar de TorchVision, dados externos ou MuJoCo. Leia as instruções de cada capítulo antes do treino.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1
python capitulos/09-autoatencao/demo.py
python capitulos/12-vision-transformers/vit.py
```

No PowerShell, ative o ambiente com `.venv\Scripts\Activate.ps1`. Os exemplos curtos rodam em CPU sem baixar datasets. A primeira instalação das bibliotecas precisa de rede.

## Capítulos

| Nº | Código e instruções | Artigo |
| --- | --- | --- |
| 01 | [Do MLP à CNN](capitulos/01-mlp-cnn/) | [LinkedIn](https://www.linkedin.com/pulse/pytorch-na-pr%C3%A1tica-do-mlp-%C3%A0-cnn-rumo-vis%C3%A3o-para-santos-daltro-vojfe/) |
| 02 | [BatchNorm, LayerNorm, pooling, losses e otimizadores](capitulos/02-normalizacao-pooling/) | [LinkedIn](https://www.linkedin.com/pulse/pytorch-normaliza%C3%A7%C3%A3o-pooling-otimizadores-e-sa%C3%ADdas-em-santos-daltro-zar3e/) |
| 03 | [ResNet e gradientes](capitulos/03-resnet/) | [LinkedIn](https://www.linkedin.com/pulse/resnet-na-pr%C3%A1tica-por-que-o-gradiente-some-e-um-muda-santos-daltro-rqtse/) |
| 04 | [Caligrafia, LSTM, GRU e atenção](capitulos/04-caligrafia/) | [LinkedIn](https://www.linkedin.com/pulse/da-caligrafia-ao-ascii-lstm-gru-e-aten%C3%A7%C3%A3o-em-pytorch-santos-daltro-pbmae/) |
| 05 | [Generalização e regularização](capitulos/05-generalizacao/) | [LinkedIn](https://www.linkedin.com/pulse/generaliza%C3%A7%C3%A3o-em-pytorch-quando-o-modelo-tira-10-e-na-santos-daltro-3iqne/) |
| 06 | [Autoencoders e VAEs](capitulos/06-autoencoders-vae/) | [LinkedIn](https://www.linkedin.com/pulse/autoencoders-e-vaes-em-pytorch-da-reconstru%C3%A7%C3%A3o-%C3%A0-de-santos-daltro-n7vue/) |
| 07 | [YOLO, U-Net e losses de segmentação](capitulos/07-deteccao-segmentacao/) | [LinkedIn](https://www.linkedin.com/pulse/da-imagem-%C3%A0-decis%C3%A3o-yolo-u-net-e-losses-vis%C3%A3o-santos-daltro-kb9ce/) |
| 08 | [Convolução, stride, padding e MaxPool](capitulos/08-cnns/) | [LinkedIn](https://www.linkedin.com/pulse/cnns-por-dentro-como-calcular-convolu%C3%A7%C3%A3o-stride-e-santos-daltro-v4xne/) |
| 09 | [Q, K e V à mão e em PyTorch](capitulos/09-autoatencao/) | [LinkedIn](https://www.linkedin.com/pulse/da-caligrafia-%C3%A0-autoaten%C3%A7%C3%A3o-q-k-e-v-calculados-m%C3%A3o-santos-daltro-tco1e/) |
| 10 | [Tokens, embeddings e posições](capitulos/10-tokens-embeddings/) | [LinkedIn](https://www.linkedin.com/pulse/antes-do-transformer-como-texto-e-pixels-viram-tokens-santos-daltro-jxzne/) |
| 11 | [Transformer original e as três famílias](capitulos/11-transformers/) | [LinkedIn](https://www.linkedin.com/pulse/transformers-por-dentro-do-paper-ao-primeiro-modelo-em-santos-daltro-enfse/) |
| 12 | [Vision Transformer em CPU](capitulos/12-vision-transformers/) | Em preparação |

Resultados de laboratórios sintéticos e tempos de execução dependem da configuração e não são métricas de produção. O repositório não inclui bases externas nem pesos. Veja a [licença MIT](LICENSE) para o código; datasets de terceiros têm condições próprias.
