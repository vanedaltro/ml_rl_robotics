# 04 · Caligrafia, LSTM, GRU e atenção

Laboratório de reconhecimento ASCII com LSTM, GRU e atenção. O teste usa escritores que ficaram fora do treino. O primeiro comando não baixa dados; o segundo baixa a base UJI se ela ainda não existir.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/da-caligrafia-ao-ascii-lstm-gru-e-aten%C3%A7%C3%A3o-em-pytorch-santos-daltro-pbmae/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 numpy pillow
python conceitos.py
python ocr.py --epocas 14 --treino 8000 --threads 3
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

Fonte dos dados e dos resultados derivados: F. Prat, M. Castro, D. Llorens, A. Marzal e J. Vilar, [UJI Pen Characters (Version 2)](https://doi.org/10.24432/C5FG8S), UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Adaptações: rasterização de traços, composição de linhas e inferências com mapas de atenção. Não inclui base, imagens ou checkpoints originais.
