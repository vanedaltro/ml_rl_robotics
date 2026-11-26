# 01 · Do MLP à CNN

Compare MLP e CNN no FashionMNIST. A base é baixada na primeira execução. O teste segue a divisão oficial.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/pytorch-na-pr%C3%A1tica-do-mlp-%C3%A0-cnn-rumo-vis%C3%A3o-para-santos-daltro-vojfe/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python treino.py --modelo ambos --epocas 4 --dispositivo cpu
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

Fonte dos dados: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist).
