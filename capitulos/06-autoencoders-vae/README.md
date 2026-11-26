# 06 · Autoencoders e VAEs

Autoencoder e VAE com latente em duas dimensões. O treino baixa o FashionMNIST e gera checkpoints e dados intermediários. `exportar.py` cria o arquivo necessário para regenerar figuras. Para teste rápido, acrescente `--epochs 1 --n 1000` ao treino.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/autoencoders-e-vaes-em-pytorch-da-reconstru%C3%A7%C3%A3o-%C3%A0-de-santos-daltro-n7vue/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python treino.py --data ./dados --download
python exportar.py
python figuras.py
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

`exportar.py` usa NumPy; `figuras.py` usa NumPy e Matplotlib. Fonte dos dados: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), publicado sob MIT pela Zalando Research; veja [a licença da base](LICENSE-FashionMNIST.txt).
