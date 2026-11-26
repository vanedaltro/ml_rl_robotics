# 05 · Generalização e regularização

FashionMNIST com corrupção de rótulos de treino. As métricas registram as curvas originais. Para conferir o fluxo rapidamente, acrescente `--epochs 1 --seeds 7`; isso não reproduz o artigo.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/generaliza%C3%A7%C3%A3o-em-pytorch-quando-o-modelo-tira-10-e-na-santos-daltro-3iqne/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python treino.py --data ./dados --download --out ./nova_execucao
python figuras.py
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

`figuras.py` lê os CSVs versionados na pasta e requer Matplotlib e NumPy. O treino grava novos CSVs na pasta apontada por `--out`. Fonte dos dados: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist).
