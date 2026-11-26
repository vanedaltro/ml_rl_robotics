# 13. Antes do UMAP: embeddings, variedades e distâncias

Um laboratório sobre como a escolha da métrica muda a comparação de vetores, caminhos e distribuições. O exemplo não executa UMAP. Ele prepara a intuição para escolher uma métrica antes de projetar os dados.

## Execute em CPU

Em um ambiente virtual com Python 3.11 ou 3.12:

```bash
python -m pip install torch==2.9.1 scipy==1.15.3 matplotlib==3.10.0
python geometria.py
python figuras.py
```

- `geometria.py`: compara cosseno e distância euclidiana, corda e comprimento de arco em uma espiral, KL nas duas direções, Jensen–Shannon, variação total, Hellinger, Wasserstein 1D, distância de energia e MMD com núcleo RBF.
- `figuras.py`: gera quatro PNGs na pasta do capítulo. Execute após instalar Matplotlib e suas dependências.
- `resultados.txt`: saída da execução de referência em PyTorch 2.9.1, CPU e precisão dupla.

Os dados são sintéticos, sem download de bases ou pesos. No exemplo da espiral, a ordem dos pontos é conhecida pelo simulador; o arco aproximado não é uma distância geodésica estimada em dados desconhecidos. A MMD ao quadrado usa a estatística V com largura 1.

[Artigo no LinkedIn](https://www.linkedin.com/pulse/antes-do-umap-embeddings-manifolds-e-geometria-dos-santos-daltro-pieve/) · [Post ilustrado](https://www.linkedin.com/feed/update/urn:li:activity:7531705196167467008/)
