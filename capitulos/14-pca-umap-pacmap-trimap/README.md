# 14. De PCA a PaCMAP: reduções de dimensionalidade na prática

PCA exato com SVD em PyTorch e uma comparação reprodutível entre PCA, UMAP, PaCMAP e TriMap. Os quatro métodos recebem os mesmos 900 pontos sintéticos em 48 dimensões, com saída em 2D. A posição latente é usada só para avaliar e colorir a figura.

## Execute em CPU

Foi executado em Python 3.12 no Linux. Em um ambiente virtual:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python pca_torch.py
python comparacao.py --out nova_execucao
python pacmap_angular.py
```

No macOS e Windows, consulte o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) para instalar a versão apropriada. O pacote Annoy, dependência de TriMap, pode exigir um compilador C++. A primeira compilação do Numba pode demorar. Após instalar os pacotes, o laboratório não baixa datasets nem pesos.

- `pca_torch.py`: ajuste e projeção com SVD, variância explicada e verificação de reconstrução. Inclui um exemplo de inversão do ranking por cosseno após centralizar.
- `comparacao.py`: compara vizinhanças e correlações de Spearman para 20 mil pares fixos, contra distâncias na entrada e na faixa latente. Gera CSV, metadados e gráfico 2×2. `--seeds 7` repete só a semente mostrada no gráfico.
- `pacmap_angular.py`: checagem separada da vizinhança por cosseno antes e depois do PCA, e teste do PaCMAP com `distance="angular"`.
- `diagramas_pca.py`: gera os três diagramas didáticos com Matplotlib.
- `results.csv`, `metadados-benchmark.json` e `resultados-*.txt`: resultados e versões da execução de referência.

As três sementes variam o ajuste, não o conjunto sintético. As métricas são calculadas nos próprios pontos projetados, sem avaliação fora da amostra. Resultados de uma faixa enrolada não estabelecem um ranking universal de métodos.

[Artigo no LinkedIn](https://www.linkedin.com/pulse/de-pca-pacmap-como-comparar-quatro-mapas-dos-mesmos-santos-daltro-rmzqe/) · [Post ilustrado](https://www.linkedin.com/feed/update/urn:li:activity:7531711254088826881/)
