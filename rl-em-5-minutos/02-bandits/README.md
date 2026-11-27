# 02 · Explorar ou aproveitar? Bandits

Três alavancas com médias reais 4, 6 e 3. Compare guloso, ε-greedy e UCB em 500 repetições, com um oráculo como referência.

```bash
python3 -m unittest -v
python3 bandits.py --rodadas 1000 --repeticoes 500 --seed 42
```

O simulador só depende da biblioteca padrão. O CSV mede recompensa média acumulada por rodada, pseudo-regret acumulado e fração de escolhas da alavanca B. Para regenerar as figuras, instale Matplotlib em seu ambiente e rode `python3 figuras.py`.

- [Artigo e explicação das fórmulas](artigo.md)
- [Diagrama das três alavancas](01-alavancas.png)
- [Gráficos do experimento](02-resultados.png)

O vídeo acompanha a edição no LinkedIn. Não incluímos o MP4 no histórico do Git.
