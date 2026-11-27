# 06 · Monte Carlo e diferenças temporais

Os dois agentes estimam o valor de **uma política fixa** a partir das mesmas trajetórias. A diferença é o alvo: Monte Carlo usa o retorno observado de cada visita após o episódio terminar; TD(0) usa `r + γ V(s')` a cada passo, sem bootstrap no terminal. Nenhum deles recebe a tabela de transições.

```bash
python3 -m unittest discover -v
python3 previsao.py
```

Num corredor de três casas com recompensa 1 na saída, V inicial zero, α = 1 e γ = 0,9, após um episódio Monte Carlo dá **[0,81; 0,9; 1]**, e TD(0) dá **[0; 0; 1]**. TD só propagará a informação às primeiras casas em novas passagens.

Na caminhada que sorteia esquerda ou direita a cada passo, só chegar ao terminal da direita paga 1. Com γ = 1, o valor exato da casa s é a probabilidade de sair à direita, `s / 6` (cinco casas, terminais 0 e 6). Essa referência é usada apenas para medir erro.

| Após quantos episódios | RMSE Monte Carlo | RMSE TD(0) |
| --- | ---: | ---: |
| 0 | 0,2357 | 0,2357 |
| 10 | 0,1866 | 0,1282 |
| 50 | 0,0975 | 0,0491 |
| 100 | 0,0906 | 0,0540 |

Resultado local, média de 100 sementes, α fixo de 0,03 (MC) e 0,10 (TD). Esses passos são escolhas didáticas, não uma garantia de que um método sempre supera o outro. Com α fixo, o erro também pode subir após novas amostras. Tente mudar os dois passos, comparando trajetórias iguais.
