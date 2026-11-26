# Capítulo 19: Avaliação sem vazamento

Laboratório PyTorch com dados sintéticos, 24 máquinas, 192 episódios e 2.304 janelas de sensores. O mesmo MLP é avaliado por janelas, episódios ou máquinas, com normalização ajustada só no treino e baseline da classe majoritária.

```bash
python -m pip install torch
python -m unittest -v
python lab_avaliacao.py
python lab_avaliacao.py --seeds 7 --out resultado-seed7.json
```

Com PyTorch 2.9.1 e uma thread em CPU Apple Silicon, três sementes controlam a divisão e o treino. Cada divisão reserva 1.728 janelas no treino e 576 no teste. São janelas não independentes: 24 amostras de largura, avanço 2. As médias abaixo incluem o desvio padrão amostral em pontos percentuais.

| Divisão | MLP | Maioria | Janelas com amostras sobrepostas |
| --- | ---: | ---: | ---: |
| Por linha | 100,00 ± 0,00% | 50,17 ± 1,14% | 100% |
| Por episódio | 68,00 ± 2,36% | 54,86 ± 4,34% | 0% |
| Por máquina | 58,10 ± 6,50% | 47,92 ± 10,83% | 0% |

O JSON guarda acurácia balanceada, matriz de erros, proporção de janelas com grupos vistos, volume testado e resultados por máquina. Métricas por janela não são métricas por falha independente. Acurácia balanceada de um grupo com só uma classe é registrada como `null`. O desenho simula diferentes previsões em produção, não serve de benchmark para manutenção real nem mede desempenho futuro (não há corte temporal no experimento).

O script de figuras necessita Pillow e fonte Arial ou DejaVu Sans:
```bash
python -m pip install Pillow
python gerar_figuras.py
```
