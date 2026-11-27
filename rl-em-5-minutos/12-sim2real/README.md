# 12 · Sim-to-real: treinando para um mundo menos perfeito

**Vídeo 13 da série.** Um atuador precisa acompanhar uma velocidade de referência. Comparamos dois controladores do mesmo tamanho: um selecionado num simulador nominal e outro selecionado com ganho, arrasto, atraso e ruído sorteados. Avaliamos ambos sem mexer mais na política.

```bash
python3 -m unittest discover -v
python3 sim2real.py
```

A planta é um modelo escalar didático:

- A ação é um comando entre −1 e +1; uma fila introduz atraso inteiro em passos.
- A velocidade evolui como `v += 0.15 * (ganho * a_aplicada - arrasto * v)`.
- A política vê somente referência e velocidade medida com ruído. O avaliador usa a velocidade simulada sem ruído para calcular a recompensa.
- O objetivo penaliza erro quadrático de rastreamento, esforço e variação do comando. A referência inverte na metade dos 80 passos.
- O treino é busca em grade de 35 políticas por recompensa acumulada/média, não PPO. Após escolher dois ganhos, a política fica congelada.

## Medição local

O treino randomizado usa 36 cenários. A avaliação usa as mesmas condições para os dois controladores: caso nominal, 100 cenários independentes no mesmo intervalo de treino (`holdout`) e 100 cenários com ganho mais fraco e atraso maior que os vistos no treino (`estresse`). Cada conjunto tem semente explícita.

| Treino | Avaliação | Recompensa média ↑ | Erro absoluto médio ↓ | Δação² médio ↓ |
| :-- | :-- | --: | --: | --: |
| Nominal | Nominal | −0,2398 | 0,2897 | 0,0350 |
| Randomizado | Nominal | −0,2407 | 0,3137 | 0,0343 |
| Nominal | Holdout | −0,3698 | 0,3737 | 0,0440 |
| Randomizado | Holdout | −0,3658 | 0,3887 | 0,0395 |
| Nominal | Estresse | −0,7183 | 0,6615 | 0,0464 |
| Randomizado | Estresse | −0,6996 | 0,6684 | 0,0436 |

A randomização melhorou **a recompensa escolhida** nesses dois conjuntos de avaliação, não todas as métricas: a ação ficou mais suave, mas o erro absoluto médio aumentou. Em estresse, ambas as políticas perderam desempenho. Isso mostra por que validar métricas separadas e condições fora da faixa importa.

**Experimento:** coloque zero no peso de `delta**2` e compare as métricas novamente. Como alternativa, altere só a semente de avaliação; não selecione ganhos usando os cenários de teste.

Tudo aqui é simulado. Não houve exportação, teste físico, garantia de estabilidade nem validação de segurança. O vídeo explica como essa ideia se estende a simuladores físicos, identificação, políticas treinadas com PPO e implantação com proteções independentes.
