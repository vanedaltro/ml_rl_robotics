# 13 · RL com modelo: ensaiar antes de agir

**Vídeo 14 da série.** O robô atravessa um corredor de seis casas até a bandeira. Andar avança uma casa; saltar tenta avançar duas, mas o robô pode ficar parado. O risco é maior nas casas 2 e 3. Cada passo custa −1, exceto a chegada que vale +9 e encerra o episódio.

```bash
python3 -m unittest discover -v
python3 model_based.py
```

## Aprender o modelo e replanejar (MPC)

1. Coletamos transições `(estado, ação, próximo_estado, recompensa)` no ambiente. Como é um simulador didático, podemos reiniciar em qualquer casa e coletar o mesmo número de amostras para cada par.
2. O modelo estima `P(próximo_estado, recompensa | estado, ação)` por frequências. Ele só vê os dados coletados. O planejador não consulta as probabilidades do simulador.
3. Com horizonte H, calculamos `valor = média prevista de [r + 0,95 × valor restante]`. O valor restante é zero no terminal ou ao esgotar o horizonte.
4. Executamos apenas a primeira ação. Se o salto falhar e a casa observada não mudar, a decisão seguinte parte dessa casa, não de onde o plano esperava que o robô estivesse. Como o modelo e H não mudam, o código guarda decisões já calculadas para a mesma casa.

Um modelo com uma amostra por par pode acreditar que certo salto sempre funciona: no futuro imaginado, sucessivos saltos repetem essa suposição. Replanejar evita seguir um plano desatualizado, mas não corrige sozinho um modelo enviesado.

## Resultados medidos

A precisão de um passo foi medida pelo Brier multiclasse do próximo estado: soma dos erros quadráticos entre probabilidades previstas e o vetor que marca apenas o estado observado. Brier menor é melhor. A base de teste tem 100 transições novas por par, semente 99. O retorno é a média de 300 episódios de avaliação, semente 101, com limite de 25 passos e desconto 0,95. As duas densidades de dados foram treinadas com 20 sementes cada; a tabela mostra médias entre os modelos.

| Estratégia | Brier independente ↓ | Retorno no ambiente ↑ | Chegou |
| :-- | --: | --: | --: |
| Só andar | n/a | 2,4396 | 100% |
| MPC, H=5, 1 amostra/par | 0,3505 | 3,6795 | 100% |
| MPC, H=1, 80 amostras/par | 0,1807 | 3,2707 | 100% |
| MPC, H=5, 80 amostras/par | 0,1807 | 4,1813 | 100% |

Chegar não distingue estas políticas: o retorno descontado recompensa chegar mais cedo. O planejador H=1 não consegue valorizar a chegada partindo das primeiras casas da mesma forma que H=5. Um horizonte maior não é garantia de melhor retorno com qualquer modelo.

## Dyna-Q: usar cada amostra de novo

O mesmo script compara Q-learning sem planejamento a Dyna-Q. Cada réplica recebe a mesma sequência de 36 transições coletadas no ambiente. Dyna mantém o modelo atualizado e, a cada amostra coletada, sorteia dez transições do modelo para executar a mesma atualização Q. Não são dez interações novas com o ambiente.

Em 20 réplicas, com 50 episódios de avaliação por política aprendida, o retorno médio foi **3,4820** sem planejamento e **3,4869** com dez atualizações imaginadas. A diferença é pequena. Mais computação não garante um salto de desempenho, principalmente se os dados iniciais não cobrem bem o ambiente.

**Experimento:** mude o número de amostras por par e compare planejamento e previsão de um passo. Confira se a conclusão resiste a outras sementes antes de generalizar.

Este laboratório usa uma tabela, não uma rede, e nenhuma ação é executada em hardware. A ideia prevê o próximo estado e recompensa; modelos latentes como os do vídeo fazem isso em representações aprendidas. O exemplo não implementa Dreamer, AlphaZero nem MuZero.
