# 14 · Aprendizado por reforço multiagente (MARL)

No vídeo 15, vários agentes aprendem enquanto dividem o mesmo ambiente. Aqui,
dois robôs escolhem simultaneamente entre os corredores 0 e 1 do armazém. Cada
episódio tem uma única decisão. Corredores diferentes valem **+1 para a equipe**;
o mesmo corredor causa colisão e vale **−1 para a equipe**.

| Robô A ↓ / Robô B → | Corredor 0 | Corredor 1 |
|---|---:|---:|
| Corredor 0 | −1 | +1 |
| Corredor 1 | +1 | −1 |

## Rode no seu computador

Só usa a biblioteca padrão do Python. Execute nesta pasta:

```bash
python3 -m unittest discover -v
python3 marl.py --sementes 32 --episodios 1200 --seed 42
```

O modo rápido é `python3 marl.py --sementes 4 --episodios 200 --seed 42`.
O programa imprime resultados determinísticos para a mesma versão do Python e
os mesmos parâmetros. Não é preciso baixar um simulador nem usar GPU.

## O que cada robô aprende?

**Aprendizes independentes.** Cada robô tem a própria tabela de duas ações e
usa uma política ε-greedy (ε = 0,15). A tabela estima a recompensa da equipe
dada a própria ação e não vê a escolha do parceiro. Se a probabilidade de o
parceiro escolher 1 mudar de 0,2 para 0,9, o retorno esperado de escolher 0
muda de −0,6 para +0,8. O armazém conserva as mesmas regras, mas cada robô
enxerga um alvo que muda com a política do outro.

**CTDE (treino centralizado, execução descentralizada).** Durante o treino, o
crítico recebe as duas ações e estima `Q(a_A, a_B)` com a média observada de cada
par. Como só há quatro pares, um ensaio de cada um dá a tabela exata neste
jogo determinístico. Depois os dois atores coletam lotes de até 20 episódios.
A atualização usa o gradiente esperado, calculado enumerando os quatro pares
sob as políticas do lote. Na execução, cada robô usa só seu próprio ator:
a ação mais provável de um não precisa consultar a do outro nem o crítico.
Os atores têm parâmetros distintos e não trocam mensagens.

Para atribuir crédito, compare a ação executada com a média das alternativas
do próprio agente, mantendo a ação do parceiro fixa:

```text
A_i = Q(a_i, a_outro) - Σ_b π_i(b) Q(b, a_outro)
∂log π_i(a_i)/∂logit_i = a_i - π_i(1)
```

Se o parceiro escolhe 0 e a política própria é 50/50, a referência contrafactual
é zero: escolher 1 dá vantagem +1 e escolher 0 dá −1. O robô não é removido do
jogo. Essa forma de crédito é inspirada no [COMA](https://arxiv.org/abs/1705.08926),
mas o laboratório não implementa o algoritmo completo.

## Verificação no Mac

Python 3.9.6 (Darwin arm64), 32 sementes consecutivas começando em 42;
1.200 episódios por método e semente. Para a métrica de execução, a escolha
gulosa de cada robô ficou congelada após o treino e o par foi avaliado uma
vez por semente, pois o ambiente e as ações finais são determinísticos.

| Método | Sem colisão no treino | Sem colisão na execução gulosa |
|---|---:|---:|
| Independente | 86,04% | 100,00% |
| CTDE | 87,93% | 100,00% |

Na semente 42, os atores CTDE terminam com probabilidades de escolher o
corredor 1 de 0,9933 e 0,0067; as decisões gulosas são (1, 0).
Ambos os métodos resolvem este jogo. O orçamento de episódios é igual, mas o
treino e a exploração diferem: o CTDE faz um ensaio inicial por par e começa
com políticas quase uniformes; o independente explora com ε fixo. Este
resultado não é um benchmark nem implica vantagem geral do CTDE. O exemplo
não modela movimento, sensores, física, comunicação aprendida ou hardware.

## E quando os agentes competem?

Em pedra, papel e tesoura, o vencedor recebe +1 e o perdedor −1. A melhor
resposta à última jogada pura gera o ciclo pedra → papel → tesoura → pedra.
A política uniforme não pode ser explorada: a melhor resposta tem valor
esperado 0; contra uma política que sempre joga pedra, papel rende +1.
Esse é um exemplo de equilíbrio de Nash, não um treino por *self-play*.

**Experimento:** diminua `--episodios` e observe como os rankings durante
o treino podem mudar, mesmo quando a execução gulosa ainda coordena.
