# 07 · Q-learning e SARSA

O robô do grid agora precisa contornar um penhasco num tabuleiro de 4 × 12. Cada passo custa 1 ponto; uma queda custa 100 e devolve o robô ao início **no mesmo episódio**. Só a meta termina o episódio; o limite de 500 passos apenas interrompe a coleta.

```sh
python3 -m unittest discover -v
python3 cliff.py --episodios 500 --repeticoes 12 --seed 42
```

Os dois algoritmos agem com ε-greedy (ε = 0,1). Ambos usam $Q \leftarrow Q + \alpha(y-Q)$, mas calculam alvos diferentes:

- **Q-learning (off-policy):** $y = r + \gamma \max_{a'} Q(s',a')$. O alvo supõe que a próxima escolha será gulosa, mesmo se o comportamento explorar.
- **SARSA (on-policy):** $y = r + \gamma Q(s',a')$. A ação $a'$ é sorteada pela política de comportamento e usada no passo seguinte, se a coleta continuar.
- Na meta, ambos usam $y=r$. No exemplo do vídeo, $Q=2$, $r=-1$, $\max Q(s')=6$, $\gamma=0{,}9$ e $\alpha=0{,}5$ dão alvo $4{,}4$ e novo valor $3{,}2$. O experimento usa $\gamma=1$ e $\alpha=0{,}5$.

Resultado no Mac: 12 sementes de 42 a 53, média dos últimos 100 episódios de cada treino. A avaliação abaixo usa a política puramente gulosa e uma única tabela treinada com seed 42 por método.

| Algoritmo | Retorno médio no treino | Quedas por episódio no treino | Passos na avaliação gulosa |
| --- | ---: | ---: | ---: |
| Q-learning | −51,74 | 0,351 | 13 |
| SARSA | −30,07 | 0,062 | 17 |

O Q-learning aprendeu uma rota curta junto ao penhasco; o SARSA preferiu uma rota mais longa, com menor exposição durante o treino exploratório. Isso é um resultado deste ambiente, com ε fixo, não uma garantia geral. O programa imprime a rota percorrida com setas em ASCII. A série começa com tabelas; no episódio seguinte, uma rede passa a devolver os valores.
