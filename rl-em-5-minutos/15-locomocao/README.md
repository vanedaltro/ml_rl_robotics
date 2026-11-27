# 15 · Como um robô aprende a andar

O capítulo final da série (vídeo 16) separa **escolher alvos articulares** de **executar esses alvos**. O comando de velocidade faz o papel do planejador; a política produz alvos e o servo proporcional-derivativo (PD) calcula as ações entre as decisões da política.

```text
comando + observação → política → alvos de junta → PD → simulador
                            ↑                            │
                            └──── feedback do estado ────┘
```

## Laboratório principal: duas pernas virtuais em Python

Não precisa de GPU, janela nem dependências externas. Rode os comandos dentro desta pasta:

```bash
python3 -m unittest discover -v
python3 locomocao.py --seed 42 --iteracoes 14 --populacao 18
```

Uma política com quatro parâmetros escolhe a amplitude das pernas alternadas e a correção de equilíbrio. Ela usa comando longitudinal, fase, inclinação e velocidade angular do tronco para gerar dois alvos a **25 Hz**. O PD roda a **200 Hz**: a cada subpasso, calcula `30 * (alvo - q) - 1.7 * dq` e limita a saída a ±5 nas unidades do modelo. O estado é integrado com passos de 5 ms.

O retorno soma uma recompensa por seguir o comando e penalidades por inclinação, mudança brusca de alvo e esforço. A queda simulada (|inclinação| ≥ 0,50 rad) encerra o episódio e desconta cinco pontos. O currículo aumenta gradualmente a faixa de comandos, varia a tração e aplica uma perturbação a meio episódio. Nenhuma política treinada vê as 20 sementes reservadas para avaliação.

Em vez de uma rede treinada com PPO, este laboratório usa **CEM** (método da entropia cruzada): sorteia parâmetros, avalia seu retorno episódico e reajusta a distribuição aos melhores candidatos. O formato de passada já está no código; o algoritmo aprende a parametrizá-la, não descobre uma marcha do zero.

Execução local no Mac, semente 42, médias de 20 cenários fora do treino:

| Política | Retorno | Quedas | Erro de velocidade nos episódios completos | Passos médios |
| :-- | --: | --: | --: | --: |
| Alvos neutros | 29,57 | 20/20 | n/a | 76,8 |
| Política inicial | 78,60 | 0/20 | 0,344 | 125,0 |
| Política treinada | 87,37 | 0/20 | 0,295 | 125,0 |

A comparação mais útil é treinada versus inicial: ambas completam os episódios, mas a treinada reduz o erro médio. Contato com o chão é prescrito por fase; não há cálculo de forças de contato, terreno com geometria ou equivalente físico da altura do tronco. O experimento ilustra a arquitetura e a recompensa, não a segurança de um robô.

## Extensão opcional: Hopper-v5 com MuJoCo e Gymnasium

Use um ambiente Python 3.13 à parte. O programa é *headless*: não cria janela, vídeo ou contexto de renderização.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mujoco.txt
python -m unittest discover -v
python hopper.py --seed 42 --iteracoes 14 --populacao 24 --horizonte 240
```

Aqui o modelo é o **Hopper** 2D de uma perna e três juntas do Gymnasium. A política usa seno, cosseno e inclinação para gerar três posições-alvo. O PD converte os alvos em **comandos normalizados** entre −1 e 1 para os atuadores do ambiente, não em valores diretos em N·m. O controlador roda uma vez a cada passo do Gymnasium, a **125 Hz**, e atualiza os alvos a **31,25 Hz**. O CEM usa a recompensa nativa do Hopper (saúde, avanço e custo de controle) e as mesmas duas sementes de ajuste para cada candidato. A avaliação usa outras oito sementes e registra separadamente término pelo ambiente e truncamento por tempo.

Teste local no Mac com Python 3.13.5, Gymnasium 1.4.0 e MuJoCo 4.0.1:

| Política | Retorno | Deslocamento | Passos | Término do ambiente | Limite de tempo |
| :-- | --: | --: | --: | --: | --: |
| Alvo neutro | 239,10 | −0,007 m | 240,0 | 0/8 | 8/8 |
| Política CEM | 270,68 | +0,418 m | 219,6 | 8/8 | 0/8 |

O resultado mostra o compromisso: a política ajustada avançou mais e ganhou maior retorno, mas todos os episódios de avaliação terminaram antes do limite. **Não é uma demonstração de marcha estável.** Não se pode extrapolar deste Hopper ou do modelo de duas pernas para um quadrúpede real.

Para experimentar, altere o custo de inclinação ou retire as perturbações do currículo no laboratório principal. Mantenha as mesmas sementes de avaliação e confira se o retorno sobe por seguir melhor o comando ou apenas por sobreviver mais tempo.

Leituras: [Hopper-v5, documentação oficial](https://gymnasium.farama.org/environments/mujoco/hopper/) · [Simulação no MuJoCo](https://mujoco.readthedocs.io/en/stable/programming/simulation.html) · [Rudin et al., *Learning to Walk in Minutes*](https://proceedings.mlr.press/v164/rudin22a.html).
