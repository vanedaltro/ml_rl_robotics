# RL em 5 minutos

Uma trilha de **15 laboratórios**, do GridWorld em Python ao controle de locomoção em simulação. Cada pasta explica o que rodar, o que observar no resultado e onde o exemplo simplifica o mundo real. Os capítulos são independentes: você pode rodar um laboratório sem executar os anteriores.

| Nº | Capítulo | Vídeo(s) | Laboratório |
| :-- | :-- | :-- | :-- |
| 01 | O que é Reinforcement Learning? | 1 | [GridWorld em Python](01-introducao/) |
| 02 | Explorar ou aproveitar? Bandits | 2 | [Laboratório de bandits](02-bandits/) |
| 03 | Processos de decisão de Markov | 3 | [Estados, transições e desconto](03-mdp/) |
| 04 | Política e valor | 4 | [Políticas e mapas de valor](04-politica-e-valor/) |
| 05 | Programação dinâmica para RL | 5 | [Iteração de valores e políticas](05-programacao-dinamica/) |
| 06 | Monte Carlo e TD | 6 | [Predição com Monte Carlo e TD](06-monte-carlo-e-td/) |
| 07 | Q-learning e SARSA | 7 | [Comparação no CliffWalking](07-q-learning-e-sarsa/) |
| 08 | DQN: da tabela à rede | 8 | [DQN em PyTorch](08-dqn/) |
| 09 | Policy Gradient | 9 | [REINFORCE em PyTorch](09-policy-gradient/) |
| 10 | Actor-Critic | 10 | [Ator e crítico em PyTorch](10-actor-critic/) |
| 11 | TRPO, PPO e primeiro treino | 11 e 12 | [Região de confiança e treino com PPO](11-trpo-ppo/) |
| 12 | Do simulador ao robô real | 13 | [Randomização de domínio](12-sim2real/) |
| 13 | RL com modelo | 14 | [Planejamento com modelo](13-model-based/) |
| 14 | RL multiagente | 15 | [Coordenação entre agentes](14-marl/) |
| 15 | Como um robô aprende a andar | 16 | [Locomoção e controle PD](15-locomocao/) |

[Artigo do episódio 1, com vídeo](https://www.linkedin.com/pulse/o-que-%25C3%25A9-reinforcement-learning-um-rob%25C3%25B4-duas-rotas-e-santos-daltro-i77be/) · [Post do episódio 1](https://www.linkedin.com/feed/update/urn:li:activity:7532110045845684224/)

Leitura complementar: [A matemática do Reinforcement Learning: de Bellman ao PPO](https://www.linkedin.com/pulse/matem%25C3%25A1tica-do-reinforcement-learning-de-bellman-ao-santos-daltro-7hofe/).

## Como rodar e validar

Para um capítulo, entre na pasta e siga o README. A trilha foi validada com Python 3.13.5 e PyTorch 2.9.1 em CPU no macOS (Apple Silicon). Para verificar a trilha inteira a partir deste diretório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-labs.txt
python validar.py
```

Para rodar apenas alguns capítulos, por exemplo o MDP e o DQN, use `python validar.py 03 08`. O validador roda os testes e uma demonstração curta por laboratório. As demonstrações completas e seus parâmetros ficam nos READMEs individuais. Todos podem rodar em CPU; nenhum requer robô físico. Os vídeos ficam nas publicações da série, não no Git.

O laboratório 15 também tem uma execução opcional do Hopper-v5 em MuJoCo, sem janela gráfica:

```bash
python -m pip install -r 15-locomocao/requirements-mujoco.txt
python validar.py 15 --mujoco
```
