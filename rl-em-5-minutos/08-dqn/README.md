# 08 · DQN: a tabela vira rede

Voltamos ao tabuleiro 3 × 5 do episódio 01. A meta vale +10, o buraco −10 e os outros passos −1. Uma rede recebe a posição normalizada (linha e coluna) e devolve um valor Q para cada ação. Aqui a posição basta para descrever o estado. No Atari, o DQN usa uma rede convolucional e quadros empilhados para observar movimento; esse exemplo didático não processa pixels.

Use Python 3.10 ou mais recente e PyTorch 2.9.1. Não precisa de GPU ou de baixar dados.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python dqn.py --episodios 250 --seed 42
```

O agente coleta com ε-greedy. A memória guarda até 2.048 transições e sorteia lotes de 32 para reduzir a dependência entre amostras vizinhas. A rede online prevê $Q_\theta(s,a)$ para a ação tomada; a rede-alvo calcula $y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$. O otimizador atualiza só a rede online para reduzir $(Q_\theta(s,a)-y)^2$. A rede-alvo é copiada a cada 50 atualizações. Na meta ou no buraco, $y=r$. O corte externo de 40 passos não zera o termo futuro.

Validação no Mac, PyTorch 2.9.1 em CPU, 250 episódios por semente. A avaliação gulosa é uma trajetória determinística por rede para este tabuleiro. O número de metas durante o treino inclui a fase de exploração.

| Semente | Passos no treino | Metas no treino | Retorno guloso antes → depois | Caminho final |
| ---: | ---: | ---: | ---: | --- |
| 7 | 2.137 | 191 | −40 → +5 | DDDDCC |
| 23 | 2.026 | 192 | −40 → +5 | DCCDDD |
| 42 | 2.123 | 195 | −40 → +5 | DCCDDD |

C/D/B/E significam cima, direita, baixo e esquerda. Os dois caminhos finais chegam em seis ações, cinco com recompensa −1 e a última com +10. Esse teste mostra que a rede aprendeu a resolver este tabuleiro; não mede generalização para layouts ou estados novos. A saída de tamanho fixo funciona para quatro ações discretas. Para controlar diretamente um torque contínuo, a série passa a aprender uma política.
