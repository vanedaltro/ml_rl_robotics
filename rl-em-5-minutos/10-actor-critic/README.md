# 10 · Actor-Critic

No tabuleiro 4 × 3, cada casa tem logits do ator e um valor do crítico. Recompensas: bandeira +1, buraco −1, passo −0,04. Uma amostra da transição basta para atualizar os dois em TD(0).

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python actor_critic.py --episodios 300 --seed 7
```

O alvo do crítico é $r+\gamma V(s')$ em transições não terminais e $r$ na bandeira ou no buraco. O corte de 60 passos só limita a coleta e mantém o bootstrap. O erro $\delta=\mathrm{alvo}-V(s)$ ajusta o crítico e estima a vantagem para os logits do ator; $\gamma^t$ pesa a contribuição no objetivo a partir do início. Os exemplos numéricos do vídeo são reproduzidos: alvo 5,50 e erro +1,50; alvo 2,70 e erro −1,30.

No Mac (CPU, PyTorch 2.9.1, semente 7), o retorno descontado observado foi −0,333 nas primeiras 30 trajetórias e +0,357 nas últimas 30. A política gulosa final percorreu C, C, D, D, D até a bandeira. Um exemplo independente em PyTorch verifica o backward no corpo compartilhado e nas duas cabeças, sem treinar a rede. O erro TD é constante na perda do ator; o crítico aprende pela própria perda. A2C/A3C não estão implementados.
