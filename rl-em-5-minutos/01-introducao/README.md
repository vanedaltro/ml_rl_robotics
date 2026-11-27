# RL em 5 minutos · 01 — O que é Reinforcement Learning?

[Artigo com fórmulas e diagramas](artigo.md) · [Índice da série](../README.md)

O mesmo robô joga dois episódios. A rota até a bandeira tem cinco recompensas de −1 e termina com +10. A tentativa que bate na borda e cai tem −1 e −10. São trajetórias definidas à mão, não um treinamento.

## Rodar

Python 3.8+; não precisa instalar pacotes nem baixar dados para o laboratório.

```bash
python3 rl-em-5-minutos/01-introducao/grid.py
```

Resultado esperado: 8 verificações; retorno **+5** para a bandeira, **−11** para a queda e **1,8098** para a bandeira com desconto 0,9.

O MP4 acompanha a edição em vídeo e não está versionado aqui. Para regenerar as figuras, instale Matplotlib e execute `python3 figuras.py` dentro desta pasta.
