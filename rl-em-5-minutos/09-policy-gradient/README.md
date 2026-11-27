# 09 · Policy Gradient

O robô do episódio 1 troca uma tabela de valores por uma tabela de **logits da política**. O PyTorch calcula os gradientes, mas as probabilidades vêm de uma distribuição categórica por casa, sem rede profunda.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python reinforce.py --episodios 400 --seed 7
```

A trajetória fornece retornos $G_t$ do fim para o início. Minimizamos $-\sum_t \gamma^t\log\pi(a_t|s_t)(G_t-b(s_t))$, que equivale a maximizar o estimador REINFORCE do retorno descontado no início. O horizonte de 25 passos é parte do objetivo deste exemplo: ao acabar o horizonte, não há retorno futuro. A baseline só conhece retornos de episódios anteriores e não recebe gradiente do ator. Uma baseline independente da ação não muda a esperança do estimador; sua escolha pode afetar a variância. O exemplo gaussiano demonstra o gradiente para uma ação contínua fixa, sem treinar um robô contínuo.

No Mac (CPU, PyTorch 2.9.1, semente 7), o retorno nas primeiras/últimas 50 rodadas foi −9,252/+0,854 sem baseline e −9,062/+1,163 com baseline. A política gulosa final alcançou +1,810 descontados em ambos os casos. Isso é um ensaio didático, não uma comparação estatística entre métodos.
