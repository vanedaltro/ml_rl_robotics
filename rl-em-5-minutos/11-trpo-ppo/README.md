# 11 · TRPO, PPO e primeiro treino

Primeiro, um problema de duas ações mostra uma atualização de política aceita depois de reduzir o passo até respeitar a KL. Em seguida, PPO-Clip aprende a chegar à bandeira no GridWorld do capítulo 10.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python ppo.py --atualizacoes 16 --seed 7
```

**Região de confiança.** A política inicial tem probabilidade 0,5 em cada ação. Calculamos o gradiente do objetivo surrogado e testamos passos menores até melhorar o objetivo sob $D_{KL}(\pi_{ant}\|\pi_{nova})\le0{,}01$. Esse retrocesso com gradiente comum não implementa o TRPO completo: não calcula a direção de segunda ordem nem usa gradiente conjugado.

**PPO.** Em cada atualização:

1. Coletamos 128 passos de cada uma de oito cópias vetorizadas do grid em CPU. Guardamos logits e log-probabilidades antigas sem gradiente.
2. Calculamos GAE com $\gamma=0{,}99$ e $\lambda=0{,}95$. Em término verdadeiro, não há bootstrap. No limite de tempo, há bootstrap com a observação anterior ao reset, mas a vantagem não atravessa episódios. O fim do lote usa o próximo valor se não terminal.
3. Normalizamos vantagens no lote e usamos o alvo não normalizado $V_{ant}(s)+\hat A$. Otimizamos quatro épocas, com razão de probabilidades, clip de 0,2, perda quadrática do crítico, bônus de entropia, Adam e norma máxima de gradiente de 0,5.
4. Medimos a KL exata das distribuições categóricas nos estados amostrados e a fração de razões fora da banda. Depois descartamos o lote.

O clip limita o incentivo de certas mudanças no objetivo. Não garante que todas as razões fiquem na banda nem impõe um limite rígido de KL. As observações têm escala fixa conhecida. Esta demonstração tem dinâmica e ações discretas; não valida controle robótico contínuo.

**Mac, CPU, PyTorch 2.9.1, semente 7.** Região de confiança: passo aceito 0,250, KL 0,00779, probabilidade da segunda ação 0,5622. PPO (16.384 transições): retorno médio não descontado de −1,275 para +0,817; taxa de chegada de 23,4% para 100% em 64 episódios com semente fixa de avaliação. A versão gulosa final obteve +0,840. Última atualização: KL 0,00678, fração de clip 0,124, entropia 0,333. Isso descreve este ensaio, não uma garantia geral.

Referências: [TRPO](https://proceedings.mlr.press/v37/schulman15.html), [PPO](https://arxiv.org/abs/1707.06347) e [GAE](https://arxiv.org/abs/1506.02438).
