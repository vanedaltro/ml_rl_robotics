# 03 · Processo de decisão de Markov

O tabuleiro do episódio 1 ganhou gelo. A posição do robô basta como estado porque a aderência é fixa e cada desvio é independente do histórico. Se o piso ou a bateria variam e afetam o próximo passo, a posição sozinha pode deixar de ser suficiente.

```bash
python3 -m unittest discover -v
python3 mdp.py
```

- **S e A:** casas válidas e movimentos cima, direita, baixo, esquerda.
- **P e R:** `modelo(s, a)` enumera as probabilidades, destinos e recompensas; `passo` sorteia um ramo. Parede e borda deixam o robô na mesma casa e probabilidades com o mesmo destino são somadas.
- **γ:** `retorno` soma as recompensas com desconto. A decisão usa a ação pedida, não a direção que o piso acabou impondo.

Na casa (1, 3), pedir direita leva a (1, 4), (0, 3) ou (2, 3) com probabilidades 80%, 10% e 10%. Na execução com semente 42 (10.000 amostras), apareceram **80,31%, 9,89% e 9,80%**, respectivamente. Não são garantias por amostra, mas frequências aproximadas.

No piso sem gelo, a rota do episódio 1 dá **G = 5** sem desconto e **G = 1,8098** com γ = 0,9. Num exemplo separado, com passos intermediários gratuitos, γ = 0,5 favorece a moeda imediata de +2; γ = 0,9 favorece o tesouro de +10 recebido no sexto passo. Experimente mudar a aderência para 1 e depois para 0,5 e inspecionar o modelo junto a uma parede.
