# 05 · Programação dinâmica

Em Coin Change, calcule primeiro os valores menores: `dp[i] = 1 + min(dp[i - moeda])`, com `dp[0] = 0`. O laboratório mostra a tabela de baixo para cima e a mesma recorrência de cima para baixo com memoização.

No tabuleiro, `V[s]` depende dos valores das casas seguintes. Como há ciclos, repetimos a equação de Bellman. **Value Iteration** aplica o máximo sobre as ações a cada varredura; **Policy Iteration** alterna avaliação e melhoria da política. Os dois usam o modelo conhecido, sem precisar aprender por tentativa e erro. O tabuleiro aqui é determinístico como no episódio 1.

```bash
python3 -m unittest discover -v
python3 programacao_dinamica.py
```

- Com moedas 1, 3 e 4, a tabela até o valor 6 é **[0, 1, 2, 1, 1, 2, 2]**. Para 6, a resposta ótima é 3 + 3, enquanto escolher sempre a maior moeda disponível dá 4 + 1 + 1.
- Com γ = 0,9, os dois planejadores chegam a **V(início) = 1,8098**. Nesta execução, Value Iteration usou sete varreduras; Policy Iteration avaliou três políticas (uma avaliação pode exigir muitas varreduras).
- A política gulosa devolveu `DCCDDD`, empatada com `DDDDCC`; ambas rendem cinco passos de −1 seguidos de +10. Teste alterar a posição da parede e compare a rota e o valor.
