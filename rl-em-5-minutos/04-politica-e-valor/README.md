# 04 · Política e valor

Um pequeno corredor isola a conta de Bellman do vídeo. Do início, ir à direita rende −1 e leva ao corredor com 80% ou encerra no buraco com 20%. Do corredor, avançar até a bandeira paga +10 no passo da chegada. O buraco deste exemplo encerra a tarefa sem penalidade adicional; é uma adaptação para reproduzir a conta do vídeo.

```bash
python3 -m unittest discover -v
python3 valores.py
```

**Vπ(s)** avalia a política a partir de um estado. **Qπ(s, a)** força a primeira ação e segue a mesma política depois. A equação usada é `Q = soma(p × (r + γ × V(próximo)))`, sem continuação após um terminal; depois, `V = soma(π(a | s) × Q)`.

| Política | V(início) via Bellman | Q(direita) | Q(esperar) | Média de 10.000 retornos |
| --- | ---: | ---: | ---: | ---: |
| Direta | 6,2000 | 6,2000 | 4,5800 | 6,2378 |
| Mista (70% direita, 30% esperar) | 5,5342 | 6,2000 | 3,9808 | 5,4904 |

A política mista espera às vezes e paga −1 antes de tentar de novo. O código compara a avaliação iterativa com simulações semeadas; a média amostral não precisa coincidir com a expectativa exata. Experimente aumentar a chance de esperar e veja como o valor de início muda.
