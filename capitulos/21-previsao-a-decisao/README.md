# Capítulo 21: da previsão à decisão

Laboratório em português: previsão supervisionada, clonagem comportamental e planejamento por recompensa, com um robô em uma pista discreta simulada. Sem controle de hardware, dados externos ou credenciais.

- [Artigo e diagramas](article.md)
- [Simulador e experimento](demo.py)
- [Testes](test_demo.py)
- [Métricas da execução de referência](results.json)

## Reproduzir

Com Python 3.10 ou mais recente. No macOS/Linux, rode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py --json resultados.json
python -m unittest -v
```

No Windows PowerShell, use `.venv\Scripts\Activate.ps1` para ativar o ambiente e os mesmos comandos Python. O experimento executa na CPU e fixa as sementes. Testado com Python 3.13 e PyTorch 2.9.1, macOS arm64; a biblioteca não garante reprodução idêntica entre versões ou plataformas.

## O que foi medido

A rede de dinâmica foi treinada em 80 episódios exploratórios secos (644 transições), com todos os pares de estado não terminal e ação cobertos. A política clonada viu 40 demonstrações no seco (264 decisões), sempre com o passo comum. Na avaliação nenhuma rede foi atualizada.

| Avaliação | Seco | Molhado |
|---|---:|---:|
| Predição correta de um passo | 300/300 | 279/321 |
| Predição no passo comum nas casas 3 e 5 | 26/26 | 0/42 |
| Chegadas com clonagem, de 32 | 32 | 0 |
| Chegadas planejando com o modelo aprendido, de 32 | 32 | 0 |
| Chegadas planejando com a regra exata, de 32 | 32 | 32 |

Os 32 episódios de exploração de teste por piso são novos (semente distinta do treino), mas repetem pares em um mundo pequeno e determinístico. As políticas foram avaliadas nos mesmos 32 inícios amostrados de 0 a 3, não em 32 mapas independentes. A regra exata é uma referência com acesso privilegiado à dinâmica do piso. Nenhum resultado implica segurança ou generalização em robótica física.

## Arquivos visuais

`01-pista.png` mostra transições e custos; `02-tres-perguntas.png` separa os três objetivos; `03-resultados.png` compara o teste em malha fechada e o recorte crítico. Para regenerar no Mac, use Python com Pillow e execute `python diagramas.py`. O script de figuras é opcional e não é necessário para executar o experimento.
