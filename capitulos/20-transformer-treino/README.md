# 20-transformer-treino

Laboratório independente em pt-BR. Título: **Capítulo 20: treinando um mini Transformer causal do zero com PyTorch**.

## Executar

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python train.py
python figures.py
```

Windows: `.venv\Scripts\activate`. Python 3.9 foi usado na validação, mas uma versão suportada por torch 2.8 pode ser utilizada. `requirements-lock.txt` registra dependências do ambiente Mac arm64; não é lock universal. O script usa CPU explicitamente por padrão (todos os tensores/modelos permanecem em CPU), dois threads e algoritmos determinísticos. Resultados exatos entre plataformas/versões não são garantidos.

## Conteúdo e saídas

- `article.md`: artigo pronto, com referências primárias e LaTeX; imagens em caminhos relativos.
- `train.py`: dados, modelo, treinamento, comparação e geração.
- `test_train.py`: cinco testes estruturais/funcionais.
- `figures.py`: três diagramas PNG reproduzíveis, incluindo fórmulas renderizadas.
- `results.json`, `splits.json`, `model.pt`: resultados e dados separados incluídos; o checkpoint é gerado localmente, não versionado.
- `images/01-shift-causal.png`, `02-pretrain-finetune.png`, `03-resultados-ppl.png`: artes do artigo/post.

O checkpoint contém state_dict e vocabulário; não contém estado do otimizador para retomar exatamente uma corrida. Só carregue checkpoints de origem confiável. Os exemplos não aceitam tokens desconhecidos nem fazem padding. Vocab fixo vem da especificação sintética, não do teste.

## Evidência

Mac arm64, Python 3.9.6, torch 2.8.0, numpy 2.0.2, matplotlib 3.9.4. `python -m unittest -v`: 5/5 PASS. `python train.py`: exit 0, 300 passos gerais + 150 ajuste + 150 técnico do zero. Tempo interno medido 6,1204 segundos, sem instalação/importação inicial. Dados por domínio 1075/230/231; teste técnico 1155 alvos; parâmetros 61486.

PPL técnica: unigram21.8578146; geral sem ajuste341.9084615; adaptação4.5101336; técnico do zero4.5762314. PPL geral pré4.4482605/pós64.0780535. Seed20261126. Checkpoints escolhidos por validação a cada25 passos; teste só consultado após treinamento das rotas, sem seleção baseada em teste. Não houve busca de hiperparâmetros ou reexecução para escolher uma corrida favorável.

Limitações: frases sintéticas com gramática compartilhada, um seed, baseline simples, adaptação e treino do zero com orçamentos totais diferentes. Não mede linguagem aberta, segurança, alinhamento, conhecimento factual ou qualidade de um LLM de produção.
