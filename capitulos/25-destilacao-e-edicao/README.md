# Capítulo 25 · Como um modelo pequeno aprende com um grande

Laboratório em PyTorch para comparar rótulos, alvos duros e alvos suaves mantendo a arquitetura e as entradas de treino fixas. Não usa API paga nem baixa dados. Veja `artigo.md` para teoria e `resultados.json` para números por semente. O laboratório também demonstra uma edição de um peso em um classificador linear. Não reproduzimos as edições de LLM citadas.

```bash
python -m pip install -r requirements.txt
python lab.py
python -m unittest -v test_lab.py
```

Opcional: `python -m pip install matplotlib` e `python gerar_figuras.py` para recriar os diagramas a partir dos resultados registrados. Para substituí-los, rode `python lab.py > resultados.json` antes. Referências primárias e limitações estão no artigo.
