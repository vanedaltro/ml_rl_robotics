# Capítulo 27 · Do símbolo ao código

Backpropagation de duas camadas densas com NumPy em float64. O PyTorch serve de referência para comparar loss e gradientes de entradas, pesos e vieses. A CNN é apresentada apenas para uma janela no artigo; o backward completo fica para o próximo capítulo.

~~~bash
python -m pip install -r requirements.txt
python lab.py
python -m unittest -v test_lab.py
~~~

Para rodar sem PyTorch: instale NumPy e rode `python lab.py --numpy-only`. Opcionalmente, instale matplotlib e rode `python gerar_figuras.py`. Consulte `artigo.md` e `resultados.json`.
