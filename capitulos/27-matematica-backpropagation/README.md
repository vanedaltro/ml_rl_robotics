# Capítulo 27 · Do símbolo ao código

Backpropagation de duas camadas densas com NumPy em float64. O PyTorch serve de referência para comparar loss e gradientes de entradas, pesos e vieses. A CNN é apresentada apenas para uma janela no artigo; o backward completo fica para o próximo capítulo.

~~~bash
python -m pip install -r requirements.txt
python lab.py
python -m unittest -v test_lab.py
~~~

Para rodar sem PyTorch: instale NumPy e rode `python lab.py --numpy-only`. Opcionalmente, instale matplotlib e rode `python gerar_figuras.py`. Consulte `artigo.md` e `resultados.json`.

[Artigo no LinkedIn](https://www.linkedin.com/pulse/do-s%C3%ADmbolo-ao-c%C3%B3digo-matem%C3%A1tica-backpropagation-antes-santos-daltro-wesbe/) · [Post do capítulo](https://www.linkedin.com/feed/update/urn:li:ugcPost:7532073941410992131/).
