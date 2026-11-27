# Capítulo 26 · Quando o modelo deve dizer “não sei”?

Laboratório em CPU que mede calibração e abstenção seletiva no claro, na penumbra e no escuro, além de uma classe não treinada. O pipeline separa treino, calibração, escolha da política e teste. Não precisa baixar imagens ou pesos.

```bash
python -m pip install -r requirements.txt
python lab.py > resultados.json
python -m unittest -v test_lab.py
```

Opcional: instale matplotlib e rode `python gerar_figuras.py`. Veja a metodologia, resultados agregados e referências no `artigo.md`. As imagens são sintéticas. A lógica de intervenção de um robô é ilustrativa, não testada em hardware.

[Artigo no LinkedIn](https://www.linkedin.com/pulse/quando-o-modelo-deve-dizer-n%C3%A3o-sei-calibra%C3%A7%C3%A3o-imagens-santos-daltro-ujexe/) · [Post do capítulo](https://www.linkedin.com/feed/update/urn:li:ugcPost:7532064539610099712/).
