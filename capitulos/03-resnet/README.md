# 03 · ResNet e gradientes

Compare fluxos de gradiente e redes com e sem conexões residuais. O treino usa FashionMNIST e gera os pesos locais usados na simulação.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/resnet-na-pr%C3%A1tica-por-que-o-gradiente-some-e-um-muda-santos-daltro-rqtse/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python mlp_gradiente.py
python residual_cpu.py --dados dados
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

Depois de rodar o treino e instalar `mujoco==3.6.0` e `PyOpenGL`, reproduza a parte opcional em Linux com Mesa/EGL:

```bash
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 MESA_SHADER_CACHE_DISABLE=true LP_NUM_THREADS=2 python simulacao_residual.py
```

O ensaio mede câmera e inferência, não controle de robô. `figuras.py` usa Pillow e Matplotlib.
