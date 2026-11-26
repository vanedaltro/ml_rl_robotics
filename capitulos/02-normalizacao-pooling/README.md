# 02 · BatchNorm, LayerNorm, pooling, losses e otimizadores

Normalização, pooling, perdas, otimizadores e benchmark de CPU. Os scripts de treino e benchmark usam FashionMNIST. Os CSVs e relatórios são da execução original.

[Ler o artigo no LinkedIn](https://www.linkedin.com/pulse/pytorch-normaliza%C3%A7%C3%A3o-pooling-otimizadores-e-sa%C3%ADdas-em-santos-daltro-zar3e/)

Entre nesta pasta e use um ambiente virtual. Instale:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python saidas.py
python laboratorio.py --epocas 3 --seed 42
python benchmark_cpu.py --min-time .35
```

Repetir o experimento pode substituir métricas no diretório atual; use uma cópia se quiser preservar os registros originais.

Para reproduzir o benchmark de renderização em Linux headless, instale `mujoco==3.6.0` e `PyOpenGL` e execute:

```bash
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 MESA_SHADER_CACHE_DISABLE=true LP_NUM_THREADS=2 python simulacao_cpu.py --frames 120 --reps 3
```

O ensaio original usou Mesa/llvmpipe. Não é um exemplo de controle do robô. `figuras.py` usa Pillow.
