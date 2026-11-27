# Treinamento distribuído com PyTorch: duas máquinas

- `artigo.md`: capítulo em português com diagramas.
- `post.txt`: rascunho para LinkedIn; preencher o link quando o artigo estiver no ar.
- `train_ddp.py`: MLP com dataset sintético, DDP, métricas globais e checkpoint.
- `figuras/`: PNGs para LinkedIn; versão vetorial SVG também disponível.

## Primeiro execute na sua máquina

Com PyTorch instalado no ambiente, execute:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  train_ddp.py --device cpu
```

## Depois use dois nós Linux

O exemplo usa `192.168.1.10` como IP ilustrativo do nó A. Use o IP privado real de A nos dois comandos. Execute um comando em cada computador, com os mesmos arquivos e ambiente. A rede deve permitir tráfego entre os processos, não somente para a porta inicial de encontro.

```bash
# Nó A
python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr=192.168.1.10 --master-port=29500 train_ddp.py --device cpu

# Nó B
python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr=192.168.1.10 --master-port=29500 train_ddp.py --device cpu
```

Para CUDA/NCCL, use Linux com GPUs NVIDIA e uma build compatível do PyTorch; troque para `--device cuda` nos dois comandos. Não exponha o rendezvous à internet.

## Reprodução e testes

Para reproduzir a validação TCP local sem duas máquinas físicas, use os mesmos comandos dos dois nós, mas com `--master-addr=127.0.0.1` e o mesmo diretório de trabalho. Selecione uma porta livre. Em seguida rode:

```bash
DDP_CHECKPOINT=ddp.pt python -m unittest -v test_train_ddp
```

Os resultados de referência em `resultados.txt` são do teste CPU/Gloo com PyTorch 2.9.1, no mesmo Mac, e não são benchmark de uma rede física.

## Duas GPUs NVIDIA no mesmo servidor

Em Linux com uma build CUDA/NCCL compatível, o argumento abaixo sobe um processo por GPU visível. Para duas GPUs por nó em dois servidores, use os comandos de dois nós acima com `--nproc-per-node=2` e `--device cuda` nos dois.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ddp.py --device cuda
```

CUDA/NCCL e rede física não foram testados neste laboratório.

## Inferência: réplicas ou estágios

O [artigo](artigo.md#9-inferência-duas-réplicas-ou-um-modelo-dividido) tem os comandos de ambos os exemplos. Gere `ddp.pt` primeiro:

```bash
torchrun --standalone --nproc-per-node=2 train_ddp.py --device cpu
DDP_CHECKPOINT=ddp.pt python -m unittest -v test_train_ddp test_inference_ddp
```

Os servidores HTTP são didáticos e escutam em loopback por padrão. Use apenas em rede privada confiável e carregue somente checkpoints confiáveis. O cliente alterna entre réplicas, não é um balanceador de produção. No exemplo de estágios, cada processo carrega o checkpoint inteiro na inicialização, retendo depois apenas as camadas de sua etapa. Veja [a saída do teste local](infer_results.log).
