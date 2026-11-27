# Capítulo 24: fine-tuning e esquecimento catastrófico

O laboratório compara um MLP completo, replay e um adapter residual em duas tarefas sintéticas com regras parcialmente conflitantes. O adapter roteado recebe a identidade da tarefa. Não compare o resultado roteado com os demais como se todos recebessem as mesmas informações.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1
python lab_finetuning.py
python -m unittest -v
```

`resultados.json` contém as três sementes obtidas em CPU. `gerar_figuras.py` usa Matplotlib para reproduzir as duas imagens, depois de executar o laboratório. As figuras também acompanham o capítulo.

## Receita opcional de LLM

`sft_lora.py` usa o formato JSONL de prompt e completion e uma validação separada. Prepare dados autorizados, um ambiente compatível e um checkpoint pequeno de sua escolha. A sintaxe e `--help` foram verificadas; o treino de LLM não foi executado neste laboratório.

```bash
python -m pip install "trl[peft]==0.26.0"
python sft_lora.py --model CAMINHO_OU_ID --train treino.jsonl --validation validacao.jsonl
```
