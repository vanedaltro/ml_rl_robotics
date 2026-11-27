"""Exemplo opt-in de SFT em LLM. Confira versão, licença e hardware."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Checkpoint local ou ID público autorizado")
    p.add_argument("--train", required=True, help="JSONL com prompt e completion de texto")
    p.add_argument("--validation", required=True, help="JSONL separado do treino")
    p.add_argument("--out", default="saida_lora")
    args = p.parse_args()
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data = load_dataset("json", data_files={
        "train": args.train, "validation": args.validation,
    })
    config = SFTConfig(
        output_dir=args.out, max_length=512, completion_only_loss=True,
        learning_rate=2e-4, num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        eval_strategy="epoch", save_strategy="epoch", report_to="none",
        bf16=False, fp16=False,
    )
    adapter = LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16,
        target_modules="all-linear", bias="none",
    )
    trainer = SFTTrainer(model=args.model, args=config,
                         train_dataset=data["train"],
                         eval_dataset=data["validation"], peft_config=adapter)
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
