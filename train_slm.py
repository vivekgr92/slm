import os
import datetime
import torch
from datasets import load_dataset
from datasets import DatasetDict, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

import wandb

print("TrainingArguments module:", TrainingArguments.__module__)



class LoggingEarlyStoppingCallback(EarlyStoppingCallback):
    def on_train_end(self, args, state, control, **kwargs):
        early_stopped = state.epoch < args.num_train_epochs
        wandb.log({
            "early_stopped": early_stopped,
            "completed_epochs": state.epoch,
            "completed_steps": state.global_step
        })
        if early_stopped:
            print(f"Early stopping triggered at epoch {state.epoch:.2f}, step {state.global_step}.")
        else:
            print(f"Training completed full {args.num_train_epochs} epochs.")

def count_total_tokens(tokenized_dataset):
    total_tokens = 0
    for example in tokenized_dataset:
        total_tokens += len(example["input_ids"])
    return total_tokens


def main():
    # --- 0. Init wandb project ---
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"slm_{now}"
    wandb.init(project="slm-tinyllama", name=run_name, resume="allow")

    # --- 1. Load cleaned dataset ---
    print("\nLoading cleaned dataset...")
    dataset = load_dataset("json", data_files="dataset/clean_kids_dataset.jsonl")

    # Optional: Split into train/validation (e.g., 90/10)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print("Total Train Samples", len(train_dataset))
    print("Total Val Samples", len(eval_dataset))

    # Log number of samples
    wandb.run.summary["num_train_samples"] = len(train_dataset)
    wandb.run.summary["num_eval_samples"] = len(eval_dataset)


    # --- 2. Load tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tinyllama/tinyllama-1.1B-chat-v1.0")

    # --- 3. Tokenize dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    TOKENIZED_DIR = "dataset/tokenized"

    if os.path.exists(TOKENIZED_DIR):
        print("\nLoading tokenized datasets from disk...")
        tokenized_datasets = load_from_disk(TOKENIZED_DIR)
        tokenized_train = tokenized_datasets["train"]
        tokenized_eval = tokenized_datasets["eval"]
    else:
        print("\nTokenizing datasets...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Save tokenized datasets to disk
        print("\nSaving tokenized datasets to disk...")
        tokenized_datasets = DatasetDict({
            "train": tokenized_train,
            "eval": tokenized_eval
        })
        tokenized_datasets.save_to_disk(TOKENIZED_DIR)


    # --- 4. Count total tokens ---
    train_token_count = count_total_tokens(tokenized_train)
    eval_token_count = count_total_tokens(tokenized_eval)
    total_token_count = train_token_count + eval_token_count

    print(f"Train token count: {train_token_count:,}")
    print(f"Eval token count: {eval_token_count:,}")
    print(f"Total token count: {total_token_count:,}")


    # Log token counts to W&B run summary (not charts)
    wandb.run.summary["train_token_count"] = train_token_count
    wandb.run.summary["eval_token_count"] = eval_token_count
    wandb.run.summary["total_token_count"] = total_token_count

    # --- 5. Data collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 6. Load model ---
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("tinyllama/tinyllama-1.1B-chat-v1.0")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        # Shift logits and labels to align
        shift_logits = torch.tensor(logits[..., :-1, :])
        shift_labels = torch.tensor(labels[..., 1:])
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss.item()) if loss.item() < 100 else float("inf")

        # Log to wandb
        wandb.log({
            "eval_loss": loss.item(),
            "eval_perplexity": perplexity
        })

        return {
            "eval_loss": loss.item(),
            "eval_perplexity": perplexity
        }

    # --- 7. Training setup ---
    training_args = TrainingArguments(
        output_dir="./slm_tinyllama_output",
        overwrite_output_dir=False,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb",
        run_name=run_name,
        resume_from_checkpoint=False,
        fp16=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=1,  # prevents building up large memory blocks
        load_best_model_at_end=True,
    )

    # --- 8. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
        callbacks=[LoggingEarlyStoppingCallback(early_stopping_patience=3)]
    )

    # --- 9. Train ---
    print("\nStarting training...")
    torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=False)

    # --- 10. Save model & tokenizer ---
    print("\nSaving model and tokenizer...")
    trainer.save_model(f"./{run_name}_final")
    # tokenizer.save_pretrained("./slm_tinyllama_final")

    print("Training complete!")


if __name__ == "__main__":
    main()
