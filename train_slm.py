import os
import datetime
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)

import wandb

print("TrainingArguments module:", TrainingArguments.__module__)


def main():
    # --- 0. Init wandb project ---
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"slm_{now}"
    wandb.init(project="slm-tinyllama", name=run_name, resume="allow")
    

    # --- 1. Load cleaned dataset ---
    print("Loading cleaned dataset...")
    dataset = load_dataset("json", data_files="dataset/clean_kids_dataset.jsonl")

    # Optional: Split into train/validation (e.g., 90/10)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # --- 2. Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tinyllama/tinyllama-1.1B-chat-v1.0")

    # --- 3. Tokenize dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 4. Data collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 5. Load model ---
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("tinyllama/tinyllama-1.1B-chat-v1.0")

    # --- 6. Training setup ---
    training_args = TrainingArguments(
        output_dir="./slm_tinyllama_output",
        overwrite_output_dir=False,  # Set to False to allow resuming
        num_train_epochs=3,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb",  # Enable wandb
        run_name="run-small-language-model",
        resume_from_checkpoint=True,
        fp16=True, 
        learning_rate=5e-5,
        weight_decay=0.01,  # add weight decay
        warmup_steps=500,

        metric_for_best_model="eval_loss",  # add this!
        greater_is_better=False,            # lower eval_loss is better
    )

    # --- 7. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )


    # --- 8. Train ---
    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    # --- 9. Save final model & tokenizer ---
    print("Saving model and tokenizer...")
    trainer.save_model("./slm_tinyllama_final")
    tokenizer.save_pretrained("./slm_tinyllama_final")

    print("Training complete!")

if __name__ == "__main__":
    main()
