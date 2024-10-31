from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np
import evaluate
import torch

def main():
    # Load dataset
    dataset = load_dataset("grammarly/coedit")
    
    # Initialize tokenizer and model
    model_name = "t5-small"  # Changed to smaller model
    print(f"\nLoading model and tokenizer from {model_name}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print(f"Successfully loaded model and tokenizer")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    def preprocess_function(examples):
        inputs = ["correct: " + doc for doc in examples["src"]]
        model_inputs = tokenizer(
            inputs,
            max_length=128,  # Reduced max length
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = tokenizer(
            examples["tgt"],
            max_length=128,  # Reduced max length
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    rouge = evaluate.load('rouge')
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }
    
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=2000,            # Evaluate less frequently
        save_strategy="steps",
        save_steps=2000,            # Save less frequently
        learning_rate=3e-5,
        per_device_train_batch_size=1,  # Minimum batch size
        per_device_eval_batch_size=1,
        num_train_epochs=1,         # Reduced epochs
        weight_decay=0.01,
        save_total_limit=2,         # Keep fewer checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        # Added CPU optimization settings
        dataloader_num_workers=4,   # Parallel data loading
        dataloader_pin_memory=False,  # Disable pin memory for CPU
        fp16=False,                 # Disable mixed precision
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving fine-tuned model...")
    model.save_pretrained("./t5-rxcx-model")
    tokenizer.save_pretrained("./t5-rxcx-model")
    print("Training completed and model saved to ./t5-rxcx-model")

if __name__ == "__main__":
    main()