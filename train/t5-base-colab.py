# CELL 1: Install packages and setup authentication
#!pip install datasets transformers evaluate torch huggingface_hub

from google.colab import userdata
from huggingface_hub import login, HfApi
import torch
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
import os
import gc
from tqdm.auto import tqdm
from datetime import datetime

# Authentication and model setup
HF_TOKEN = userdata.get('HUGGING_FACE_TOKEN')
login(HF_TOKEN)

# Version control setup
VERSION = "v1.0.0"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_TAGS = [VERSION, f"trained_{TIMESTAMP}", "A100", "t5-base", "formal-language"]

# Model identification
api = HfApi()
username = "renix-codex"
model_name = "formal-lang-rxcx-model"  # New model name
model_id = f"{username}/{model_name}"

# Create repository if it doesn't exist
try:
    api.create_repo(model_id, exist_ok=True)
    print(f"Repository {model_id} is ready")
except Exception as e:
    print(f"Repository setup: {str(e)}")

# Verify setup
try:
    user_info = api.whoami()
    print(f"Successfully logged in as: {user_info['name']}")
    print(f"Model will be saved as: {model_id}")
    print(f"Version: {VERSION}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Info:")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print("Setup verification failed")
    raise e

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def process_dataset_chunk(dataset, start_idx, end_idx, tokenizer, is_validation=False):
    chunk = dataset.select(range(start_idx, end_idx))
    inputs = ["make formal: " + doc for doc in chunk["src"]]  # Updated prompt
    
    batch_size = 16  # Adjusted for t5-base
    all_inputs = []

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        model_inputs = tokenizer(
            batch_inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        batch_targets = chunk["tgt"][i:i + batch_size]
        target_encoding = tokenizer(
            batch_targets,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        model_inputs["labels"] = target_encoding["input_ids"]
        all_inputs.append(model_inputs)
        clear_memory()

    return all_inputs

def train_on_batch(model, batch, optimizer, grad_accum_steps):
    loss = model(**batch).loss
    loss = loss / grad_accum_steps
    loss.backward()
    return loss.item()

def main():
    clear_memory()
    print("\nInitial GPU Memory:")
   # !nvidia-smi

    print("\nLoading dataset...")
    dataset = load_dataset("grammarly/coedit")
    total_examples = len(dataset['train'])
    print(f"Total training examples: {total_examples}")

    # Initialize T5-base model
    base_model = "t5-base"
    print(f"\nLoading model and tokenizer from {base_model}")

    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(
        base_model,
        low_cpu_mem_usage=True
    )

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")

    # T5-base optimized settings for A100
    chunk_size = 400
    micro_batch_size = 2
    grad_accum_steps = 16
    learning_rate = 3e-5

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    num_chunks = (total_examples + chunk_size - 1) // chunk_size
    
    print(f"\nTraining configuration:")
    print(f"- Model: T5-base ({base_model})")
    print(f"- Total examples: {total_examples}")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Number of chunks: {num_chunks}")
    print(f"- Micro batch size: {micro_batch_size}")
    print(f"- Gradient accumulation steps: {grad_accum_steps}")
    print(f"- Effective batch size: {micro_batch_size * grad_accum_steps}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Sequence length: 128")

    start_time = datetime.now()
    total_loss = 0
    num_batches = 0

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_examples)

        chunk_start_time = datetime.now()
        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}")
        print(f"Examples {start_idx} to {end_idx}")

        try:
            batches = process_dataset_chunk(
                dataset['train'],
                start_idx,
                end_idx,
                tokenizer
            )

            model.train()
            accumulated_loss = 0
            steps = 0

            for batch_idx, batch in enumerate(batches):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}

                loss = train_on_batch(model, batch, optimizer, grad_accum_steps)
                accumulated_loss += loss
                total_loss += loss
                steps += 1
                num_batches += 1

                if steps % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Batch {batch_idx + 1}, Loss: {accumulated_loss / steps:.4f}")
                    accumulated_loss = 0
                    steps = 0

                clear_memory()

            # Save checkpoint every 20 chunks
            if (chunk_idx + 1) % 20 == 0:
                checkpoint_path = f"./checkpoints/chunk_{chunk_idx}"
                model.save_pretrained(checkpoint_path)
                print(f"Saved checkpoint after chunk {chunk_idx + 1}")

            chunk_time = datetime.now() - chunk_start_time
            elapsed_time = datetime.now() - start_time
            avg_loss = total_loss / num_batches
            
            print(f"\nChunk Statistics:")
            print(f"Time for chunk: {chunk_time}")
            print(f"Total time elapsed: {elapsed_time}")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Examples processed: {end_idx}/{total_examples}")

            clear_memory()
            print("\nGPU Memory after chunk:")
           # !nvidia-smi

        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            error_path = f"./checkpoints/error_chunk_{chunk_idx}"
            model.save_pretrained(error_path)
            print(f"Saved error checkpoint at chunk {chunk_idx}")
            continue

    final_time = datetime.now() - start_time
    print(f"\nTraining completed in {final_time}")
    print(f"Final average loss: {total_loss / num_batches:.4f}")

    # Save final model
    print("\nSaving final model...")
    final_path = f"./final_model_{VERSION}"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Create model card content
    model_card = f"""
    # Formal Language T5 Model
    
    This model is fine-tuned from T5-base for formal language correction.
    
    ## Model Details
    - Base Model: T5-base
    - Training Dataset: Grammarly/COEDIT
    - Version: {VERSION}
    - Training Time: {final_time}
    - Final Loss: {total_loss / num_batches:.4f}
    
    ## Usage
    ```python
    from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
    
    model = AutoModelForSeq2SeqGeneration.from_pretrained("{model_id}")
    tokenizer = AutoTokenizer.from_pretrained("{model_id}")
    
    text = "make formal: your informal text here"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    formal_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```
    
    ## Training Configuration
    - Batch Size: {micro_batch_size}
    - Gradient Accumulation: {grad_accum_steps}
    - Learning Rate: {learning_rate}
    - Sequence Length: 128
    - Training Examples: {total_examples}
    
    ## Performance
    Average Loss: {total_loss / num_batches:.4f}
    Training Time: {final_time}
    """

    # Push to hub with version and model card
    print("\nPushing to Hugging Face Hub...")
    model.push_to_hub(model_id, tags=MODEL_TAGS, commit_message=f"Training completed {VERSION}")
    tokenizer.push_to_hub(model_id, tags=MODEL_TAGS)
    
    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=model_id,
        commit_message=f"Update model card for {VERSION}"
    )

    print(f"Training completed and model pushed to {model_id}")
    print(f"Version: {VERSION}")
    print(f"Tags: {MODEL_TAGS}")

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)
    
    print("Starting T5-base training on A100...")
    print("\nInitial GPU status:")
   # !nvidia-smi
    
    main()