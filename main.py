from fastapi import FastAPI, Request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import uvicorn

app = FastAPI(title="Text Formalization API")

# Load the model and tokenizer
model_path = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)

@app.post("/formalize")
async def formalize_text(request: Request):
    input_data = await request.json()
    input_text = input_data.get("text", "")
    
    # Modify the prompt to be specific to grammar correction and formalization
    input_text = f"fix grammar and formalize: {input_text}"
    
    # Tokenize and generate
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        max_length=50,  # Shorter output length
        num_beams=2,  # Reduce beams for more concise results
        temperature=0.6,  # Lower temperature for less creative generation
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False  # More deterministic output
    )
    
    formal_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the formalized text
    return {
        "original_text": input_data.get("text", ""),
        "formal_text": formal_text.strip()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
