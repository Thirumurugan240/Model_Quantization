"""
run_quantized.py
----------------
Load the small (quantised) model we saved earlier and chat with it.

Run it:  python run_quantized.py
"""

from transformers import AutoTokenizer
from optimum.quanto import QuantizedModelForCausalLM

# The folder where quantize_qwen.py saved the small model.
SAVE_FOLDER = "./qwen-int8"

# --- Step 1: load the small model and tokenizer from the local folder ---
print("Loading the small model...")
tokenizer = AutoTokenizer.from_pretrained(SAVE_FOLDER)
model = QuantizedModelForCausalLM.from_pretrained(SAVE_FOLDER)

# --- Step 2: write your question (the INPUT) ---
question = "What is model quantisation? Answer in 3 short sentences."

# Qwen is a chat model, so we put our question in chat format.
messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

# --- Step 3: let the model generate an answer (the OUTPUT) ---
print("Thinking...")
output = model.generate(**inputs, max_new_tokens=200)

# Keep only the new words the model added (remove our question).
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# --- Step 4: show the input and the output ---
print("\nINPUT :", question)
print("OUTPUT:", answer)
