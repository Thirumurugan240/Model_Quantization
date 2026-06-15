# Model Quantisation — (Qwen2.5-0.5B, runs on CPU)

A hands-on package for teaching **model quantisation** and running a quantised
LLM **locally** for fast output. Built and tested on a **CPU-only Windows**
machine (no NVIDIA GPU required).

## Contents

| File | What it is |
|------|------------|
| `01_model_quantisation_guide.ipynb` | Step-by-step teaching notebook (theory + runnable demos) |
| `quantize_qwen.py` | Downloads **Qwen2.5-0.5B-Instruct**, quantises it, saves locally |
| `run_quantized.py` | Loads the saved quantised model and generates text (with tok/s) |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quantise the model and store it locally (creates ./qwen-int8)
python quantize_qwen.py

# 3. Run the quantised model (prints the input question and the model's answer)
python run_quantized.py
#    To ask something else, just edit the `question = "..."` line in run_quantized.py

# 4. Open the teaching notebook
jupyter notebook 01_model_quantisation_guide.ipynb
```

## Why these tools?

This machine is **CPU-only**, so GPU-only quantisers (bitsandbytes, GPTQ, AWQ)
do **not** apply here. We use:

- **optimum-quanto** — pure-PyTorch INT8/INT4 quantisation that runs on CPU.
  Easiest path from Python; used by the scripts and notebook.
- **GGUF / llama.cpp** — the *fastest* CPU route; covered in the notebook's
  bonus section (optional install).

## What "quantisation" buys you

Storing weights in fewer bits → smaller model, less RAM, faster token
generation, runs offline & privately. INT8 ≈ 4× smaller than FP32 with
negligible quality loss; INT4 ≈ 8× smaller with a small, usually unnoticeable
drop.

