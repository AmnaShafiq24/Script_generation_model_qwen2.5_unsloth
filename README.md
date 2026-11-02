# Tone-Controlled Script Writer (LoRA Adapter)

A LoRA adapter fine-tuned on **Qwen2.5-7B-Instruct** that generates short video scripts in a **user-selected tone**. Trained with [Unsloth](https://github.com/unslothai/unsloth) QLoRA on a Colab T4 GPU.

## Overview

- **Base model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Method:** QLoRA fine-tuning (4-bit base, LoRA adapters on attention + MLP projections)
- **Framework:** Unsloth + TRL `SFTTrainer`
- **Hardware:** Single T4 GPU (Google Colab, free tier)
- **Task:** Short-form video script generation, conditioned on a tone label passed in the prompt

The adapter does not replace the base model — it's a small set of trained weights applied on top of it. You need both the base model and this adapter to run inference.

## Supported tones

| Tone | Style |
|---|---|
| `humorous` | Playful, witty, light jokes and funny comparisons |
| `formal` | Professional and precise, no slang, no jokes |
| `inspirational` | Uplifting, motivating, speaks directly to the viewer |
| `educational` | Clear and patient, builds ideas step by step |
| `dramatic` | Suspenseful, intense, short punchy sentences |
| `casual` | Relaxed and conversational, like talking to a friend |
| `conversational` | Warm, like a knowledgeable friend explaining something interesting |
| `documentary` | Measured, precise, slightly formal |
| `energetic` | High-energy, punchy, short sentences with momentum |

## Training data

The training set was **synthetically generated**, not sourced from an external dataset. The base model itself wrote ~300–500 script examples across 60 topics × 9 tones, each generated with that tone's style guide in the system prompt. Examples outside a 60–300 word range were filtered out. Scripts were briefly spot-checked, but quality varies since the data wasn't hand-written or reviewed line by line.

**Known limitation:** `casual` and `conversational` overlap stylistically — outputs may not always be clearly distinguishable between the two.

## Prompt format

Training and inference both use this structure:

**System prompt:**
```
You are a professional video script writer. Always write the script in the tone the user requests.
```

**User message:**
```
Tone: <tone>
Topic: <topic>
Write a short video script.
```

The model was trained to read the tone from this exact `Tone: ...` line, so inference requests should follow the same format for reliable results.

## Usage

### Load the adapter (with Unsloth)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_adapter",  # path to this repo's lora_adapter/ folder after cloning
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### Generate a script

```python
def generate_script(topic, tone, max_new_tokens=400):
    messages = [
        {"role": "system", "content": "You are a professional video script writer. Always write the script in the tone the user requests."},
        {"role": "user", "content": f"Tone: {tone}\nTopic: {topic}\nWrite a short video script."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=0.7, top_p=0.9, do_sample=True,
    )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

generate_script("why houseplants keep dying", "humorous")
```

### Without Unsloth (standard PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

## Repository structure

```
.
├── README.md
├── .gitignore
├── .gitattributes          # Git LFS tracking rules
├── notebook/
│   └── tone_finetune_unsloth.ipynb   # training notebook
└── lora_adapter/
    ├── adapter_config.json       # LoRA configuration (rank, target modules, etc.)
    ├── adapter_model.safetensors # Trained LoRA weights (tracked via Git LFS)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

This repo tracks `*.safetensors` and `*.bin` via **Git LFS**. Run `git lfs pull` after cloning to fetch the actual weight files — without it, you'll only get small pointer files.

```bash
git clone <this-repo-url>
cd <repo>
git lfs pull
```

## Training configuration

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Batch size | 2 (× 4 gradient accumulation = 8 effective) |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |
| Precision | fp16 (T4 does not support bf16) |
| Max sequence length | 2048 |

## Known limitations

- Trained on **synthetic** data generated by the base model itself — not real-world scripts, so style and factual grounding are only as good as the base model's own writing.
- `casual` and `conversational` tones may not be reliably distinguishable.
- Per-tone example counts are not perfectly balanced; tones with more filtered-out generations have fewer training examples.
- Not evaluated against a held-out test set — quality assessment so far is based on spot checks.

## Used by

This adapter powers the tone-controlled script generation feature in [Script Studio](#), an AI-assisted video content creation tool. It is published as a standalone model repo so it can be versioned, loaded, and reused independently.