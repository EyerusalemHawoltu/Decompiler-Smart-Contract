"""
EVM CFG → Solidity Decompiler — Gradio web app.

Run from the project root:
    pip install gradio transformers torch
    python app.py
"""

import os
import sys
import torch
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
NOVA_DIR = os.path.join(ROOT, 'nova')
CKPT     = os.path.join(ROOT, 'checkpoints', 'checkpoint-24060')
sys.path.insert(0, NOVA_DIR)

from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer, NovaForCausalLM
from prepare_solidity_dataset import normalize_cfg

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f'Device: {DEVICE}')

# ── Load model once at startup ─────────────────────────────────────────────────
print('Loading tokenizer (downloads from HuggingFace on first run)...')
tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True
)
tokenizer.add_tokens(
    ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
    special_tokens=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
nova_tok = NovaTokenizer(tokenizer)

print(f'Loading model from {CKPT} ...')
dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32
model = NovaForCausalLM.from_pretrained(
    CKPT, torch_dtype=dtype, trust_remote_code=True
).to(DEVICE).eval()
print('Model ready!\n')


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def decompile(cfg_text: str, version: str, max_new_tokens: int) -> str:
    if not cfg_text.strip():
        return "⚠️  Please paste an EVM CFG."

    version = version.strip() or '0.8.x'
    cfg_norm = normalize_cfg(cfg_text.strip())

    prompt_before = f'# This is the EVM CFG for a Solidity {version} function:\n'
    prompt_after  = '\nWhat is the Solidity source code?\n'
    input_text    = prompt_before + cfg_norm + prompt_after
    char_types    = (
        '0' * len(prompt_before) +
        '1' * len(cfg_norm) +
        '0' * len(prompt_after)
    )

    enc         = nova_tok.encode(input_text, '', char_types)
    input_ids   = torch.LongTensor([enc['input_ids'].tolist()]).to(DEVICE)
    nova_mask   = torch.LongTensor(enc['nova_attention_mask']).unsqueeze(0).to(DEVICE)
    no_mask_idx = torch.LongTensor([enc['no_mask_idx']]).to(DEVICE)

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        nova_attention_mask=nova_mask,
        no_mask_idx=no_mask_idx,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        outputs[0][input_ids.size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded.strip()


# ── Example CFG ───────────────────────────────────────────────────────────────
EXAMPLE = """\
Block 0x0:
  - PUSH1 0x80
  - PUSH1 0x40
  - MSTORE
  - CALLVALUE
  - DUP1
  - ISZERO
  - PUSH2 0x10
  - JUMPI
Block 0x10:
  - JUMPDEST
  - POP
  - PUSH1 0x4
  - CALLDATASIZE
  - LT
  - PUSH2 0x2a
  - JUMPI"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="EVM Decompiler", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🔍 EVM CFG → Solidity Decompiler
Paste an EVM Control Flow Graph and get the reconstructed Solidity source code.
*Fine-tuned on 256K functions · BLEU-4: 0.8665 · Exact match: 66.2%*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            cfg_input = gr.Textbox(
                lines=22,
                placeholder="Paste EVM CFG here...",
                label="EVM Control Flow Graph",
                value=EXAMPLE,
            )
            with gr.Row():
                version_box = gr.Textbox(
                    value="0.8.9",
                    label="Solidity version",
                    scale=1,
                )
                max_tok = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max output tokens",
                    scale=2,
                )
            run_btn = gr.Button("🔓 Decompile", variant="primary", size="lg")

        with gr.Column(scale=1):
            sol_output = gr.Code(
                language="javascript",   # Gradio has no Solidity lang; JS highlighting is close
                label="Decompiled Solidity",
                lines=22,
            )

    run_btn.click(
        fn=decompile,
        inputs=[cfg_input, version_box, max_tok],
        outputs=sol_output,
    )

    gr.Markdown("""
---
> ⚠️ **Note:** Running on CPU is slow (~1–3 min per function). GPU/MPS is recommended.
> Model: `Nova-Solidity-1.3B` | Checkpoint: `checkpoint-24060` | Training: 3 epochs, 256K functions
    """)

if __name__ == '__main__':
    demo.launch(share=False)
