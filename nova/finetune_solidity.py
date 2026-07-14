"""
Fine-tune Nova for EVM CFG → Solidity decompilation.

This is the direct analogue of Nova's BCR (Binary Code Recovery) fine-tuning step,
adapted for our task:
    Nova BCR:  X86-64 assembly  →  C source code
    Ours:      EVM CFG text     →  Solidity source code

Starting point: deepseek-ai/deepseek-coder-1.3b-base loaded into NovaForCausalLM
(identical to Nova's own pre-training start point).  For faster convergence you can
swap in 'lt-asset/nova-1.3b' which already has hierarchical attention pre-trained on
assembly.

Run from the nova/ directory on HPC:
    torchrun --nproc-per-node=4 finetune_solidity.py
    # or with DeepSpeed:
    deepspeed --num_gpus=4 finetune_solidity.py

Outputs (set MODEL_SAVE_FOLDER below):
    {MODEL_SAVE_FOLDER}/checkpoint-*/   <- HuggingFace model checkpoints
    {MODEL_SAVE_FOLDER}/logs/           <- TensorBoard training logs
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset import SolidityDataset
from modeling_nova import NovaTokenizer, NovaForCausalLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.manual_seed(7)

# ── Paths ──────────────────────────────────────────────────────────────────
# Start from DeepSeek-Coder base (same as Nova's own pre-training).
# Alternative: 'lt-asset/nova-1.3b' for faster convergence (already has
# hierarchical-attention weights trained on assembly).
MODEL_LOAD_FOLDER = 'deepseek-ai/deepseek-coder-1.3b-base'

# Where to save checkpoints. On HPC, set this to a path on your scratch storage.
MODEL_SAVE_FOLDER = '../checkpoints/nova-solidity-1.3b'

DATA_DIR = '../data-tokenized/nova-solidity-bcr'
# ───────────────────────────────────────────────────────────────────────────

kwargs = DistributedDataParallelKwargs(static_graph=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

# Tokenizer — must match the one used in prepare_solidity_dataset.py
tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True
)
tokenizer.add_tokens(
    ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
    special_tokens=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
nova_tok = NovaTokenizer(tokenizer)

rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
if rank0:
    print(f'Vocabulary size: {len(tokenizer.get_vocab())}')

# Model
model = NovaForCausalLM.from_pretrained(MODEL_LOAD_FOLDER, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
if rank0:
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded from {MODEL_LOAD_FOLDER} | Parameters: {n_params:,}')

# Dataset — raw text columns (input / output / char_types).
# SolidityDataset tokenises on-the-fly and builds nova_attention_mask per sample,
# avoiding the ~1.3 TB pre-computation that caused OOM during dataset prep.
train_data = load_from_disk(f'{DATA_DIR}/train', keep_in_memory=True)
valid_data = load_from_disk(f'{DATA_DIR}/valid', keep_in_memory=True)
train_dataset = SolidityDataset(nova_tok, tokenizer, train_data, max_len=2048)
valid_dataset = SolidityDataset(nova_tok, tokenizer, valid_data, max_len=2048)

if rank0:
    print(f'Train: {len(train_dataset)} | Valid: {len(valid_dataset)}')

# Training arguments
# Effective batch = nproc * per_device_batch * gradient_accumulation
# With 4 GPUs, per_device=8, accum=4 → effective batch = 128
trainer_args = TrainingArguments(
    output_dir=MODEL_SAVE_FOLDER,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    deepspeed='ds_config.json',
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_steps=500,
    optim='adamw_torch',
    weight_decay=0.01,
    num_train_epochs=3,            # our dataset is smaller than Nova's, use 3 epochs
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    save_strategy='epoch',
    save_only_model=True,
    logging_dir=os.path.join(MODEL_SAVE_FOLDER, 'logs'),
    logging_strategy='steps',
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=1000,
    prediction_loss_only=True,
    bf16=True,
    report_to='tensorboard',
    dataloader_num_workers=4,       # overlap on-the-fly tokenisation with GPU compute
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

if rank0:
    print('Starting fine-tuning ...')
trainer.train()
if rank0:
    print(f'Done. Checkpoints saved to {MODEL_SAVE_FOLDER}')
