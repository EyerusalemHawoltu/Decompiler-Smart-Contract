"""
Full fine-tune: 80% train split, 3 epochs, 1 GPU, no DeepSpeed.
Uses the properly held-out train/valid/test split from prepare_solidity_dataset.py.
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer
from dataset import SolidityDataset
from modeling_nova import NovaTokenizer, NovaForCausalLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.manual_seed(7)

MODEL_LOAD_FOLDER = 'deepseek-ai/deepseek-coder-1.3b-base'
# Paths are env-overridable so the SAME script trains both the CFG model and the
# Gigahorse model (only DATA_DIR / MODEL_SAVE differ).
MODEL_SAVE_FOLDER = os.environ.get('MODEL_SAVE', '../checkpoints/nova-solidity-1.3b-clean')
DATA_DIR          = os.environ.get('DATA_DIR',   '../data-tokenized/nova-solidity-bcr')

tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_FOLDER, trust_remote_code=True)
tokenizer.add_tokens(
    ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
    special_tokens=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
nova_tok = NovaTokenizer(tokenizer)
print(f'Vocabulary size: {len(tokenizer.get_vocab())}')

model = NovaForCausalLM.from_pretrained(MODEL_LOAD_FOLDER, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Load 80/10/10 split — train on train only, monitor on valid
train_data = load_from_disk(f'{DATA_DIR}/train', keep_in_memory=True)
valid_data = load_from_disk(f'{DATA_DIR}/valid', keep_in_memory=True)

train_dataset = SolidityDataset(nova_tok, tokenizer, train_data, max_len=2048)
valid_dataset = SolidityDataset(nova_tok, tokenizer, valid_data, max_len=2048)
print(f'Train: {len(train_dataset)} | Valid: {len(valid_dataset)}')

trainer_args = TrainingArguments(
    output_dir=MODEL_SAVE_FOLDER,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_steps=500,
    optim='adamw_torch',
    weight_decay=0.01,
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    save_strategy='epoch',
    save_only_model=False,   # save optimizer/scheduler/rng -> resumable on preemption
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=1000,
    prediction_loss_only=True,
    bf16=True,
    report_to='none',
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

print('Starting full fine-tuning ...')
# Resume only if a checkpoint already exists (survives preemption); otherwise
# start fresh from the base model — required for the clean retrain.
from transformers.trainer_utils import get_last_checkpoint
_last = get_last_checkpoint(MODEL_SAVE_FOLDER) if os.path.isdir(MODEL_SAVE_FOLDER) else None
print(f'Resuming from: {_last}' if _last else 'Fresh start (no prior checkpoint)')
trainer.train(resume_from_checkpoint=_last)
print(f'Done. Checkpoints saved to {MODEL_SAVE_FOLDER}')
