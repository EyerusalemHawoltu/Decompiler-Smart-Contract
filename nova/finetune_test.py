"""
Quick sanity-check fine-tune: 5000 samples, 1 GPU, 1 epoch, no DeepSpeed.
Run this first to confirm the full pipeline works before submitting the
24-hour 4-GPU job.
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
MODEL_SAVE_FOLDER = '../checkpoints/nova-solidity-test'
DATA_DIR          = '../data-tokenized/nova-solidity-bcr'
MAX_TRAIN_SAMPLES = 5000
MAX_VALID_SAMPLES = 500

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_FOLDER, trust_remote_code=True)
tokenizer.add_tokens(
    ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
    special_tokens=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
nova_tok = NovaTokenizer(tokenizer)
print(f'Vocabulary size: {len(tokenizer.get_vocab())}')

# Model
model = NovaForCausalLM.from_pretrained(MODEL_LOAD_FOLDER, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Dataset — slice to small subset
train_data = load_from_disk(f'{DATA_DIR}/train').select(range(MAX_TRAIN_SAMPLES))
valid_data = load_from_disk(f'{DATA_DIR}/valid').select(range(MAX_VALID_SAMPLES))
train_dataset = SolidityDataset(nova_tok, tokenizer, train_data, max_len=2048)
valid_dataset = SolidityDataset(nova_tok, tokenizer, valid_data, max_len=2048)
print(f'Train: {len(train_dataset)} | Valid: {len(valid_dataset)}')

# Training — single GPU, no DeepSpeed, 1 epoch
trainer_args = TrainingArguments(
    output_dir=MODEL_SAVE_FOLDER,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_steps=50,
    optim='adamw_torch',
    weight_decay=0.01,
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    save_strategy='epoch',
    save_only_model=True,
    logging_dir=os.path.join(MODEL_SAVE_FOLDER, 'logs'),
    logging_strategy='steps',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=100,
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

print('Starting test fine-tuning ...')
trainer.train()
print(f'Done. Checkpoint saved to {MODEL_SAVE_FOLDER}')
