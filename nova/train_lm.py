import torch
from dataset import LMDataset
from modeling_nova import NovaForCausalLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
kwargs = DistributedDataParallelKwargs(static_graph=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
torch.manual_seed(7)

model_load_folder = 'deepseek-ai/deepseek-coder-1.3b-base'
model_save_folder = '{path_to_save_model}'


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Vocabulary:', len(tokenizer.get_vocab()))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = NovaForCausalLM.from_pretrained(model_load_folder, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Loaded, Start Loading Data, Parameters: {}'.format(sum(p.numel() for p in model.parameters())))

# Load train data
train_data_dirs = [
    '../data-tokenized/nova-anghabench-lm/train', '../data-tokenized/nova-the-stack-lm/train'
]
train_data = concatenate_datasets([
    load_from_disk(data_dir, keep_in_memory=True) for data_dir in train_data_dirs
]).shuffle(seed=7, keep_in_memory=True)
train_data.set_format(type='torch', columns=['input_ids', 'labels', 'nova_attention_mask'])
train_dataset = LMDataset(tokenizer, train_data)

# Load valid data
valid_data_dirs = [
    '../data-tokenized/nova-anghabench-lm/valid', '../data-tokenized/nova-the-stack-lm/valid'
]
valid_data = concatenate_datasets([
    load_from_disk(data_dir, keep_in_memory=True) for data_dir in valid_data_dirs
]).shuffle(seed=7, keep_in_memory=True)
valid_data.set_format(type='torch', columns=['input_ids', 'labels', 'nova_attention_mask'])
valid_dataset = LMDataset(tokenizer, valid_data)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Data Loaded, Train: {}, Valid: {}'.format(len(train_dataset), len(valid_dataset)))
    print('Start Training, Gradient Checkpoint:', model.model.gradient_checkpointing)

trainer_args = TrainingArguments(
    output_dir=model_save_folder,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    deepspeed='ds_config.json',
    learning_rate=3e-5,
    lr_scheduler_type='cosine',
    warmup_steps=1000,
    optim='adamw_torch',
    weight_decay=0.01,
    num_train_epochs=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    save_strategy='epoch',
    save_only_model=True,
    logging_dir='logs/',
    logging_strategy='steps',
    logging_steps=1000,
    evaluation_strategy='steps',
    eval_steps=4000,
    prediction_loss_only=True,
    bf16=True
)
trainer = Trainer(
    model=model, args=trainer_args, train_dataset=train_dataset, eval_dataset=valid_dataset
)

trainer.train()
