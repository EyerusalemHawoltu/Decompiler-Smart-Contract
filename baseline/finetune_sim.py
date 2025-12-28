import os
import torch
import json
import time
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from dataset import SIMDataset
from datasets import concatenate_datasets, load_from_disk
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_load_folder = '{path_to_model_after_training}'
model_save_folder = '{path_to_save_model}'


ds_config = json.load(open('ds_config.json', 'r'))
dschf = HfDeepSpeedConfig(ds_config)
torch.manual_seed(7)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Vocabulary:', len(tokenizer.get_vocab()))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# idx >= ID are special tokens for instruction
ID = tokenizer.encode('<label-1>')[1]

model = AutoModelForCausalLM.from_pretrained(model_load_folder, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Loaded, Start Loading Data, Parameters: {}'.format(sum(p.numel() for p in model.parameters())))

# Load train data
train_data_dirs = [
    '../data-tokenized/binarycorp/train',
]
train_data = concatenate_datasets([
    load_from_disk(data_dir, keep_in_memory=True) for data_dir in train_data_dirs
]).shuffle(seed=7, keep_in_memory=True)
train_data.set_format(type='torch', columns=['asm-O0-input_ids', 'asm-O3-input_ids'])
train_dataset = SIMDataset(tokenizer, train_data)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=500, num_training_steps=int(1.1 * len(train_dataset) / 64)
)
engine, _, train_dataloader, _ = deepspeed.initialize(model=model, training_data=train_dataset, config_params=ds_config, optimizer=optimizer, lr_scheduler=scheduler)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Data Loaded, Train: {}'.format(len(train_dataset)))
    print('Start Training, Gradient Checkpoint:', engine.module.model.gradient_checkpointing)

start_time = time.time()
train_loss = []
engine.module.train()
for epoch in range(1):
    for i, data in enumerate(train_dataloader):
        O0_input_ids = data['O0_input_ids'].to(engine.device)
        O3_input_ids = data['O3_input_ids'].to(engine.device)
        O0_attention_mask = data['O0_attention_mask'].to(engine.device)
        O3_attention_mask = data['O3_attention_mask'].to(engine.device)
        
        O0_h = engine.module(input_ids=O0_input_ids, attention_mask=O0_attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        O3_h = engine.module(input_ids=O3_input_ids, attention_mask=O3_attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]   # [B, L, H]

        O0_e, O3_e = [], []
        bsz = O0_h.size(0)
        for idx in range(bsz):
            O0_e.append(O0_h[idx][O0_input_ids[idx] >= ID].mean(dim=0).unsqueeze(0))
            O3_e.append(O3_h[idx][O3_input_ids[idx] >= ID].mean(dim=0).unsqueeze(0))
        O0_e = torch.cat(O0_e, dim=0)
        O3_e = torch.cat(O3_e, dim=0)

        # a softened ground-truth for similarity matrix
        gt = torch.eye(bsz).type_as(O0_e) * 3 + torch.ones((bsz, bsz)).type_as(O0_e)
        gt = torch.nn.functional.normalize(gt, dim=-1, p=1).view(-1, bsz)
        sim = torch.nn.functional.cosine_similarity(O0_e.unsqueeze(1), O3_e.unsqueeze(0), dim=-1)
        loss = nn.KLDivLoss(reduction="sum")(nn.LogSoftmax(dim=-1)(sim).view(-1, bsz), gt)

        engine.backward(loss)
        engine.step()
        train_loss.append(loss.mean().item())
        
        if i % 100 == 0:
            torch.cuda.empty_cache()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, time: {}s'.format(
                    epoch + 1, i, len(train_dataloader), round(sum(train_loss) / len(train_loss), 4), 
                    round(engine.optimizer.param_groups[0]['lr'], 8), int(time.time() - start_time)
                ))
            start_time = time.time()
            train_loss = []
        
    engine.save_checkpoint(model_save_folder)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('chackpoint saved')
