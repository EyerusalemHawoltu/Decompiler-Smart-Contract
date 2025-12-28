import os
import torch
import json
import time
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from dataset import CONTRADataset
from datasets import concatenate_datasets, load_from_disk
import torch.nn as nn
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = 'false'
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_load_folder = '{path_to_model_after_train_lm.py}'
model_save_folder = '{path_to_save_model}'


ds_config = json.load(open('ds_config.json', 'r'))
dschf = HfDeepSpeedConfig(ds_config)
torch.manual_seed(7)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
ID = tokenizer.encode('<label-1>')[1]
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Vocabulary:', len(tokenizer.get_vocab()), 'ID:', ID)

model = AutoModelForCausalLM.from_pretrained(model_load_folder, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.gradient_checkpointing_enable()
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Loaded, Start Loading Data, Parameters: {}'.format(sum(p.numel() for p in model.parameters())))

# Load train data
train_data_dirs = [
    '../data-tokenized/anghabench-contra/train', '../data-tokenized/the-stack-contra/train'
]
train_data = concatenate_datasets([
    load_from_disk(data_dir, keep_in_memory=True) for data_dir in train_data_dirs
]).shuffle(seed=7, keep_in_memory=True)
train_data.set_format(type='torch', columns=[
    'src-input_ids', 'src-labels', 'asm-O0-input_ids', 'asm-O0-labels', 'asm-O1-input_ids', 'asm-O1-labels', 
    'asm-O2-input_ids', 'asm-O2-labels', 'asm-O3-input_ids', 'asm-O3-labels'])
train_dataset = CONTRADataset(tokenizer, train_data)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=200, num_training_steps=int(1.1 * len(train_dataset) / 64)
)
engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
    model=model, optimizer=optimizer, training_data=train_dataset, lr_scheduler=scheduler, config_params=ds_config
)
engine.train()

start_time = time.time()
lm_loss_lst, cross_loss_lst, inner_loss_lst = [], [], []
for idx, data in enumerate(train_dataloader):
    input_ids = data['input_ids'].to(engine.device)
    labels = data['labels'].to(engine.device)
    attention_mask = data['attention_mask'].to(engine.device)

    bsz, gsz, L = input_ids.size()
    
    input_ids = input_ids.view(bsz * gsz, L)
    labels = labels.view(bsz * gsz, L)
    attention_mask = attention_mask.view(bsz * gsz, L)

    lm_loss = 0
    hidden_states = []
    for i in range(bsz):
        output = engine(input_ids=input_ids[i * gsz: (i + 1) * gsz, :], attention_mask=attention_mask[i * gsz: (i + 1) * gsz, :], labels=labels[i * gsz: (i + 1) * gsz, :], return_dict=True, output_hidden_states=True)
        lm_loss = lm_loss + output.loss / bsz
        hidden_states.append(output.hidden_states[-1])
    hidden_states = torch.cat(hidden_states, dim=0)

    # hidden_states = output.hidden_states[-1]        # [bsz * gsz, L, H]
    embeds = []
    for i in range(hidden_states.size(0)):
        if i % gsz == 0:
            embeds.append(hidden_states[i][input_ids[i] != tokenizer.pad_token_id].mean(dim=0).unsqueeze(0))
        else:
            embeds.append(hidden_states[i][input_ids[i] >= ID].mean(dim=0).unsqueeze(0))
    # [bsz * gsz, H]
    embeds = torch.cat(embeds, dim=0).view(bsz, gsz, -1)              # [bsz, 5, H]

    cross_loss = 0

    # A softened ground-truth distribution for similarity matrix
    gt = torch.eye(bsz).type_as(embeds) * 2 + torch.ones((bsz, bsz)).type_as(embeds)
    gt = torch.nn.functional.normalize(gt, dim=-1, p=1).view(-1, bsz)
    for i in range(gsz - 1):
        l1 = embeds[:, i: i + 1, :].permute(1, 0, 2)                # [1, bsz, H]
        l2 = embeds[:, i + 1: i + 2, :].permute(1, 0, 2)
        sim = torch.nn.functional.cosine_similarity(l1.unsqueeze(2), l2.unsqueeze(1), dim=-1)
        
        logits = nn.LogSoftmax(dim=-1)(sim)
        cross_loss = cross_loss + nn.KLDivLoss(reduction="sum")(logits.view(-1, bsz), gt)
        
    cross_loss = cross_loss / (gsz - 1)

    d0s = torch.norm(embeds[:, 1, :] - embeds[:, 0, :])
    d1s = torch.norm(embeds[:, 2, :] - embeds[:, 0, :])
    d2s = torch.norm(embeds[:, 3, :] - embeds[:, 0, :])
    d3s = torch.norm(embeds[:, 4, :] - embeds[:, 0, :])
    d10 = torch.norm(embeds[:, 2, :] - embeds[:, 1, :])
    d20 = torch.norm(embeds[:, 3, :] - embeds[:, 1, :])
    d30 = torch.norm(embeds[:, 4, :] - embeds[:, 1, :])
    d21 = torch.norm(embeds[:, 3, :] - embeds[:, 2, :])
    d31 = torch.norm(embeds[:, 4, :] - embeds[:, 2, :])
    zero = torch.tensor(0).type_as(d10)
    inner_loss = (
        torch.max(d10 - d20, zero).sum() + torch.max(d20 - d30, zero).sum() + torch.max(d21 - d31, zero).sum() + 
        torch.max(d0s - d1s, zero).sum() + torch.max(d1s - d2s, zero).sum() + torch.max(d2s - d3s, zero).sum()
    ) / bsz

    engine.backward(lm_loss + 0.1 * cross_loss + inner_loss)
    engine.step()

    lm_loss_lst.append(lm_loss.item())
    cross_loss_lst.append(cross_loss.item())
    inner_loss_lst.append(inner_loss.item())

    if idx % 100 == 0:
        torch.cuda.empty_cache()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('step: {}/{}, loss: {}, lm_loss: {}, cross_loss: {}, inner_loss: {}, lr: {}, time: {}s'.format(
                idx, len(train_dataloader), round(np.mean(lm_loss_lst + cross_loss_lst + inner_loss_lst), 4), 
                round(np.mean(lm_loss_lst), 4), round(np.mean(cross_loss_lst), 4), round(np.mean(inner_loss_lst), 4), 
                round(engine.optimizer.param_groups[0]['lr'], 8), int(time.time() - start_time)
            ))
        start_time = time.time()
        lm_loss_lst, cross_loss_lst, inner_loss_lst = [], [], []
engine.save_checkpoint(model_save_folder)
