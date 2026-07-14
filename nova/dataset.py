import numpy as np
import torch


class SolidityDataset(torch.utils.data.Dataset):
    def __init__(self, nova_tokenizer, tokenizer, data, max_len=2048):
        super().__init__()
        self.nova_tokenizer = nova_tokenizer
        self.tokenizer      = tokenizer
        self.data           = data
        self.max_len        = max_len
        self.pad_id         = tokenizer.eos_token_id
        self.eos            = tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inp  = item['input']
        out  = item['output']
        ct   = item['char_types']
        eos_zeros = '0' * len(self.eos)
        try:
            enc = self.nova_tokenizer.encode(inp, out + self.eos, ct + eos_zeros)
        except Exception:
            L = self.max_len
            return {
                'input_ids':           torch.full((L,), self.pad_id, dtype=torch.long),
                'labels':              torch.full((L,), -100,        dtype=torch.long),
                'attention_mask':      torch.zeros(L,   dtype=torch.bool),
                'nova_attention_mask': torch.zeros(L, L, dtype=torch.bool),
            }
        ids      = list(enc['input_ids'])
        lbs      = list(enc['labels'])
        raw_mask = enc['nova_attention_mask']
        actual   = len(ids)
        if actual > self.max_len:
            ids      = ids[:self.max_len]
            lbs      = lbs[:self.max_len]
            raw_mask = raw_mask[:self.max_len, :self.max_len]
            actual   = self.max_len
        pad  = self.max_len - actual
        ids  = ids  + [self.pad_id] * pad
        lbs  = lbs  + [-100]        * pad
        attn = [1]  * actual        + [0] * pad
        nova = np.zeros((self.max_len, self.max_len), dtype=bool)
        nova[:actual, :actual] = raw_mask
        return {
            'input_ids':           torch.LongTensor(ids),
            'labels':              torch.LongTensor(lbs),
            'attention_mask':      torch.BoolTensor(attn),
            'nova_attention_mask': torch.BoolTensor(nova),
        }


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        super(LMDataset, self).__init__()
        self.data      = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        input_ids = self.data[ind]['input_ids']
        labels    = self.data[ind]['labels']
        L = self.data[ind]['nova_attention_mask'].size(0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        nova_attention_mask = torch.zeros(input_ids.size(0), input_ids.size(0)).type(torch.bool)
        nova_attention_mask[:L, :L] = self.data[ind]['nova_attention_mask']
        return {
            'input_ids':           input_ids,
            'attention_mask':      attention_mask,
            'nova_attention_mask': nova_attention_mask,
            'labels':              labels,
        }


class BCRDataset(LMDataset):
    def __init__(self, tokenizer, data):
        super().__init__(tokenizer, data)
