import torch


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        super(LMDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        input_ids = self.data[ind]['input_ids']
        labels = self.data[ind]['labels']
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class BCRDataset(LMDataset):
    def __init__(self, tokenizer, data):
        super().__init__(tokenizer, data)


class CONTRADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        super(CONTRADataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        result = {'input_ids': [], 'labels': [], 'attention_mask': []}
        for k in ['src', 'asm-O0', 'asm-O1', 'asm-O2', 'asm-O3']:
            input_ids = self.data[ind][f'{k}-input_ids'].unsqueeze(0)
            labels = self.data[ind][f'{k}-labels'].unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
            
            result['input_ids'].append(input_ids)
            result['labels'].append(labels)
            result['attention_mask'].append(attention_mask)
        for k in result:
            result[k] = torch.cat(result[k], dim=0)
        return result


class SIMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        super(SIMDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        O0_input_ids = self.data[ind]['asm-O0-input_ids']
        O3_input_ids = self.data[ind]['asm-O3-input_ids']

        O0_attention_mask = torch.ones_like(O0_input_ids)
        O0_attention_mask = O0_attention_mask.masked_fill(O0_input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        O3_attention_mask = torch.ones_like(O3_input_ids)
        O3_attention_mask = O3_attention_mask.masked_fill(O3_input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        return {
            'O0_input_ids': O0_input_ids,
            'O0_attention_mask': O0_attention_mask,
            'O3_input_ids': O3_input_ids,
            'O3_attention_mask': O3_attention_mask
        }
