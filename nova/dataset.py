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
        
        L = self.data[ind]['nova_attention_mask'].size(0)
        
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        nova_attention_mask = torch.zeros(input_ids.size(0), input_ids.size(0)).type(torch.bool)
        nova_attention_mask[: L, : L] = self.data[ind]['nova_attention_mask']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'nova_attention_mask': nova_attention_mask,
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
        result = {'input_ids': [], 'labels': [], 'attention_mask': [], 'nova_attention_mask': []}
        for k in ['src', 'asm-O0', 'asm-O1', 'asm-O2', 'asm-O3']:
            input_ids = self.data[ind][f'{k}-input_ids'].unsqueeze(0)
            labels = self.data[ind][f'{k}-labels'].unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)

            L = self.data[ind][f'{k}-nova_attention_mask'].size(0)
            nova_attention_mask = torch.zeros(self.data[ind][f'{k}-input_ids'].size(0), self.data[ind][f'{k}-input_ids'].size(0)).type(torch.bool)
            nova_attention_mask[: L, : L] = self.data[ind][f'{k}-nova_attention_mask']
            
            result['input_ids'].append(input_ids)
            result['labels'].append(labels)
            result['attention_mask'].append(attention_mask)
            result['nova_attention_mask'].append(nova_attention_mask.unsqueeze(0))
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

        O0_L = self.data[ind]['asm-O0-nova_attention_mask'].size(0)
        O3_L = self.data[ind]['asm-O3-nova_attention_mask'].size(0)

        O0_nova_attention_mask = torch.zeros(O0_input_ids.size(0), O0_input_ids.size(0)).type(torch.bool)
        O0_nova_attention_mask[: O0_L, : O0_L] = self.data[ind]['asm-O0-nova_attention_mask']
        O3_nova_attention_mask = torch.zeros(O3_input_ids.size(0), O3_input_ids.size(0)).type(torch.bool)
        O3_nova_attention_mask[: O3_L, : O3_L] = self.data[ind]['asm-O3-nova_attention_mask']

        O0_attention_mask = torch.ones_like(O0_input_ids)
        O0_attention_mask = O0_attention_mask.masked_fill(O0_input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        O3_attention_mask = torch.ones_like(O3_input_ids)
        O3_attention_mask = O3_attention_mask.masked_fill(O3_input_ids.eq(self.tokenizer.pad_token_id), 0.0).type(torch.bool)
        return {
            'O0_input_ids': O0_input_ids,
            'O0_attention_mask': O0_attention_mask,
            'O0_nova_attention_mask': O0_nova_attention_mask,
            'O3_input_ids': O3_input_ids,
            'O3_attention_mask': O3_attention_mask,
            'O3_nova_attention_mask': O3_nova_attention_mask
        }
