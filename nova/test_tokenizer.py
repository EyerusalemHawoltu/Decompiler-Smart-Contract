from modeling_nova import NovaTokenizer
import numpy as np
from transformers import AutoTokenizer

prompt_before = "# This is the assembly code:"
asm = "mov eax , $1<label-1>mov ebx , $2<label-2>mov ecx , eax<label-3>add ecx , ebx<label-4>"
prompt_after = "What is the source code?"

input_text = prompt_before + asm
output_text = prompt_after
char_types = '0' * len(prompt_before) + '1' * len(asm) + '0' * len(prompt_after)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)
nova_tokenizer = NovaTokenizer(tokenizer)

output = nova_tokenizer.encode(input_text, output_text, char_types)
print(output['input_ids'])
print(output['labels'])
input_ids = tokenizer.convert_ids_to_tokens(output['input_ids'].tolist())
input_ids = [
    token.replace('Ċ', '\\n').replace('Ġ', ' ').replace('ĉ', '\\t').replace('<label', '[INST').replace('>', ']') for token in input_ids
]
attention_mask = output['nova_attention_mask']
attention_mask[-6:,  np.array([13, 20, 27, 34], dtype=np.int32)] = 0.5
# attention_mask = attention_mask.tolist()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, '#440154'), (0.5, '#20a486'), (0.7, '#75d054'), (0.9, '#fde725'), (1., '#e6e6e6')]
# colors = [(0, '#000000'), (0.5, '#a6cee3'), (0.7, '#ffd966'), (0.9, '#b2df8a'), (1., '#ffffbf')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

plt.figure(figsize=(9, 9))
ax = sns.heatmap(attention_mask, linewidths=0.1, cmap=cmap, cbar=False)
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(attention_mask)) + 0.5, input_ids, fontsize=13)
ax.set_yticks(np.arange(len(attention_mask)) + 0.5, input_ids, fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.savefig('attention.pdf', bbox_inches='tight')
plt.savefig('attention.svg', bbox_inches='tight', format='svg')