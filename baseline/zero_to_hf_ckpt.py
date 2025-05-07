import torch
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
tokenizer.add_tokens(['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)], special_tokens=True)

model = AutoModel.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer.get_vocab()))

deepspeed_model_save_folder = ''
huggingface_model_save_folder = ''

model.load_state_dict(
    torch.load(f'{deepspeed_model_save_folder}/mp_rank_00_model_states.pt', map_location='cpu')['module']
)
model.save_pretrained(huggingface_model_save_folder)
