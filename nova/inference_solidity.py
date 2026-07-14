"""
Run inference with the fine-tuned Nova-Solidity model.

Given an EVM CFG text, generates the most likely Solidity function implementation.

Usage (interactive / single function):
    python inference_solidity.py \\
        --model_path ../checkpoints/nova-solidity-1.3b \\
        --cfg_file path/to/cfg.txt \\
        --version 0.8.9

Usage (batch over a JSON dataset file):
    python inference_solidity.py \\
        --model_path ../checkpoints/nova-solidity-1.3b \\
        --input_json ../Cleaned_Aligned_JSON/combined_0.8.9.json \\
        --output_json results_0.8.9.json \\
        --version 0.8.9 \\
        --n_samples 100

Run from the nova/ directory.
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer, NovaForCausalLM
from prepare_solidity_dataset import normalize_cfg, EVM_OPCODES


def build_prompt(version: str, cfg_text: str) -> tuple:
    """Return (input_text, char_types) for inference (no output text)."""
    cfg_norm = normalize_cfg(cfg_text)
    prompt_before = f'# This is the EVM CFG for a Solidity {version} function:\n'
    prompt_after = '\nWhat is the Solidity source code?\n'
    input_text = prompt_before + cfg_norm + prompt_after
    char_types = (
        '0' * len(prompt_before) +
        '1' * len(cfg_norm) +
        '0' * len(prompt_after)
    )
    return input_text, char_types


def load_model_and_tokenizer(model_path: str, device: str = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(
        'deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True
    )
    tokenizer.add_tokens(
        ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
        special_tokens=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    nova_tok = NovaTokenizer(tokenizer)

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = NovaForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    return tokenizer, nova_tok, model


@torch.no_grad()
def generate_solidity(
    model, tokenizer, nova_tok,
    cfg_text: str,
    version: str = '0.8.x',
    max_new_tokens: int = 512,
    num_return_sequences: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    device: str = 'cuda',
) -> list[str]:
    """Generate Solidity source code from a CFG text.

    Returns a list of `num_return_sequences` candidate Solidity strings.
    Use num_return_sequences=1 for greedy/deterministic output (temperature=0.2
    gives near-greedy behaviour while still allowing diversity in beam search).
    """
    input_text, char_types = build_prompt(version, cfg_text)
    enc = nova_tok.encode(input_text, '', char_types)

    input_ids = torch.LongTensor([enc['input_ids'].tolist()]).to(device)
    nova_mask = torch.LongTensor(enc['nova_attention_mask']).unsqueeze(0).to(device)
    no_mask_idx = torch.LongTensor([enc['no_mask_idx']]).to(device)

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=(num_return_sequences > 1 or temperature > 0),
        nova_attention_mask=nova_mask,
        no_mask_idx=no_mask_idx,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = input_ids.size(1)
    results = []
    for out in outputs:
        decoded = tokenizer.decode(
            out[input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        results.append(decoded.strip())
    return results


def run_single(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading model from {args.model_path} on {device} ...')
    tokenizer, nova_tok, model = load_model_and_tokenizer(args.model_path, device)

    with open(args.cfg_file) as f:
        cfg_text = f.read()

    candidates = generate_solidity(
        model, tokenizer, nova_tok,
        cfg_text=cfg_text,
        version=args.version,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        device=device,
    )
    for i, c in enumerate(candidates):
        print(f'\n--- Candidate {i+1} ---\n{c}')


def run_batch(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading model from {args.model_path} on {device} ...')
    tokenizer, nova_tok, model = load_model_and_tokenizer(args.model_path, device)

    with open(args.input_json) as f:
        data = json.load(f)

    func_names = list(data.keys())
    if args.n_samples:
        func_names = func_names[:args.n_samples]

    results = {}
    for idx, func_name in enumerate(func_names):
        entry = data[func_name]
        cfg = entry.get('cfg_representation', '').strip()
        gt = entry.get('solidity_definition', '').strip()
        if not cfg:
            continue

        candidates = generate_solidity(
            model, tokenizer, nova_tok,
            cfg_text=cfg,
            version=args.version,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            device=device,
        )
        results[func_name] = {
            'cfg': cfg,
            'ground_truth': gt,
            'predictions': candidates,
        }
        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{len(func_names)}] done')

        # Save incrementally so we don't lose progress
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'Results saved to {args.output_json}')


def main():
    parser = argparse.ArgumentParser(description='Nova-Solidity inference')
    parser.add_argument('--model_path', required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--version', default='0.8.x',
                        help='Solidity pragma version hint (e.g. 0.8.9)')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    # Single-function mode
    parser.add_argument('--cfg_file', default=None,
                        help='Path to a text file containing a single CFG')

    # Batch mode
    parser.add_argument('--input_json', default=None,
                        help='Path to a combined_*.json dataset file')
    parser.add_argument('--output_json', default='inference_results.json')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of functions to evaluate (default: all)')

    args = parser.parse_args()

    if args.cfg_file:
        run_single(args)
    elif args.input_json:
        run_batch(args)
    else:
        parser.error('Provide either --cfg_file or --input_json')


if __name__ == '__main__':
    main()
