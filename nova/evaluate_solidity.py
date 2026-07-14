"""
Evaluate the Nova-Solidity decompiler on a results JSON file.

Computes:
  - BLEU-1 through BLEU-4 (syntactic similarity, via NLTK)
  - CodeBERT cosine similarity (semantic similarity)
  - Exact match rate

Input JSON format (produced by inference_solidity.py):
    {
      "functionName": {
        "ground_truth": "function foo(...) { ... }",
        "predictions": ["function foo(...) { ... }", ...]
      }, ...
    }

Usage:
    python evaluate_solidity.py \\
        --results_json inference_results.json \\
        --output_csv evaluation_scores.csv

Run from the nova/ directory.
"""

import argparse
import csv
import json
import os
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel

CODEBERT_MODEL = 'microsoft/codebert-base'
smoother = SmoothingFunction().method4


def tokenize_for_bleu(code: str) -> list[str]:
    """Character-level tokenization preserving Solidity keywords."""
    import re
    # Split on whitespace and punctuation boundaries
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return tokens


def bleu_scores(reference: str, hypothesis: str) -> dict:
    ref_tokens = tokenize_for_bleu(reference)
    hyp_tokens = tokenize_for_bleu(hypothesis)
    if not hyp_tokens:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
    scores = {}
    for n in [1, 2, 3, 4]:
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f'bleu{n}'] = sentence_bleu(
            [ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoother
        )
    return scores


class CodeBERTScorer:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        print(f'Loading CodeBERT ({CODEBERT_MODEL}) ...')
        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
        self.model = AutoModel.from_pretrained(CODEBERT_MODEL).to(device).eval()

    @torch.no_grad()
    def embed(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(
            code, return_tensors='pt', max_length=512,
            truncation=True, padding=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        # Use [CLS] token embedding as the function representation
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return emb

    def similarity(self, ref: str, hyp: str) -> float:
        e1 = self.embed(ref)
        e2 = self.embed(hyp)
        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
        return float(cos)


def evaluate(results_json: str, output_csv: str, use_codebert: bool = True):
    with open(results_json) as f:
        data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = CodeBERTScorer(device) if use_codebert else None

    rows = []
    bleu1_all, bleu2_all, bleu3_all, bleu4_all, codebert_all = [], [], [], [], []
    exact_matches = 0

    for func_name, entry in data.items():
        gt = entry.get('ground_truth', '').strip()
        preds = entry.get('predictions', [])
        if not gt or not preds:
            continue

        # Use the first prediction as the primary candidate
        pred = preds[0].strip()

        b = bleu_scores(gt, pred)
        bleu1_all.append(b['bleu1'])
        bleu2_all.append(b['bleu2'])
        bleu3_all.append(b['bleu3'])
        bleu4_all.append(b['bleu4'])

        cb_sim = scorer.similarity(gt, pred) if scorer else 0.0
        codebert_all.append(cb_sim)

        exact = 1 if pred == gt else 0
        exact_matches += exact

        rows.append({
            'function': func_name,
            'bleu1': round(b['bleu1'], 4),
            'bleu2': round(b['bleu2'], 4),
            'bleu3': round(b['bleu3'], 4),
            'bleu4': round(b['bleu4'], 4),
            'codebert_sim': round(cb_sim, 4),
            'exact_match': exact,
        })

    n = len(rows)
    print(f'\n=== Evaluation Results ({n} functions) ===')
    print(f'  BLEU-1:           {np.mean(bleu1_all):.4f}')
    print(f'  BLEU-2:           {np.mean(bleu2_all):.4f}')
    print(f'  BLEU-3:           {np.mean(bleu3_all):.4f}')
    print(f'  BLEU-4:           {np.mean(bleu4_all):.4f}')
    if use_codebert:
        print(f'  CodeBERT sim:     {np.mean(codebert_all):.4f}')
    print(f'  Exact match:      {exact_matches}/{n} = {exact_matches/n:.4f}')

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nPer-function scores saved to {output_csv}')

    # Summary row
    summary = {
        'function': 'AVERAGE',
        'bleu1': round(np.mean(bleu1_all), 4),
        'bleu2': round(np.mean(bleu2_all), 4),
        'bleu3': round(np.mean(bleu3_all), 4),
        'bleu4': round(np.mean(bleu4_all), 4),
        'codebert_sim': round(np.mean(codebert_all), 4) if use_codebert else '-',
        'exact_match': f'{exact_matches}/{n}',
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nova-Solidity decompiler')
    parser.add_argument('--results_json', required=True,
                        help='JSON file produced by inference_solidity.py')
    parser.add_argument('--output_csv', default='evaluation_scores.csv',
                        help='Where to save per-function scores')
    parser.add_argument('--no_codebert', action='store_true',
                        help='Skip CodeBERT scoring (faster, BLEU only)')
    args = parser.parse_args()

    evaluate(
        args.results_json,
        args.output_csv,
        use_codebert=not args.no_codebert,
    )


if __name__ == '__main__':
    main()
