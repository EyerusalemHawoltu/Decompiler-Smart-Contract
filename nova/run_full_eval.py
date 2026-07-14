"""
run_full_eval.py — Run inference on all test samples and compute BLEU-4 + exact match.

Features:
  - Checkpoint/resume: saves every SAVE_EVERY samples, resumes from where it left off
  - Greedy decoding (fastest)
  - Writes results/full_eval.json  (used by the website eval page)
  - Writes results/full_scores.csv (summary per sample)

Usage (submit via SLURM or run interactively on GPU node):
    cd /scratch/eh3115/Decompliler/Decompliler/nova
    python run_full_eval.py

Run from the nova/ directory.
"""

import json, os, sys, time, csv, math
import torch
from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer, NovaForCausalLM
from prepare_solidity_dataset import normalize_cfg
from prepare_gigahorse_dataset import normalize_tac

# ── Paths (relative to nova/) ────────────────────────────────────────────────
# CKPT_DIR / EVAL_OUT / EVAL_CSV can be overridden via env so a fresh model can
# be evaluated without clobbering a previous run's results (or resuming stale).
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT       = os.environ.get(
                 'CKPT_DIR',
                 os.path.join(ROOT, 'checkpoints', 'nova-solidity-1.3b-clean', 'checkpoint-1467'))
TEST_JSON  = os.environ.get('TEST_JSON', os.path.join(ROOT, 'data-tokenized', 'test_set.json'))
OUT_JSON   = os.environ.get('EVAL_OUT', os.path.join(ROOT, 'results', 'full_eval_retrained.json'))
SCORES_CSV = os.environ.get('EVAL_CSV', os.path.join(ROOT, 'results', 'full_scores_retrained.csv'))
# representation knobs: REPR_FIELD = cfg_representation | tac_representation
#                       REPR_LABEL = CFG | TAC   (used in the prompt)
REPR_FIELD = os.environ.get('REPR_FIELD', 'cfg_representation')
REPR_LABEL = os.environ.get('REPR_LABEL', 'CFG')
N_SAMPLE   = int(os.environ.get('N_SAMPLE', '0'))   # 0 = all

MAX_NEW_TOKENS = 512
SAVE_EVERY     = 100     # save checkpoint every N samples
LOG_EVERY      = 50

# ── BLEU-4 ────────────────────────────────────────────────────────────────────
def bleu_n(ref_tokens, hyp_tokens, n):
    from collections import Counter
    if len(hyp_tokens) < n:
        return 0.0
    hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
    matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
    total   = max(sum(hyp_ngrams.values()), 1)
    return matches / total

def brevity_penalty(ref_len, hyp_len):
    if hyp_len >= ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - ref_len / hyp_len)

def compute_bleu4(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not hyp:
        return 0.0
    scores = [bleu_n(ref, hyp, n) for n in range(1, 5)]
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s) for s in scores) / 4
    bp = brevity_penalty(len(ref), len(hyp))
    return round(bp * math.exp(log_avg), 4)

def compute_exact(reference: str, hypothesis: str) -> int:
    def norm(s):
        return ' '.join(s.split())
    return 1 if norm(reference) == norm(hypothesis) else 0

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[eval] device = {device}', flush=True)
    if device == 'cpu':
        print('[eval] FATAL: CUDA not available on this node — refusing to run '
              'on CPU (would take days). Exiting so SLURM reschedules.', flush=True)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(
        'deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True
    )
    tok.add_tokens(
        ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
        special_tokens=True,
    )
    tok.pad_token    = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    nova_tok = NovaTokenizer(tok)

    print(f'[eval] loading model from {CKPT}')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = NovaForCausalLM.from_pretrained(
        CKPT, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    print('[eval] model ready')
    return tok, nova_tok, model, device


@torch.no_grad()
def infer_one(tok, nova_tok, model, device, cfg_text: str, version: str) -> tuple[str, int]:
    cfg_norm      = (normalize_tac if REPR_LABEL == 'TAC' else normalize_cfg)(cfg_text.strip())
    prompt_before = f'# This is the EVM {REPR_LABEL} for a Solidity {version} function:\n'
    prompt_after  = '\nWhat is the Solidity source code?\n'
    input_text    = prompt_before + cfg_norm + prompt_after
    char_types    = '0'*len(prompt_before) + '1'*len(cfg_norm) + '0'*len(prompt_after)

    enc         = nova_tok.encode(input_text, '', char_types)
    input_ids   = torch.LongTensor([enc['input_ids'].tolist()]).to(device)
    nova_mask   = torch.LongTensor(enc['nova_attention_mask']).unsqueeze(0).to(device)
    no_mask_idx = torch.LongTensor([enc['no_mask_idx']]).to(device)

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        nova_attention_mask=nova_mask,
        no_mask_idx=no_mask_idx,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(
        outputs[0][input_ids.size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded.strip(), input_ids.size(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)

    # Load test set
    print(f'[eval] loading test set from {TEST_JSON}')
    with open(TEST_JSON) as f:
        test_data = json.load(f)

    all_keys = list(test_data.keys())
    if N_SAMPLE:
        all_keys = all_keys[:N_SAMPLE]
    total    = len(all_keys)
    print(f'[eval] {total} test samples')

    # Load existing checkpoint if any
    done = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            done = json.load(f)
        print(f'[eval] resuming — {len(done)} already done')

    remaining = [k for k in all_keys if k not in done]
    print(f'[eval] {len(remaining)} remaining')

    if not remaining:
        print('[eval] all done! Rebuilding scores CSV...')
        write_csv(done)
        return

    # Load model
    tok, nova_tok, model, device = load_model()

    # Run inference
    t0 = time.time()
    for i, key in enumerate(remaining):
        entry   = test_data[key]
        cfg     = entry.get(REPR_FIELD, '').strip()
        gt      = entry.get('solidity_definition', '').strip()
        version = entry.get('version', '0.8.x')

        if not cfg:
            done[key] = {'cfg': cfg, 'ground_truth': gt, 'predicted': '', 'bleu4': 0.0, 'exact_match': 0}
            continue

        t_sample = time.time()
        try:
            predicted, _ = infer_one(tok, nova_tok, model, device, cfg, version)
        except Exception as e:
            print(f'  [WARN] {key}: {e}')
            predicted = ''

        bleu4       = compute_bleu4(gt, predicted)
        exact_match = compute_exact(gt, predicted)

        # Clean name for display
        display_name = key.rsplit('_v', 1)[0] if '_v' in key else key

        done[key] = {
            'key':          key,
            'name':         display_name,
            'cfg':          cfg,
            'ground_truth': gt,
            'predicted':    predicted,
            'bleu4':        bleu4,
            'exact_match':  exact_match,
            'version':      version,
        }

        elapsed = time.time() - t0
        done_count = len(done)
        rate = done_count / elapsed if elapsed > 0 else 0
        eta = (total - done_count) / rate if rate > 0 else 0

        if (i + 1) % LOG_EVERY == 0 or i == 0:
            print(f'  [{done_count}/{total}] bleu4={bleu4:.3f} em={exact_match} '
                  f'sample={time.time()-t_sample:.1f}s ETA={eta/3600:.1f}h')

        # Incremental checkpoint
        if done_count % SAVE_EVERY == 0:
            _save(done)
            print(f'  [checkpoint] saved {done_count} results')

    # Final save
    _save(done)
    write_csv(done)
    _print_summary(done)


def _save(done: dict):
    with open(OUT_JSON, 'w') as f:
        json.dump(done, f)


def write_csv(done: dict):
    # 1) compact scores CSV (function, scores)
    with open(SCORES_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['function', 'bleu4', 'exact_match', 'version'])
        for key, item in done.items():
            w.writerow([key, item.get('bleu4', 0), item.get('exact_match', 0), item.get('version', '')])
    print(f'[eval] scores written to {SCORES_CSV}')

    # 2) full table CSV: cfg | expected solidity | model output (+ scores)
    table_csv = SCORES_CSV.replace('.csv', '') + '_table.csv'
    with open(table_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['function', 'version', 'cfg', 'expected_solidity',
                    'model_output', 'bleu4', 'exact_match'])
        for key, item in done.items():
            w.writerow([
                key,
                item.get('version', ''),
                item.get('cfg', ''),
                item.get('ground_truth', ''),
                item.get('predicted', ''),
                item.get('bleu4', 0),
                item.get('exact_match', 0),
            ])
    print(f'[eval] full table written to {table_csv}')


def _print_summary(done: dict):
    bleus  = [v['bleu4'] for v in done.values()]
    exacts = [v['exact_match'] for v in done.values()]
    n = len(bleus)
    if n == 0:
        return
    avg_b  = sum(bleus) / n
    em_pct = 100 * sum(exacts) / n
    print(f'\n[eval] === RESULTS ===')
    print(f'  Total:        {n}')
    print(f'  Avg BLEU-4:   {avg_b:.4f}')
    print(f'  Exact match:  {em_pct:.1f}%  ({sum(exacts)}/{n})')
    print(f'  BLEU >= 0.9:  {sum(1 for b in bleus if b >= 0.9)}')
    print(f'  BLEU < 0.5:   {sum(1 for b in bleus if b < 0.5)}')


if __name__ == '__main__':
    main()
