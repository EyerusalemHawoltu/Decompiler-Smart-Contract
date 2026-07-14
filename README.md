# AI-Powered Smart Contract Bytecode Decompiler

Recover **Solidity source** from **EVM bytecode** using a fine-tuned **Nova** model.
Bytecode is lifted into an intermediate representation, then translated to Solidity.
Two interchangeable front-ends (input representations) are supported and compared:

| Track | Input to the model | Prep script | Trained model |
|-------|--------------------|-------------|---------------|
| **A — CFG** | improved control-flow graph (BFS + PUSH0 + 453k selector table) | `nova/prepare_cfg_dataset.py` | `nova-cfg-1.3b` |
| **B — Gigahorse** | three-address code (explicit data flow) | `nova/prepare_gigahorse_dataset.py` | `nova-gigahorse-1.3b` |

Both tracks use the **same model code** and the **same leakage-free data cleaning**,
so the only difference between the two trained models is the input representation.

> For deep reference (every file, all gotchas, harvesting from scratch) see
> **[`PIPELINE.md`](PIPELINE.md)**. This README is the run-guide.

---

## 1. Get the data (Google Drive)

Large data & harvested representations are **not** in git — download them from:

**📦 https://drive.google.com/drive/folders/1MKjDQCdvkHjN8zMHZPbwBq02nJ4kgxct?usp=drive_link**

Place these under the project root so the paths below resolve:

```
Contracts_By_Version_Cleaned/<ver>/<addr>_<Name>.sol.cleaned      # ground-truth Solidity (needed to CLEAN)
datasets/cfg_dataset/cfg.jsonl                                    # harvested CFG   (Track A input)
datasets/gigahorse_dataset/gigahorse_tac.jsonl                    # harvested TAC   (Track B input)
# (Contracts_Bytecode/  is only needed if you want to re-harvest from bytecode — see PIPELINE.md §3)
```

---

## 2. Environment (once)

```bash
conda create -n nova python=3.10 -y && conda activate nova
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate deepspeed pyevmasm sentencepiece scikit-learn
export HF_HOME=$PWD/hf_cache          # cache the deepseek-coder base + tokenizer
```
On the cluster, use `/home/eh3115/.conda/envs/nova/bin/python` and `sbatch` the
scripts in `hpc/`. Everything below shows the **plain command**; each has a matching
`hpc/*.slurm` you can `sbatch` instead.

Flow for each track:  **clean data → train Nova → evaluate**.

---

## Track A — CFG

### A1. Clean the data (align → dedup → leakage-free split)
Turns `cfg.jsonl` + the Solidity source into a clean train/valid/test set:
```bash
cd nova
python prepare_cfg_dataset.py          #  or:  sbatch hpc/prep_cfg.slurm   (needs --mem=128G)
```
What it does:
1. **Align** each CFG function to its Solidity source function (by name + arity).
2. **Normalize** the CFG for Nova (adds `<label-N>` hierarchical-attention markers).
3. **Dedup by Solidity body** — *pragma-invariant*: the same function compiled under
   different compiler versions collapses to one, so it can't leak across splits.
4. **60/20/20 split** by unique body → `train ∩ test = 0`.

Output → `datasets/cfg_dataset/nova-cfg/{train,valid,test}` + `datasets/cfg_dataset/test_set.json`
(≈ 22,735 / 7,578 / 7,578 unique functions).

Verify zero leakage:
```bash
python - <<'PY'
from datasets import load_from_disk
d=load_from_disk("../datasets/cfg_dataset/nova-cfg"); b=lambda s:{" ".join(x.split()) for x in d[s]["output"]}
tr,te=b("train"),b("test"); print("train∩test:",len(tr&te),"(must be 0)")
PY
```

### A2. Train Nova
```bash
cd nova
DATA_DIR=../datasets/cfg_dataset/nova-cfg \
MODEL_SAVE=../checkpoints/nova-cfg-1.3b \
python finetune_full.py                #  or:  sbatch hpc/train_cfg.slurm   (1 GPU, ~2–3h, 3 epochs)
```
Loads `deepseek-coder-1.3b-base` into `NovaForCausalLM` (architecture unchanged).
Saves to `checkpoints/nova-cfg-1.3b/`.

### A3. Evaluate
```bash
cd nova
CKPT_DIR=../checkpoints/nova-cfg-1.3b/checkpoint-2130 \
TEST_JSON=../datasets/cfg_dataset/test_set.json \
REPR_FIELD=cfg_representation REPR_LABEL=CFG N_SAMPLE=2000 \
EVAL_OUT=../results/eval_cfg.json EVAL_CSV=../results/eval_cfg.csv \
python run_full_eval.py                #  or:  sbatch hpc/eval_model.slurm  (see PIPELINE.md §6)
```
Writes `results/eval_cfg.json` + `results/eval_cfg_table.csv` (cfg | expected | model output).

---

## Track B — Gigahorse

> Requires Gigahorse only if you re-harvest TAC from bytecode (Singularity sandbox —
> see PIPELINE.md §1/§3). If you downloaded `gigahorse_tac.jsonl` from Drive, skip
> straight to B1.

### B1. Clean the data (align → dedup → leakage-free split)
Turns `gigahorse_tac.jsonl` + the Solidity source into a clean train/valid/test set:
```bash
cd nova
python prepare_gigahorse_dataset.py    #  or:  sbatch hpc/prep_gh.slurm   (needs --mem=128G)
```
Same cleaning as Track A (align → `<label-N>` → **dedup by body / ignore pragma** →
leakage-free 60/20/20 split).
Output → `datasets/gigahorse_dataset/nova-gigahorse/{train,valid,test}` + `test_set.json`
(≈ 27,882 / 9,293 / 9,293 unique functions).

Verify zero leakage:
```bash
python - <<'PY'
from datasets import load_from_disk
d=load_from_disk("../datasets/gigahorse_dataset/nova-gigahorse"); b=lambda s:{" ".join(x.split()) for x in d[s]["output"]}
tr,te=b("train"),b("test"); print("train∩test:",len(tr&te),"(must be 0)")
PY
```

### B2. Train Nova
```bash
cd nova
DATA_DIR=../datasets/gigahorse_dataset/nova-gigahorse \
MODEL_SAVE=../checkpoints/nova-gigahorse-1.3b \
python finetune_full.py                #  or:  sbatch hpc/train_gh.slurm   (1 GPU, ~2–3h, 3 epochs)
```
Saves to `checkpoints/nova-gigahorse-1.3b/`.

### B3. Evaluate
```bash
cd nova
CKPT_DIR=../checkpoints/nova-gigahorse-1.3b/checkpoint-2613 \
TEST_JSON=../datasets/gigahorse_dataset/test_set.json \
REPR_FIELD=tac_representation REPR_LABEL=TAC N_SAMPLE=2000 \
EVAL_OUT=../results/eval_gh.json EVAL_CSV=../results/eval_gh.csv \
python run_full_eval.py
```
Writes `results/eval_gh.json` + `results/eval_gh_table.csv`.

---

## 3. Compare the two tracks

```bash
python compare_models.py       # BLEU-4 / exact% / normalized-match% for CFG vs Gigahorse
python normalized_match.py     # fair "correct modulo variable names" metric
```

### Reference results (leakage-free, 2,000-sample)
| Model | BLEU-4 | Exact | Normalized-match |
|-------|-------:|------:|-----------------:|
| nova-cfg-1.3b | 0.171 | 0.45% | 6.55% |
| nova-gigahorse-1.3b | 0.154 | 0.55% | 6.80% |

**Finding:** the two tie → the input representation is **not** the bottleneck (model
capacity is). Note: pre-dedup scores (~0.79 BLEU) were **train/test leakage /
memorization**, not real generalization — always verify `train ∩ test = 0`.

> ⚠️ BLEU / exact-match are harsh (they punish unrecoverable variable renames) and
> CodeBERT cosine is **saturated** (scores a wrong function ~0.97). Use
> `normalized_match.py` for a fair number. See PIPELINE.md §7.

---

## Acknowledgements

Built on **Nova** (model architecture in `nova/modeling_nova.py`, unchanged). If you
use this work, please also cite Nova:

```bibtex
@inproceedings{jiang2025nova,
  title={Nova: Generative Language Models for Assembly Code with Hierarchical Attention and Contrastive Learning},
  author={Nan Jiang and Chengxiao Wang and Kevin Liu and Xiangzhe Xu and Lin Tan and Xiangyu Zhang and Petr Babkin},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
