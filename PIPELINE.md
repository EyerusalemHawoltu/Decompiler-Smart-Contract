# Nova Smart-Contract Decompiler — Full Pipeline

End-to-end pipeline that recovers **Solidity source** from **EVM bytecode** by
lifting bytecode into an intermediate representation and translating it with a
fine-tuned **Nova** language model. (The base Nova model & citation are in
[`README.md`](README.md); this file documents *our* decompilation pipeline.)

It supports **two interchangeable front-ends** (input representations):

| Front-end | What it produces | Harvest script |
|-----------|------------------|----------------|
| **Improved CFG** | basic blocks + stack opcodes (BFS dispatcher recovery, PUSH0, 453k selector table) | `run_cfg_dataset.py` |
| **Gigahorse TAC** | three-address code with explicit data flow + internal functions | `run_gigahorse_dataset.py` |

Both are trained and evaluated with the *same* code and the *same* leakage-free
data-cleaning discipline, so the only difference between the two trained models is
the input representation.

---

## Data & Models Access

The large data and model artifacts are **not** stored in this git repository
(GitHub caps files at 100 MB; these total ~130 GB). They are available here:

**📦 Google Drive:** https://drive.google.com/drive/folders/1MKjDQCdvkHjN8zMHZPbwBq02nJ4kgxct?usp=drive_link

This includes (as applicable):
- `Contracts_Bytecode/` — deployed runtime bytecode (`.hex`, 109,679 contracts)
- `Contracts_By_Version_Cleaned/` — ground-truth Solidity (`.sol.cleaned`)
- `datasets/cfg_dataset/`, `datasets/gigahorse_dataset/` — harvested `*.jsonl`,
  the prepared HF datasets (`nova-cfg/`, `nova-gigahorse/`), and `test_set.json`
- `checkpoints/nova-cfg-1.3b/`, `checkpoints/nova-gigahorse-1.3b/` — trained models

After cloning the repo, download these into the project root so the paths in the
stages below resolve. The repo itself holds the **code, SLURM scripts, docs, and
small result CSVs**.

---

## 0. Repository layout

```
Decompliler/
├── Contracts_Bytecode/<ver>/<addr>_<Name>.hex                    # deployed runtime bytecode (input)
├── Contracts_By_Version_Cleaned/<ver>/<addr>_<Name>.sol.cleaned  # ground-truth Solidity
│
├── evm_cfg_builder/                # IMPROVED CFG builder (BFS + PUSH0 + merged 453k hashes)
├── gigahorse-toolchain/            # Gigahorse clone (source)
├── gigahorse_sbx/                  # Gigahorse Singularity SANDBOX (built, runnable)   [HPC]
├── gh_cache/                       # Gigahorse compiled-Datalog cache (persists)        [HPC]
│
├── nova/                           # model + pipeline code
│   ├── modeling_nova.py            #   Nova architecture      (UNCHANGED from upstream)
│   ├── generation_utils.py         #   generation logic       (UNCHANGED)
│   ├── dataset.py                  #   SolidityDataset (builds attention mask on the fly)
│   ├── bytecode_to_cfg.py          #   extract_function_cfgs()  -> improved CFG text
│   ├── selector_extract.py         #   lightweight dispatcher-only extractor (baseline)
│   ├── prepare_solidity_dataset.py #   original CFG prep (provides normalize_cfg + dedup helpers)
│   ├── prepare_cfg_dataset.py      #   ← DATA CLEANING for the improved-CFG dataset
│   ├── prepare_gigahorse_dataset.py#   ← DATA CLEANING for the Gigahorse-TAC dataset
│   ├── finetune_full.py            #   training (reads DATA_DIR / MODEL_SAVE from env)
│   └── run_full_eval.py            #   evaluation (env-configurable repr / checkpoint / test set)
│
├── run_cfg_dataset.py              # STAGE 1a: harvest per-function CFG for all contracts
├── run_gigahorse_dataset.py        # STAGE 1b: harvest per-function Gigahorse TAC
├── gigahorse_tac_serialize.py      #   Gigahorse relational CSVs -> per-function TAC text
│
├── normalized_match.py             # fair "correct modulo variable names" metric
├── compare_models.py               # head-to-head: CFG vs Gigahorse
├── codebert_sim.py                 # CodeBERT semantic similarity (NOTE: saturated — see §7)
│
├── datasets/
│   ├── cfg_dataset/                # cfg.jsonl -> nova-cfg/ (HF) + test_set.json
│   └── gigahorse_dataset/          # gigahorse_tac.jsonl -> nova-gigahorse/ (HF) + test_set.json
│
├── checkpoints/                    # nova-cfg-1.3b/ , nova-gigahorse-1.3b/
├── hpc/                            # SLURM scripts (one per stage)
└── results/                        # eval outputs + comparison
```

---

## 1. Environment

### Conda env (`nova`)
```bash
conda create -n nova python=3.10 -y
conda activate nova
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate deepspeed pyevmasm sentencepiece
pip install scikit-learn            # only for codebert_sim.py
```
On the cluster the env is `/home/eh3115/.conda/envs/nova/bin/python` — commands
below use that interpreter (adjust for your machine).

### Base model & tokenizer
Training loads `deepseek-ai/deepseek-coder-1.3b-base` into `NovaForCausalLM` and
adds Nova's special tokens (`<unk>`, `<cls>`, `<label-1..256>`). It downloads from
HuggingFace on first run — set `HF_HOME=/scratch/$USER/hf_cache` (download once on
a node with internet so compute nodes read the cache).

### Gigahorse (only for the Gigahorse front-end)
Gigahorse is a Souffle/Datalog decompiler; run it via Singularity **sandbox**:
```bash
module load singularity/3.8.3
export SINGULARITY_CACHEDIR=/scratch/$USER/.sing_cache SINGULARITY_TMPDIR=/scratch/$USER/.sing_tmp
# Build a SANDBOX dir (the .sif squashfs won't mount on older kernels). Do this on a COMPUTE node.
singularity build --sandbox gigahorse_sbx docker://ghcr.io/nevillegrech/gigahorse-toolchain:latest
```
> The first Gigahorse run compiles the Datalog into `gh_cache/` (~minutes); later
> runs reuse it, so keep `gh_cache/` around.

---

## 2. Pipeline overview

```
bytecode ─► [1] harvest representation ─► *.jsonl
                                            │
              [2] DATA CLEANING  (align → dedup by body → leakage-free 60/20/20 split)
                                            │
                          nova-<repr>/ (train/valid/test)  +  test_set.json
                                            │
              [3] fine-tune Nova ─► checkpoints/nova-<repr>-1.3b
                                            │
              [4] evaluate ─► results/eval_<repr>.json
                                            │
              [5] compare CFG vs Gigahorse
```
Run the two front-ends independently; they share only the source Solidity.

---

## 3. STAGE 1 — Harvest the representation (bytecode → jsonl)

Each harvester walks all **109,679 `.hex` files**, emits one record per contract
`{stem, version, n_func, <repr>: {func_signature: text}}`, appends to a single
JSONL (inode-safe), and is **resumable** (skips stems already present).

### 1a. Improved CFG  (CPU, ~15 min)
```bash
sbatch hpc/cfg_dataset.slurm          # -> datasets/cfg_dataset/cfg.jsonl
```
`run_cfg_dataset.py` calls `bytecode_to_cfg.extract_function_cfgs()` — the improved
builder (BFS dispatcher reconstruction, PUSH0 opcode support, merged 453k selector
table). Env: `CFG_OUT`, `CFG_WORKERS` (default 16).

**Sanity check the improvements are active** (0.8.20 uses PUSH0 → must be > 0):
```bash
python - <<'PY'
import json; v=[0,0]
for l in open("datasets/cfg_dataset/cfg.jsonl"):
    r=json.loads(l)
    if r["version"]=="0.8.20": v[0]+=1; v[1]+=r["n_func"]
print("0.8.20:", v[0], "contracts,", v[1], "functions")   # expect ~629 / ~6700
PY
```

### 1b. Gigahorse TAC  (CPU, Singularity, ~6 h — resumable/requeue)
```bash
sbatch hpc/gigahorse_dataset.slurm    # -> datasets/gigahorse_dataset/gigahorse_tac.jsonl (~4.7 GB)
```
`run_gigahorse_dataset.py` per batch: stage `.hex` → run `gigahorse.py` inside the
sandbox → serialize each contract's TAC via `gigahorse_tac_serialize.serialize_tac`
→ write `{stem,version,n_func,tac:{sig:tac_text}}` → **delete the batch's
~150-files-per-contract output** (protects the inode quota).
Env: `GH_SBX`, `GH_CACHE`, `GH_WORK`, `GH_TAC`, `GH_JOBS`, `GH_BATCH`, `GH_TIMEOUT`.

TAC example (what feeds the model):
```
function transfer(address,uint256)
  block:
    v4 = SUB v3 0x4
    v5 = SLT v4 0x40
    v6 = ISZERO v5
    JUMPI 0x749 v6
```

---

## 4. STAGE 2 — DATA CLEANING (the critical stage: jsonl → clean splits)

Both front-ends use identical logic (`prepare_cfg_dataset.py` /
`prepare_gigahorse_dataset.py` share `extract_sol_functions` + `match_body`).

**Step by step:**

1. **Align to Solidity.** For each harvested function signature (e.g.
   `transfer(address,uint256)`), parse the contract's `.sol.cleaned`, extract every
   `function <name>(…){…}` (brace-matched), and match by **name (+arity tiebreak
   for overloads)** → `(representation, solidity_body)` pairs.

2. **Normalize input for Nova.** Add `<label-N>` markers on each instruction/
   statement line (hierarchical-attention anchors): `normalize_cfg` for CFG,
   `normalize_tac` for TAC. Build
   `input = "# This is the EVM <CFG|TAC> for a Solidity <ver> function:\n<repr>\nWhat is the Solidity source code?\n"`,
   `output = solidity_body`, and a `char_types` mask (`0` prompt / `1` repr / `0` output).

3. **Dedup by SOLIDITY BODY — pragma-invariant (removes contamination).**
   Group all pairs by `md5(normalized_solidity_body)`. The *same function under
   different pragma versions* has an identical body but different bytecode → it
   collapses into ONE group. Keep **one representative per body** (richest repr).
   This is exactly "remove duplicates / ignore the pragma."

4. **Leakage-free 60/20/20 split.** Shuffle the *unique-body* keys (seed 42) and
   assign **whole bodies** to train/valid/test. A body lives in exactly one split →
   `train ∩ test = 0`.

5. **Emit** an HF `DatasetDict` (`train/valid/test`; columns
   `version, func_name, input, output, char_types`) + `test_set.json`
   (`{key:{cfg_representation|tac_representation, solidity_definition, version}}`).

### Run it
```bash
sbatch hpc/prep_cfg.slurm     # -> datasets/cfg_dataset/nova-cfg/ + test_set.json
sbatch hpc/prep_gh.slurm      # -> datasets/gigahorse_dataset/nova-gigahorse/ + test_set.json
```
> Needs RAM (the Gigahorse JSONL is 4.7 GB) — the SLURM scripts request `--mem=128G`.
> **Never run prep on the login node** (it will OOM / be killed).

### ALWAYS verify zero leakage
```bash
python - <<'PY'
from datasets import load_from_disk
d = load_from_disk("datasets/gigahorse_dataset/nova-gigahorse")   # or cfg_dataset/nova-cfg
b = lambda s: {" ".join(x.split()) for x in d[s]["output"]}
tr,va,te = b("train"),b("valid"),b("test")
print("unique bodies train/valid/test:", len(tr),len(va),len(te))
print("train∩test:", len(tr&te), " train∩valid:", len(tr&va), " test∩valid:", len(te&va))  # all 0
PY
```
Reference sizes — **CFG**: 22,735 / 7,578 / 7,578 — **Gigahorse**: 27,882 / 9,293 / 9,293.

---

## 5. STAGE 3 — Fine-tune Nova (one model per representation)

`finetune_full.py` loads deepseek-coder-1.3b into `NovaForCausalLM`, trains on
`$DATA_DIR`, saves to `$MODEL_SAVE`. Architecture unchanged; only data/paths differ.
Config: 3 epochs, batch 8 × grad-accum 4 (eff. 32), BF16, cosine LR 2e-5,
`save_strategy=epoch`, resumable via `get_last_checkpoint`, 1 GPU.

```bash
sbatch hpc/train_cfg.slurm    # DATA_DIR=…/nova-cfg,       MODEL_SAVE=…/checkpoints/nova-cfg-1.3b
sbatch hpc/train_gh.slurm     # DATA_DIR=…/nova-gigahorse, MODEL_SAVE=…/checkpoints/nova-gigahorse-1.3b
```
~2–3 h each on one H100/H200. A correct start prints e.g.
`Train: 27882 | Valid: 9293` and `Fresh start (no prior checkpoint)`.
The scripts set `--requeue` and save optimizer/scheduler/RNG so preempted jobs
resume cleanly.

---

## 6. STAGE 4 — Evaluate  +  STAGE 5 — Compare

`run_full_eval.py` is env-configurable so one script evals either model:

| Env var | CFG model | Gigahorse model |
|---------|-----------|-----------------|
| `CKPT_DIR` | `…/nova-cfg-1.3b/checkpoint-2130` | `…/nova-gigahorse-1.3b/checkpoint-2613` |
| `TEST_JSON` | `datasets/cfg_dataset/test_set.json` | `datasets/gigahorse_dataset/test_set.json` |
| `REPR_FIELD` | `cfg_representation` | `tac_representation` |
| `REPR_LABEL` | `CFG` | `TAC` |
| `N_SAMPLE` | `0`=all, or `2000` for a quick pass | |
| `EVAL_OUT` / `EVAL_CSV` | `results/eval_cfg.json` / `.csv` | `results/eval_gh.json` / `.csv` |

```bash
sbatch --job-name=eval_cfg --export=ALL,CKPT_DIR=…/nova-cfg-1.3b/checkpoint-2130,\
TEST_JSON=…/datasets/cfg_dataset/test_set.json,REPR_FIELD=cfg_representation,REPR_LABEL=CFG,\
N_SAMPLE=2000,EVAL_OUT=…/results/eval_cfg.json,EVAL_CSV=…/results/eval_cfg.csv hpc/eval_model.slurm

sbatch --job-name=eval_gh --export=ALL,CKPT_DIR=…/nova-gigahorse-1.3b/checkpoint-2613,\
TEST_JSON=…/datasets/gigahorse_dataset/test_set.json,REPR_FIELD=tac_representation,REPR_LABEL=TAC,\
N_SAMPLE=2000,EVAL_OUT=…/results/eval_gh.json,EVAL_CSV=…/results/eval_gh.csv hpc/eval_model.slurm
```
Each writes `eval_<x>.json` (per-function cfg/ground_truth/predicted/bleu4/exact)
and `eval_<x>_table.csv` (readable cfg | expected | output).

### Head-to-head
```bash
python compare_models.py       # BLEU-4 / exact% / normalized-match% for both models
python normalized_match.py     # fair "correct modulo variable names" (runs on *_table.csv)
```

---

## 7. Evaluation notes (read before trusting a number)

- **BLEU / exact-match are harsh** — they punish *unrecoverable* variable renames
  (names are erased at compile time). A functionally-correct output can score 0.
- **CodeBERT cosine is SATURATED** — scores a hallucinated function ~0.97.
  `codebert_sim.py` is kept for completeness but **do not headline it**.
- **Use `normalized_match.py`** for a fair number: α-renames locals (credits
  renames) but still fails wrong types / different logic.

### Reference results (leakage-free, 2,000-sample)
| Model | BLEU-4 | Exact | Norm-match |
|-------|-------:|------:|-----------:|
| nova-cfg-1.3b | 0.171 | 0.45% | 6.55% |
| nova-gigahorse-1.3b | 0.154 | 0.55% | 6.80% |

**Finding:** the two tie → input representation is **not** the bottleneck (model
capacity is). Pre-dedup numbers (~0.79 BLEU) were **train/test leakage /
memorization**, not real performance.

---

## 8. Gotchas / lessons (all already handled in the scripts)

| Problem | Fix |
|---------|-----|
| Train/test leakage from splitting by pragma | dedup by **body**, split by body (Stage 2) |
| Gigahorse writes ~150 files/contract → inode quota | batch + harvest + delete per batch |
| `.sif` won't mount ("bad superblock") | build a **sandbox** dir instead |
| Gigahorse "read-only cache" error | bind a writable `gh_cache` over the container cache |
| Singularity build "Failed to create thread" | build on a **compute node**, not login |
| `python: command not found` in SLURM | use full conda path `/…/envs/nova/bin/python` |
| eval silently runs on CPU (0% GPU, hours wasted) | `run_full_eval.py` fail-fasts if CUDA unavailable |
| prep OOM on login node | run prep as a `--mem=128G` SLURM job |
| PUSH0 (`0x5f`) decoded as INVALID on 0.8.20 | `_patch_push0()` in `evm_cfg_builder/.../cfg/cfg.py` |

---

## 9. Quick end-to-end

```bash
# ── CFG track ──
sbatch hpc/cfg_dataset.slurm      # 1a harvest
sbatch hpc/prep_cfg.slurm         # 2  clean/dedup/split
sbatch hpc/train_cfg.slurm        # 3  train
#     eval_cfg (see §6)           # 4

# ── Gigahorse track ──
# one-time: build gigahorse_sbx, warm gh_cache
sbatch hpc/gigahorse_dataset.slurm  # 1b harvest (~6h)
sbatch hpc/prep_gh.slurm            # 2  clean/dedup/split
sbatch hpc/train_gh.slurm           # 3  train
#     eval_gh (see §6)              # 4

python compare_models.py            # 5  CFG vs Gigahorse
```
Chain stages automatically with `sbatch --dependency=afterok:<jobid>`.
