# Nova: Generative Language Model For Assembly Code

## Abstract
Binary code analysis is the foundation of crucial tasks in the security domain; thus building effective binary analysis techniques is more important than ever. Large language models (LLMs) although have brought impressive improvement to source code tasks, do not directly generalize to assembly code due to the unique challenges of assembly: (1) the low information density of assembly and (2) the diverse optimizations in assembly code. To overcome these challenges, this work proposes a hierarchical attention mechanism that builds attention summaries to capture the semantics more effectively, and designs contrastive learning objectives to train LLMs to learn assembly optimization. Equipped with these techniques, this work develops Nova, a generative LLM for assembly code. Nova outperforms existing techniques on binary code decompilation by up to 146.54%, and outperforms the latest binary code similarity detection techniques by up to 6.17%, showing promising abilities on both assembly generation and understanding tasks.

## Introduction of Nova
Nova is pre-trained with the language modeling objective starting from DeepSeek-Coder checkpoints, using the disassembly code from [AnghaBench](https://github.com/albertan017/LLM4Decompile) and C/C++ program compiled from [The-Stack](https://huggingface.co/datasets/bigcode/the-stack).

This is the repository of the instruciton-tuned model of Nova that is good at binary code recovery, with 1.3B parameters.
The other models in this series:
- [Nova-1.3b](https://huggingface.co/lt-asset/nova-1.3b): Foundation model for binary code with 1.3B parameters.
- [Nova-1.3b-bcr](https://huggingface.co/lt-asset/nova-1.3b-bcr): Nova-1.3b model further instruction-tuned for binary code recovery.

## Dependencies

```bash
conda create -n nova python=3.10
conda activate nova

pip install -r requirements.txt
```

If encounter compatability issue during packages installation, downgrade vllm.

Or use a docker image:
```bash
docker pull jiang719/nova

docker run --gpus all -it jiang719/nova
```

## File Structure
* `baseline`: code for the baseline pre-traininng, and fine-tuning on assembly code
    * Data Pre-processing
        * `prepare_lm_dataset.py`: prepare and tokenize the data for *language modeling training (next token prediction)*
        * `prepare_contra_dataset.py`: prepare and tokenize the data for *contrastive learning training*
        * `prepare_bcr_dataset.py`: prepare and tokenize the data for *fine-tuning* binary code recovery
        * `prepare_sim_dataset.py`: prepare and tokenize the data for *fine-tuning* binary code similarity detection

    * Pre-training and Fine-tuning
        * `dataset.py`: Dataset class, used during pre-training and fine-tuning
        * `train_lm.py`: pre-train the model with language modeling (using HuggingFace Trainer)
        * `train_lm_contra.py`: pre-train the model together with language modeling and contrastive learning (using DeepSpeed)
        * `finetune_bcr.py`: fine-tune trained model for binary code recovery task
        * `finetune_sim.py`: fine-tune trained model for binary code similarity detection task
    * Inference and Evaluation
        * `inference_bcr.py`: infer the fine-tuned model on the HumanEval-Decompile benchmark, for the binary code recovery task.
        * `inference_sim.py`: infer the fine-tuned model on the CodeArt's evaluation set, for the binary code similarity detection
        * `evaluate.py`: evaluate binary code recovery task
* `nova`: code for Nova pre-training and fine-tuning
    * Nova Model Architecture
        * `modeling_nova.py`: the code for Nova's model class, extended from Llama
        * `generation_utils.py`: the code for Nova's specialized inference, used by Nova's model class
    * The rest code for data pre-processing, pre-training and fine-tuning are structured in similar ways as the baseline approach.
* `benchmark`: test data and results for HumanEval-Decompile
* `data`: folder to save the processed data
* `data-tokenize`: folder to save the tokenized dataset


## Train

### Step 1: Obtain Assembly Code for Training Data

Training use data from two sources: *Anghabench* and *The-Stack*

For *Anghabench* dataset, download the dataset from https://github.com/brenocfg/AnghaBench, and run the following command to prepare `n` data instances (use the whole Anghabench if `n` is not provided, 1 million in total):

```bash
cd data
python compile_anghabench.py --root {patch_to_downloaded_AnghaBench} --output anghabench/anghabench.jsonl --n 100
```

The data will be saved at `data/anghabench/anghabench.jsonl`

For *The-Stack* dataset, run the following command to download `n` C files from `bigcode/the-stack-dedup`, and compile the files to obtain Assembly functions (the-stack is huge, the paper only downloads 50K files due to computation resource limits):

```bash
cd data
python compile_the_stack.py --output the-stack/the-stack.jsonl -n 200
```

The data will be saved at `data/the-stack/the-stack.jsonl`

### Step 2: Normalize Assembly Code

Normalize the data (according to description in the paper), by running the following command:

```bash
cd data
python normalize.py --dataset {anghabench|the-stack|codeart|binarycorp}
```

* binary code recovery: for pre-training and model and fine-tuning for binary code recovery, you need to normalize the anghabench and the-stack.
* binary code similarity detection: for fine-tuning and testing the model for bianry code similarity detection, you need to normalize the bianrycorp and codeart. You first need to download the [binarycorp dataset](https://zenodo.org/records/18072785?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjEwMGQ0NzEyLTU4NmUtNDI4MS1iNjI5LTRhODA3ZTQzNzRiMiIsImRhdGEiOnt9LCJyYW5kb20iOiIyMTM1YzU3MDRkMGFhZjE5MTgwNzYxMmYyZmM3YTBkOCJ9.WUvC4Ecwa_tM2Eio9ift8AuJ1PgnA2bJ9HXgxHstKEPfvO3fdMb5OgN_Vry2HfECtpe82PqZLNKDj1HkqSJM8g), and set the folder path in the script. You can skip this part if you only want to use the model for binary code recovery.

### Step 3: Tokenize Data

Before training the model, the data need to be tokenized and save to HuggingFace Dataset format:

* tokenize the pre-training data
```bash
# enter baseline folder for training the baseline model, or enter nova folder for training the nova model
cd nova
python prepare_lm_dataset.py        # build the dataset for language modeling
python prepare_contra_dataset.py    # build the dataset for contrastive learning
```

* tokenize the fine-tuning data
```bash
cd nova
python prepare_bcr_dataset.py       # build the dataset for binary code recovery fine-tuning
python prepare_sim_dataset.py       # build the dataset or similarity detection fine-tuning
```

### Step 4: Pre-Training

The pre-training contains two steps

1. First train the model with the language modeling objective, by running the following command:
```bash
# this uses Transformers library's Trainer (run with 4 GPUs in this example)
torchrun --nproc-per-node=4 train_lm.py
```

You will need to set the `model_save_folder` path in the code.

2. Then start from the trained model from the above step, further train it together with language modeling and contrastive learning, by running the following command:
```bash
# this uses DeepSpeed (run with 4 GPUs in this example)
deepspeed --num_gpus=4 train_lm_contra.py
```

You will need to set the `model_load_folder` and `model_save_folder` path in the code.

### Step 5: Fine-Tuning for Downstream Tasks

* binary code recovery:
```bash
# this uses Transformers library's Trainer (run with 4 GPUs in this example)
torchrun --nproc-per-node=4 finetune_bcr.py
```

You will need to set the `model_load_folder` and `model_save_folder` path in the code.

* binary code similarity detection:
```bash
# this uses DeepSpeed (run with 4 GPUs in this example)
deepspeed --num_gpus=4 finetune_sim.py
```

You will need to set the `model_load_folder` and `model_save_folder` path in the code.

## Evaluation

Before inference, you may need to convert the DeepSpeed checkpoints to HuggingFace's style: use `zero_to_hf_ckpt.py` if the DeepSpeed config uses Zero1 or Zero2, use DeepSpeed's official script if it uses Zero3 optimization.

* To evaluate on HumanEval-Decompile, which tests the binary code recovery ability:
```bash
python inference_bcr.py
python evaluate.py
```

* To evaluate on CodeArt's test set, which tests the binary code similarity detection ability:
```bash
python inference_sim.py
```
The evaluation uses the code in [CodeArt's repository](https://github.com/ziansu/codeart)


## Citation
```
@inproceedings{jiang2025nova,
    title={Nova: Generative Language Models for Assembly Code with Hierarchical Attention and Contrastive Learning},
    author={Nan Jiang and Chengxiao Wang and Kevin Liu and Xiangzhe Xu and Lin Tan and Xiangyu Zhang and Petr Babkin},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```