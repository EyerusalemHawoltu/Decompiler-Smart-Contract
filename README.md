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
* `baseline`: code for the baseline approach, standard fine-tuning on assembly code
    * `dataset.py`: Dataset class, used during training
    * `evaluate.py`: evaluate binary code recovery task
    * `finetune_bcr.py`: fine-tune trained Nova model for binary code recovery task
    * `finetune_sim.py`: fine-tune trained Nova model for binary code similarity detection task
    * `inference_bcr.py`: infer the fine-tuned model on the HumanEval-Decompile benchmark
    * `inference_sim.py`: infer the fine-tuned model on the CodeArt's evaluation set
    * `prepare_bcr_dataset.py`: prepare and tokenize the data for fine-tuning binary code recovery
    * `prepare_sim_dataset.py`: prepare and tokenize the data for fine-tuning binary code similarity
    * `prepare_contra_dataset.py`: prepare and tokenize the data for contrastive learning training
    * `prepare_lm_dataset.py`: prepare and tokenize the data for language modeling training (next token prediction)
    * `train_lm_contra.py`: train the model together with language modeling and contrastive learning (using DeepSpeed)
    * `train_lm.py`: train the model with language modeling (using HuggingFace Trainer)
* `benchmark`: test data and results for HumanEval-Decompile
* `data`: folder to save the processed data
* `data-tokenize`: folder to save the tokenized dataset
* `nova`: code for Nova
    * `dataset.py`: the code for the Dataset class, used during training
    * `generation_utils.py`: the code for Nova's specialized inference, used by Nova's model class
    * `modeling_nova.py`: the code for Nova's model class, extended from Llama
    * `inference_bcr.py`: the code to run Nova on the HumanEval-Decompile benchmark
    * `evaluate.py`: evaluate binary code recovery task


## Train

### Step 1: Obtain Assembly Code for Training Data

Training use data from two sources: Anghabench and The-Stack

For Anghabench dataset, download the dataset from https://github.com/brenocfg/AnghaBench, and run the following command to prepare `n` data instances:

```bash
cd data
python compile_anghabench.py --root <patch to AnghaBench> --output anghabench/anghabench.jsonl --n 100
```

The data will be saved at `data/anghabench/anghabench.jsonl`

For The-Stack dataset, run the following command:

```bash
cd data
python compile_the_stack.py
```

The data will be saved at `data/the-stack/the-stack.jsonl`

### Step 2: Normalize Assembly Code

Normalize the data (according to description in the paper), by running the following command:

```bash
cd data
python normalize.py
```

### Step 3: Tokenize Data

Before training the model, the data need to be tokenized and save to HuggingFace Dataset format:

```bash
cd baseline
python prepare_lm_dataset.py
python prepare_contra_dataset.py
```

### Step 4: Training

First train the model with the language modeling objective, by running the following command:

```bash
cd baseline
python train_lm.py
```

You will need to set the `model_load_folder` and `model_save_folder` path in the code.

Then start from the trained model from the above step, further train it together with language modeling and contrastive learning, by running the following command:

```bash
cd baseline
python train_lm_contra.py
```

For training the Nova model, running the corresponding code in the `nova` folder.

## Usage

```bash
cd nova
python inference_bcr.py
python evaluate.py
```

## Citation
```
@misc{jiang2024nova,
      title={Nova: Generative Language Models for Assembly Code with Hierarchical Attention and Contrastive Learning}, 
      author={Nan Jiang and Chengxiao Wang and Kevin Liu and Xiangzhe Xu and Lin Tan and Xiangyu Zhang},
      year={2024},
      eprint={2311.13721},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2311.13721}, 
}
```