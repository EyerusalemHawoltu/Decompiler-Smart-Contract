FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required ubuntu packages
RUN apt-get update && apt-get install -y screen vim wget git gcc g++ build-essential && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"

RUN touch /root/.vimrc
RUN echo 'syntax on\nset tabstop=2\nset softtabstop=2\nset showcmd\nset showmatch\nset incsearch\nset hlsearch\nset ruler'>> /root/.vimrc

# Remove default Anaconda channels to avoid ToS requirement, use only conda-forge
RUN conda config --remove channels defaults || true
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict

# Create a new conda environment with Python 3.10 to avoid conflicts with base env
RUN conda create -y --override-channels -c conda-forge -n nova python=3.10
SHELL ["conda", "run", "-n", "nova", "/bin/bash", "-c"]

# Install packages
RUN conda install -y --override-channels -c conda-forge -c nvidia cuda-nvcc
RUN pip install torch==2.3.0 clang==12.0.1 datasets==2.20.0 huggingface-hub==0.24.2 matplotlib==3.9.1 openai==1.37.1 transformers==4.40.2 vllm==0.5 accelerate==0.33.0 deepspeed==0.14.4

# Activate the environment by default
RUN echo "conda activate nova" >> ~/.bashrc
ENV PATH="/root/miniconda3/envs/nova/bin:$PATH" 


# Copy git repo
ARG HOME="/home/nova/"
WORKDIR ${HOME}
COPY . .

CMD ["/bin/bash"]