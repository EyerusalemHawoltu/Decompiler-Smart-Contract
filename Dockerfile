FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required ubuntu packages
RUN apt-get update && apt-get install -y screen vim wget git && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin:$PATH

RUN touch /root/.vimrc
RUN echo 'syntax on\nset tabstop=2\nset softtabstop=2\nset showcmd\nset showmatch\nset incsearch\nset hlsearch\nset ruler'>> /root/.vimrc

# Install packages
RUN conda install -y python=3.10
RUN conda install -y nvidia::cuda-nvcc
RUN pip install torch==2.3.1 clang==12.0.1 datasets==2.20.0 huggingface-hub==0.24.2 matplotlib==3.9.1 openai==1.37.1 transformers==4.40.2 vllm accelerate==0.33.0 deepspeed==0.14.4 


# Copy git repo
ARG HOME="/home/nova/"
WORKDIR ${HOME}
COPY . .

CMD /bin/bash