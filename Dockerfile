# Use Python 3.11 as the base image
FROM python:3.11-bullseye

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including texlive-full
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==24.2

# Install Python packages
RUN pip install --no-cache-dir \
    anthropic==0.34.0 \
    aider-chat==0.50.1 \
    backoff==2.2.1 \
    openai==1.40.6 \
    matplotlib==3.9.2 \
    pypdf==4.3.1 \
    pymupdf4llm==0.0.10 \
    torch==2.4.0 \
    numpy==1.26.4 \
    transformers==4.44.0 \
    datasets==2.21.0 \
    tiktoken==0.7.0 \
    wandb==0.17.7 \
    tqdm==4.66.5 \
    scikit-learn==1.5.1 \
    einops==0.8.0

# Set the default command to an empty array
CMD ["/bin/bash"]