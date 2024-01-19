ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -qy conda \
    && conda install -y -c conda-forge \
      cudatoolkit==${CUDA_VERSION} \
      cudnn \
      cupti \
      pip \
      python=3.8 \
    && pip3 install --upgrade pip --no-cache-dir \
    && conda clean --all --force-pkgs-dirs --yes

RUN git clone https://github.com/wells-wood-research/timed-design.git /app/timed-design
WORKDIR /app/timed-design
RUN pip3 install -r requirements.txt \
    && pip3 install . \
    && conda clean --all --force-pkgs-dirs --yes

# Create a data directory
RUN mkdir -p /app/data

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.maxUploadSize", "2", "--server.baseUrlPath", "timed", "--", "--path_to_models", "/scratch/timed_dataset/models/", "--path_to_pdb", "/scratch/datasets/biounit/", "--path_to_data", "/app/data/", "--workers", "12"]