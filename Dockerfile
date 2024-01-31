ARG CUDA=12.0.0
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

# Install Miniconda and set up the PATH
ENV PATH="/opt/conda/bin:$PATH"
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Create a Conda environment with Python 3.8
RUN conda create -n timed_design python=3.8 -y \
    && echo "source activate timed_design" > ~/.bashrc

# Activate the Conda environment
ENV PATH /opt/conda/envs/timed_design/bin:$PATH

# Install packages in the Conda environment
RUN conda install -n timed_design -c conda-forge \
      cudatoolkit \
      cudnn \
      cupti \
      pip -y \
    && pip install --upgrade pip --no-cache-dir \
    && conda clean --all --force-pkgs-dirs --yes

# Clone the repository
RUN git clone https://github.com/wells-wood-research/timed-design.git /app/timed-design

# Change the working directory
WORKDIR /app/timed-design

# Clone the repository
RUN git checkout hide-streamlit-warnings

RUN source ~/.bashrc \
    && pip install -r requirements.txt \
    && pip install .

# Create a data directory
RUN mkdir -p /app/data

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.maxUploadSize", "2", "--server.baseUrlPath", "timed", "--", "--path_to_models", "/scratch/timed_dataset/models/", "--path_to_pdb", "/scratch/datasets/biounit/", "--path_to_data", "/app/data/", "--workers", "12", "--client.showErrorDetails=false"]