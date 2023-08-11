# syntax=docker/dockerfile:1.4
# dockerlab template: workspace_pytorch_1_13
# https://github.com/hughplay/dockerlab

FROM deepbase/dockerlab:workspace_pytorch_1_13


# ==================================================
# Install python packages
# ==================================================


# Setup TUNA mirror (optional)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN cat <<EOT >> ~/.condarc
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOT


# By default, install packages from `requirements.txt` with pip.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Another way is installing packages from a `env.yaml` with conda.
# COPY env.yaml /tmp/env.yaml
# RUN conda env create -f /tmp/env.yaml && rm -f /tmp/env.yaml


# ==================================================
# Post-installation steps
#
# Create a user that has the same UID and GID as the host user. This will
# prevent many privileges issues.
# ==================================================


# TUNA mirror for apt
# RUN sed -i -e 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Add a user with the same UID and GID as the host user, to prevent privilege issues.
ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker
RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi


# Copy installed configuration files from root to user
COPY misc/init_workspace /usr/local/bin
RUN chmod +x /usr/local/bin/init_workspace
RUN /usr/local/bin/init_workspace --user ${USER_NAME} --home /home/${USER_NAME}


# backup $HOME for reverse mounting $HOME
RUN rsync -a /home/${USER_NAME}/ /${USER_NAME}_home_bak


# Switch to the created user
USER ${USER_NAME}


# Set working directory to /project
WORKDIR "/project"
