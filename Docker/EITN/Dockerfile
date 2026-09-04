FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    vim \
    nano \
    sudo \
    openssh-client \
    ca-certificates \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nodejs \
    npm \
    jq \
    htop \
    net-tools \
    iputils-ping \
    dnsutils \
    unzip \
    zip \
    openmpi-bin \
    libopenmpi-dev \
    cmake \
    libgsl-dev \
    libltdl-dev \
    libncurses-dev \
    libreadline-dev \
    libboost-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN if ! getent group ${USER_GID} >/dev/null; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
    && if ! getent passwd ${USER_UID} >/dev/null; then \
         useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash; \
       else \
         existing=$(getent passwd ${USER_UID} | cut -d: -f1) \
         && usermod -l ${USERNAME} -d /home/${USERNAME} -m -s /bin/bash "$existing" \
         && groupmod -n ${USERNAME} $(getent group ${USER_GID} | cut -d: -f1) 2>/dev/null || true; \
       fi \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

RUN mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}
WORKDIR /workspace

RUN python3 -m venv /home/${USERNAME}/nrn && \
    /home/${USERNAME}/nrn/bin/pip install --no-cache-dir \
        NEURON==8.2.6 mpi4py tvb-library tvb-contrib

RUN python3 -m venv /home/${USERNAME}/bsb && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir mpi4py && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir tvb-library tvb-contrib && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir "bsb[parallel,neuron]"

RUN python3 -m venv /home/${USERNAME}/mfm && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir mpi4py numpy cython setuptools

RUN git clone --branch v3.8 --depth 1 https://github.com/nest/nest-simulator.git /tmp/nest-src && \
    mkdir /tmp/nest-build && cd /tmp/nest-build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/home/${USERNAME}/mfm \
          -Dwith-python=ON \
          -DPYTHON_EXECUTABLE=/home/${USERNAME}/mfm/bin/python3 \
          -Dwith-mpi=ON \
          -Dwith-gsl=ON \
          -Dwith-boost=ON \
          /tmp/nest-src && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/nest-src /tmp/nest-build

RUN /home/${USERNAME}/mfm/bin/pip install --no-cache-dir tvb-library tvb-contrib && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir jupyterlab && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir "bsb[parallel,nest]"

RUN git clone --branch feature/auto_mfm https://github.com/dbbs-lab/cerebellar-models.git /tmp/cerebellar-models && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir "/tmp/cerebellar-models[nest]" && \
    rm -rf /tmp/cerebellar-models

ENV PATH="/home/${USERNAME}/nrn/bin:$PATH"
ENV VIRTUAL_ENV="/home/${USERNAME}/nrn"

RUN echo '\n# EITN virtual environments\nalias use-nrn="source ~/nrn/bin/activate"\nalias use-bsb="source ~/bsb/bin/activate"\nalias use-mfm="source ~/mfm/bin/activate"' \
    >> /home/${USERNAME}/.bashrc

CMD ["/bin/bash"]
