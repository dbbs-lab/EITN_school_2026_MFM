ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} ubuntu:24.04

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
        NEURON==8.2.6 mpi4py

# Build NEST 3.8 once to a shared prefix
RUN python3 -m venv /home/${USERNAME}/mfm && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir mpi4py numpy cython setuptools

RUN git clone --branch v3.8 --depth 1 https://github.com/nest/nest-simulator.git /tmp/nest-src && \
    mkdir /tmp/nest-build && cd /tmp/nest-build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/home/${USERNAME}/nest \
          -Dwith-python=ON \
          -DPYTHON_EXECUTABLE=$(which python3) \
          -Dwith-mpi=ON \
          -Dwith-gsl=ON \
          -Dwith-boost=ON \
          /tmp/nest-src && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/nest-src /tmp/nest-build

# bsb venv — BSB next branch + NEURON + NEST + Jupyter + cerebellar-models
RUN python3 -m venv /home/${USERNAME}/bsb && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir mpi4py

RUN git clone --branch next --depth 1 https://github.com/dbbs-lab/bsb.git /tmp/bsb && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir maturin && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir \
        /tmp/bsb/packages/bsb-native \
        /tmp/bsb/packages/bsb-otel \
        "/tmp/bsb/packages/bsb-core[parallel]" \
        /tmp/bsb/packages/bsb-hdf5 \
        /tmp/bsb/packages/bsb-json \
        /tmp/bsb/packages/bsb-yaml \
        /tmp/bsb/packages/bsb-nest \
        "/tmp/bsb/packages/bsb[parallel,nest]" && \
    rm -rf /tmp/bsb

RUN /home/${USERNAME}/bsb/bin/pip install --no-cache-dir jupyterlab && \
    /home/${USERNAME}/bsb/bin/pip install --no-cache-dir cerebellar-models

# mfm venv — BSB + NEST + TVB + Jupyter + cerebellar-models
RUN /home/${USERNAME}/mfm/bin/pip install --no-cache-dir jupyterlab && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir "bsb[parallel,nest]"

RUN git clone --branch feature/auto_mfm https://github.com/dbbs-lab/cerebellar-models.git /tmp/cerebellar-models && \
    /home/${USERNAME}/mfm/bin/pip install --no-cache-dir "/tmp/cerebellar-models[nest]" && \
    rm -rf /tmp/cerebellar-models

# Symlink shared NEST into both venvs
RUN NEST_PYDIR=$(find /home/${USERNAME}/nest -type d -name "nest" -path "*/lib/*" | head -1) && \
    NEST_SP=$(dirname "$NEST_PYDIR") && \
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    for VENV in bsb mfm; do \
        VENV_SP=/home/${USERNAME}/${VENV}/lib/python${PYVER}/site-packages && \
        ln -s ${NEST_PYDIR} ${VENV_SP}/nest && \
        for d in ${NEST_SP}/*.dist-info ${NEST_SP}/*.egg-info; do \
            [ -e "$d" ] && ln -s "$d" ${VENV_SP}/; \
        done; \
    done \
    || true

ENV PATH="/home/${USERNAME}/nrn/bin:$PATH"
ENV VIRTUAL_ENV="/home/${USERNAME}/nrn"

RUN echo '\n# EITN virtual environments\nalias use-nrn="source ~/nrn/bin/activate"\nalias use-bsb="source ~/bsb/bin/activate"\nalias use-mfm="source ~/mfm/bin/activate"' \
    >> /home/${USERNAME}/.bashrc

CMD ["/bin/bash"]
