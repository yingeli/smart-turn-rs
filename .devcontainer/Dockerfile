FROM huggingface/transformers-all-latest-gpu:latest

# Install baseline packages
RUN apt-get update && \
    apt-get install --yes --no-install-recommends --no-install-suggests \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Generate the desired locale (en_US.UTF-8)
RUN locale-gen en_US.UTF-8

# Remove the `ubuntu` user and add a user `coder` so that you're not developing as the `root` user
RUN userdel -r ubuntu && \
    useradd coder \
    --create-home \
    --shell=/bin/bash \
    --groups=docker \
    --uid=1000 \
    --user-group && \
    echo "coder ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers.d/nopasswd

USER coder

# Install Google Gemini CLI
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash - \
    && apt-get install --yes --no-install-recommends --no-install-suggests \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @google/gemini-cli

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
