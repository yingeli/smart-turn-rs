FROM huggingface/transformers-all-latest-torch-nightly-gpu:latest

# Install baseline packages
RUN apt-get update && \
    apt-get install --yes --no-install-recommends --no-install-suggests \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Google Gemini CLI
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash - \
    && apt-get install --yes --no-install-recommends --no-install-suggests \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @google/gemini-cli

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
