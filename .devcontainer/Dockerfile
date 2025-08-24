FROM nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1

# Install baseline packages
RUN apt-get update && \
    apt-get install --yes --no-install-recommends --no-install-suggests \
    sudo \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Google Gemini CLI
RUN npm install -g @google/gemini-cli

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y