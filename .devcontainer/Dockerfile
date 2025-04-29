# Use official Python 3.12 image
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Python dependencies and system dependencies, then clean up in a single RUN command
RUN apt-get update && \
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Set default user to vscode
USER vscode
