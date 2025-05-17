# Use official Python 3.12 image
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Python dependencies and system dependencies, then clean up in a single RUN command
RUN apt update && \
    apt install -y sqlite3 && \
    apt install -y postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspaces/newsletters

# Set default user to vscode
USER vscode
