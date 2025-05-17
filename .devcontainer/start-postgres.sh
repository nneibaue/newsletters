#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p ./postgres-data

# Start the Postgres container
docker-compose -f ./.devcontainer/docker-compose.yml up -d
