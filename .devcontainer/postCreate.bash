#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "Using proxy:"
env | grep -iE '^(http|https|no)_proxy=' || true

echo "Updating apt + installing packages..."
sudo -E apt-get update
sudo -E apt-get install -y --no-install-recommends \
  build-essential \
  python3-dev \
  ca-certificates \
  curl \
  gnupg
sudo -E rm -rf /var/lib/apt/lists/*

# Install uv (official installer)
# This puts uv in ~/.local/bin by default.
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH for this script invocation
export PATH="$HOME/.local/bin:$PATH"

echo "Running uv sync..."
cd /workspaces/idioms
uv sync
