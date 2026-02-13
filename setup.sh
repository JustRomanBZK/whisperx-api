#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "  WhisperX API — Auto Setup"
echo "========================================="

# --- 1. NVIDIA Driver ---
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] NVIDIA driver already installed"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "[*] Installing NVIDIA driver..."
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-550
    echo "[!] Reboot required. Run this script again after reboot."
    exit 0
fi

# --- 2. Docker ---
if command -v docker &>/dev/null; then
    echo "[OK] Docker already installed"
else
    echo "[*] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi

# Ensure current user is in docker group
if ! groups "$USER" | grep -q '\bdocker\b'; then
    echo "[*] Adding $USER to docker group..."
    sudo usermod -aG docker "$USER"
    echo "[*] Re-executing script with docker group..."
    exec sg docker "$0"
fi

# --- 3. NVIDIA Container Toolkit ---
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    echo "[OK] NVIDIA Container Toolkit already working"
else
    echo "[*] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    sleep 2

    echo "[*] Verifying GPU access in Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; then
        echo "[OK] Docker can access GPU"
    else
        echo "[!] Docker cannot access GPU. Check NVIDIA drivers and Container Toolkit."
        exit 1
    fi
fi

# --- 4. Setup .env ---
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "========================================="
    echo "  EDIT .env BEFORE CONTINUING"
    echo "  Set API_KEY and HF_TOKEN"
    echo "========================================="
    echo ""
    echo "Run: nano .env"
    echo "Then: docker compose up -d"
    exit 0
else
    echo "[OK] .env exists"
fi

# --- 5. Build & Start ---
echo "[*] Building and starting WhisperX API..."
docker compose up -d --build

echo ""
echo "========================================="
echo "  WhisperX API is running!"
echo "  http://localhost:8000/docs"
echo "  Health: curl http://localhost:8000/health"
echo "========================================="
