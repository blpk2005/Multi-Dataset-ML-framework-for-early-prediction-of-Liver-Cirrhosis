#!/usr/bin/env bash
# ==============================================================================
# Automated Deployment Script for Liver Cirrhosis Prediction Platform
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Starting Liver Cirrhosis ML Service Deployment"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# 1. Check for .env file
if [ ! -f .env ]; then
    echo "[*] Creating .env from .env.example..."
    cp .env.example .env
fi

# 2. Check if model artifact exists, train if missing
if [ ! -f "artifacts/model_pipeline.joblib" ]; then
    echo "[*] Model artifact not found. Initiating model training and export..."
    python3 -m src.train
fi

# 3. Choose Deployment Mode: Docker Compose (preferred) or Local System
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "[*] Docker & Docker Compose detected. Deploying via containerized stack..."
    docker compose down --remove-orphans || true
    docker compose build --pull
    docker compose up -d

    echo "[*] Waiting for services to become healthy..."
    sleep 5
    for i in {1..12}; do
        if curl -s http://localhost:8000/health | grep -q "healthy"; then
            echo "[✓] API Service is healthy and responsive!"
            break
        fi
        echo "Waiting for healthcheck... ($i/12)"
        sleep 5
    done
else
    echo "[!] Docker Compose not found. Running in local Python environment mode..."
    if [ ! -d ".venv" ]; then
        echo "[*] Setting up virtual environment..."
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
    fi
    echo "[*] Starting Uvicorn background process..."
    nohup .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 > service.log 2>&1 &
    sleep 3
    curl -s http://localhost:8000/health
fi

echo "=========================================================="
echo " Deployment Complete!"
echo " - API Documentation: http://localhost:8000/docs"
echo " - Health Probe:       http://localhost:8000/health"
echo " - Readiness Probe:    http://localhost:8000/ready"
echo " - Metrics Endpoint:   http://localhost:8000/metrics"
echo " - Prometheus:         http://localhost:9090"
echo " - Grafana Dashboards: http://localhost:3000 (admin / admin)"
echo "=========================================================="

