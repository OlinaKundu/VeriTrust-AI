#!/bin/bash
set -e

echo "=================================================="
echo "      STARTING VERITRUST AI ON HUGGING FACE SPACES"
echo "=================================================="

# Ensure temp directory exists
mkdir -p /app/backend/temp
mkdir -p /app/demo_assets

# 1. Start FastAPI Backend (Port 8000)
echo "[1/3] Starting FastAPI Backend on 127.0.0.1:8000..."
cd /app/backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 2. Start Next.js Frontend (Port 3000)
echo "[2/3] Starting Next.js Production Server on 127.0.0.1:3000..."
cd /app/frontend
npm run start -- -p 3000 &
FRONTEND_PID=$!

# 3. Start Nginx Gateway (Port 7860)
echo "[3/3] Launching Nginx Gateway on 0.0.0.0:7860..."
nginx -g "daemon off;"
