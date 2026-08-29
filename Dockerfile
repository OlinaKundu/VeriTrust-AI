# -------------------------------------------------------------
# Stage 1: Build Next.js Frontend
# -------------------------------------------------------------
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# -------------------------------------------------------------
# Stage 2: Final Multi-Modal Python & Gateway Runtime
# -------------------------------------------------------------
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies (Nginx, FFmpeg, OpenCV libs, Node.js runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python ML & Forensics dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy Backend codebase & Demo Assets
COPY backend/ /app/backend/
COPY demo_assets/ /app/demo_assets/
COPY scripts/ /app/scripts/

# Copy built Next.js Frontend from Stage 1
COPY --from=frontend-builder /app/frontend /app/frontend

# Copy Nginx configuration and entrypoint startup script
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose Hugging Face Spaces port
EXPOSE 7860

# Launch all microservices
CMD ["/app/start.sh"]
