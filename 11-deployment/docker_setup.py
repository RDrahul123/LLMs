"""
Docker Setup — Generate Docker configuration for LLM services.

Generates:
  - Dockerfile for FastAPI LLM service
  - docker-compose.yml with API, Redis, Prometheus, Grafana
  - .env template
  - Health check configuration
  - Resource limits
  - Prometheus config
  - Grafana datasource config

Run: python docker_setup.py
"""

import os
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

DOCKERFILE = textwrap.dedent("""\
    # ---- Build Stage ----
    FROM python:3.11-slim AS builder

    WORKDIR /app

    # Install dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

    # ---- Runtime Stage ----
    FROM python:3.11-slim

    # Security: run as non-root
    RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

    WORKDIR /app

    # Copy installed packages
    COPY --from=builder /install /usr/local

    # Copy application code
    COPY . .

    # Set ownership
    RUN chown -R llmuser:llmuser /app

    USER llmuser

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
        CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"

    EXPOSE 8000

    # Run with uvicorn
    CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
""")


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

DOCKER_COMPOSE = textwrap.dedent("""\
    version: "3.9"

    services:
      # ---- LLM API Service ----
      api:
        build:
          context: .
          dockerfile: Dockerfile
        container_name: llm-api
        ports:
          - "${API_PORT:-8000}:8000"
        environment:
          - OPENAI_API_KEY=${OPENAI_API_KEY}
          - REDIS_URL=redis://redis:6379/0
          - ENV=${ENV:-development}
          - LOG_LEVEL=${LOG_LEVEL:-INFO}
          - DEFAULT_MODEL=${DEFAULT_MODEL:-gpt-4o}
        depends_on:
          redis:
            condition: service_healthy
        networks:
          - llm-network
        deploy:
          replicas: ${API_REPLICAS:-2}
          resources:
            limits:
              cpus: "${API_CPU_LIMIT:-1.0}"
              memory: "${API_MEMORY_LIMIT:-512M}"
            reservations:
              cpus: "${API_CPU_RESERVE:-0.25}"
              memory: "${API_MEMORY_RESERVE:-128M}"
          restart_policy:
            condition: on-failure
            delay: 5s
            max_attempts: 3
        healthcheck:
          test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 10s
        logging:
          driver: json-file
          options:
            max-size: "10m"
            max-file: "3"

      # ---- Redis (Cache + Rate Limiting) ----
      redis:
        image: redis:7-alpine
        container_name: llm-redis
        ports:
          - "${REDIS_PORT:-6379}:6379"
        volumes:
          - redis-data:/data
        networks:
          - llm-network
        deploy:
          resources:
            limits:
              cpus: "0.5"
              memory: "256M"
        healthcheck:
          test: ["CMD", "redis-cli", "ping"]
          interval: 10s
          timeout: 5s
          retries: 3
        command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

      # ---- Prometheus (Metrics) ----
      prometheus:
        image: prom/prometheus:v2.51.0
        container_name: llm-prometheus
        ports:
          - "${PROMETHEUS_PORT:-9090}:9090"
        volumes:
          - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
          - prometheus-data:/prometheus
        networks:
          - llm-network
        deploy:
          resources:
            limits:
              cpus: "0.5"
              memory: "256M"

      # ---- Grafana (Dashboards) ----
      grafana:
        image: grafana/grafana:10.4.0
        container_name: llm-grafana
        ports:
          - "${GRAFANA_PORT:-3000}:3000"
        environment:
          - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
          - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
          - GF_USERS_ALLOW_SIGN_UP=false
        volumes:
          - grafana-data:/var/lib/grafana
          - ./config/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
        networks:
          - llm-network
        deploy:
          resources:
            limits:
              cpus: "0.5"
              memory: "256M"
        depends_on:
          - prometheus

    networks:
      llm-network:
        driver: bridge

    volumes:
      redis-data:
      prometheus-data:
      grafana-data:
""")


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

REQUIREMENTS = textwrap.dedent("""\
    fastapi==0.110.0
    uvicorn[standard]==0.28.0
    openai==1.12.0
    pydantic==2.6.1
    structlog==24.1.0
    prometheus-client==0.20.0
    httpx==0.27.0
    tiktoken==0.6.0
    numpy==1.26.4
    redis==5.0.1
""")


# ---------------------------------------------------------------------------
# .env template
# ---------------------------------------------------------------------------

ENV_TEMPLATE = textwrap.dedent("""\
    # === API Configuration ===
    OPENAI_API_KEY=sk-your-key-here
    ENV=development
    LOG_LEVEL=INFO
    DEFAULT_MODEL=gpt-4o

    # === Ports ===
    API_PORT=8000
    REDIS_PORT=6379
    PROMETHEUS_PORT=9090
    GRAFANA_PORT=3000

    # === API Scaling ===
    API_REPLICAS=2
    API_CPU_LIMIT=1.0
    API_MEMORY_LIMIT=512M
    API_CPU_RESERVE=0.25
    API_MEMORY_RESERVE=128M

    # === Grafana ===
    GRAFANA_USER=admin
    GRAFANA_PASSWORD=changeme

    # === Rate Limiting ===
    RATE_LIMIT_REQUESTS=60
    RATE_LIMIT_WINDOW=60

    # === Budget ===
    DAILY_TOKEN_LIMIT=1000000
    MONTHLY_TOKEN_LIMIT=10000000
    DAILY_COST_LIMIT=10.0
    MONTHLY_COST_LIMIT=100.0
""")


# ---------------------------------------------------------------------------
# Prometheus Config
# ---------------------------------------------------------------------------

PROMETHEUS_CONFIG = textwrap.dedent("""\
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: "llm-api"
        static_configs:
          - targets: ["api:8000"]
        metrics_path: /metrics
        scrape_interval: 10s

      - job_name: "redis"
        static_configs:
          - targets: ["redis:6379"]

      - job_name: "prometheus"
        static_configs:
          - targets: ["localhost:9090"]
""")


# ---------------------------------------------------------------------------
# Grafana Datasource
# ---------------------------------------------------------------------------

GRAFANA_DATASOURCES = textwrap.dedent("""\
    apiVersion: 1

    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
        editable: false
""")


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

DOCKERIGNORE = textwrap.dedent("""\
    __pycache__
    *.pyc
    .env
    .git
    .gitignore
    venv/
    .venv/
    node_modules/
    *.log
    .pytest_cache/
    .mypy_cache/
    .ruff_cache/
    docker-compose.yml
    Dockerfile
    .dockerignore
    config/
    grafana-data/
    prometheus-data/
""")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_configs(output_dir: str = ".") -> dict[str, str]:
    """Generate all Docker configuration files."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Create config directories
    config_dir = base / "config"
    config_dir.mkdir(exist_ok=True)
    grafana_dir = config_dir / "grafana"
    grafana_dir.mkdir(exist_ok=True)

    files = {
        str(base / "Dockerfile"): DOCKERFILE,
        str(base / "docker-compose.yml"): DOCKER_COMPOSE,
        str(base / "requirements.txt"): REQUIREMENTS,
        str(base / ".env.example"): ENV_TEMPLATE,
        str(base / ".dockerignore"): DOCKERIGNORE,
        str(config_dir / "prometheus.yml"): PROMETHEUS_CONFIG,
        str(grafana_dir / "datasources.yml"): GRAFANA_DATASOURCES,
    }

    for filepath, content in files.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  Created: {filepath}")

    return files


# ---------------------------------------------------------------------------
# Dockerfile Variants
# ---------------------------------------------------------------------------


DOCKERFILE_GPU = textwrap.dedent("""\
    # GPU-enabled variant (NVIDIA)
    FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

    RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

    WORKDIR /app

    COPY requirements.txt .
    RUN pip3 install --no-cache-dir -r requirements.txt

    COPY . .

    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

    HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
        CMD python3 -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

    EXPOSE 8000
    CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
""")


DOCKERFILE_LIGHTWEIGHT = textwrap.dedent("""\
    # Lightweight variant (Alpine)
    FROM python:3.11-alpine

    WORKDIR /app

    # Install build dependencies
    RUN apk add --no-cache gcc musl-dev

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Remove build dependencies
    RUN apk del gcc musl-dev

    COPY . .

    EXPOSE 8000
    CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
""")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main():
    print("Docker Configuration Generator")
    print("=" * 40)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker_configs")
    print(f"\nGenerating configs in: {output_dir}\n")

    files = generate_configs(output_dir)

    # Also generate variant Dockerfiles
    variant_dir = Path(output_dir) / "variants"
    variant_dir.mkdir(exist_ok=True)
    (variant_dir / "Dockerfile.gpu").write_text(DOCKERFILE_GPU)
    (variant_dir / "Dockerfile.lightweight").write_text(DOCKERFILE_LIGHTWEIGHT)
    print(f"  Created: {variant_dir / 'Dockerfile.gpu'}")
    print(f"  Created: {variant_dir / 'Dockerfile.lightweight'}")

    print(f"\nGenerated {len(files) + 2} configuration files.")
    print("\nTo deploy:")
    print(f"  cd {output_dir}")
    print("  cp .env.example .env  # Edit with your API key")
    print("  docker-compose up --build -d")
    print("  docker-compose ps")
    print("  docker-compose logs -f api")


if __name__ == "__main__":
    main()
