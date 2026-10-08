FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    poppler-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock* README.md* /app/
RUN touch /app/README.md
RUN if [ -f /app/requirements.lock ]; then \
        pip install --no-cache-dir -r /app/requirements.lock; \
    fi

COPY src /app/src
RUN pip install --no-cache-dir --no-deps /app

RUN useradd --create-home --uid 10001 lab \
    && mkdir -p /data /runs /app/configs /home/lab/.cache /tmp/industrial-lab \
    && chown -R lab:lab /data /runs /app /home/lab /tmp/industrial-lab

USER 10001

CMD ["python", "-m", "industrial_lab.cli", "serve-api"]
