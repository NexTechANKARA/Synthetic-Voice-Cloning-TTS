# Hugging Face Space image. The deploy workflow uploads this file as the Space's Dockerfile,
# next to vc_lib/, data/, checkpoints/ and backend/.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user . .
USER user
ENV NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
