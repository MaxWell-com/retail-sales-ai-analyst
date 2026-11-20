FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
COPY eval eval
# Bake the synthetic warehouse into the image: the container is self-contained.
RUN python scripts/make_synthetic.py && python scripts/build_db.py && rm -rf data/parquet
ENV PORT=8765 OLLAMA_URL=http://ollama:11434 VT_MODEL=qwen2.5:7b
EXPOSE 8765
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')"
CMD ["python", "-m", "app.server"]
