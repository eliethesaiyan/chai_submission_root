FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models/entity_extractor ./models/entity_extractor
COPY run.sh README.md ./
RUN test -s models/entity_extractor/model.safetensors \
    && chmod -R a+rX models/entity_extractor \
    && useradd --create-home --uid 10001 clinical \
    && mkdir -p outputs && chown clinical:clinical outputs
USER clinical
RUN python -c "from safetensors import safe_open; f = safe_open('models/entity_extractor/model.safetensors', framework='pt'); assert list(f.keys()), 'Model weights are empty'"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
