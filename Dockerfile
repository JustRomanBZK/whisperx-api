FROM ghcr.io/jim60105/whisperx:no_model

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m ensurepip --upgrade && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/app/
COPY main.py /app/main.py

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["python", "main.py"]
