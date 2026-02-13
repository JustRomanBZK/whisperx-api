FROM ghcr.io/jim60105/whisperx:no_model

USER root
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY patches/ /app/patches/

RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip install --no-cache-dir --upgrade whisperx \
    && python -m pip install --no-cache-dir --upgrade "pyannote-audio>=4.0.3" \
    && python /app/patches/patch_pyannote4.py

COPY app/ /app/app/
COPY main.py /app/main.py

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["python", "main.py"]
