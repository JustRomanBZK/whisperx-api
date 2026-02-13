# WhisperX API

HTTP-обёртка над `whisperx` CLI с API Key авторизацией и GPU lock.

## Quick Start (Ubuntu + NVIDIA GPU)

```bash
git clone https://github.com/JustRomanBZK/whisperx-api.git
cd whisperx-api
chmod +x setup.sh
./setup.sh
```

Скрипт автоматически установит: NVIDIA драйвер, Docker, NVIDIA Container Toolkit, проверит GPU, создаст `.env` и запустит сервис.

## Ручной деплой

```bash
git clone https://github.com/JustRomanBZK/whisperx-api.git
cd whisperx-api
cp .env.example .env     # заполнить API_KEY и HF_TOKEN
docker compose up -d
```

Документация: `http://localhost:8000/docs`

## Авторизация

Все запросы (кроме `/health`) требуют заголовок `X-API-Key`:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/health
```

Без ключа или с неверным ключом → `401 Unauthorized`.

## API

- `GET /health` → `ok` (без авторизации, для healthcheck)
- `POST /transcribe` → результат (`.srt`/`.vtt`/`.txt`/`.json`)

### POST /transcribe

`multipart/form-data`:

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | file | **Обязательно.** Аудио (wav/mp3/m4a) |
| `model` | string | Модель Whisper (default: tiny) |
| `language` | string | Код языка (uk/ru/en). Если не задан — автоопределение |
| `output_format` | string | srt/vtt/txt/json (default: srt) |
| `diarize` | bool | Определение спикеров (требует HF token) |
| `batch_size` | int | Размер батча (default: 1) |
| `device` | string | cuda/cpu (default: cuda) |
| `compute_type` | string | float16/float32/int8 (default: float16) |
| `config` | string | JSON-объект с параметрами (альтернатива отдельным полям) |
| `hf_token` | string | HuggingFace token |

HF Token также принимается через: `X-HF-Token` header, `Authorization: Bearer ...`, env `HF_TOKEN`.

### Примеры

```bash
# curl
curl -X POST "http://localhost:8000/transcribe" \
  -H "X-API-Key: your-secret-key" \
  -H "X-HF-Token: hf_***" \
  -F "file=@audio.wav" \
  -F "model=tiny" \
  -F "output_format=srt" \
  -F "diarize=true"
```

```python
# Python
import requests

resp = requests.post(
    "http://localhost:8000/transcribe",
    headers={"X-API-Key": "your-secret-key", "X-HF-Token": "hf_***"},
    files={"file": open("audio.wav", "rb")},
    data={"model": "tiny", "output_format": "srt"},
    timeout=600,
)
print(resp.text)
```

## Переменные окружения

| Переменная | Default | Описание |
|-----------|---------|----------|
| `API_KEY` | — | **Обязательно.** Ключ для авторизации |
| `HF_TOKEN` | — | HuggingFace token для diarization |
| `WHISPERX_MODEL` | tiny | Модель Whisper |
| `WHISPERX_DEVICE` | cuda | Устройство |
| `WHISPERX_COMPUTE_TYPE` | float16 | Тип вычислений |
| `WHISPERX_BATCH_SIZE` | 1 | Размер батча |
| `WHISPERX_LANGUAGE` | — | Язык (auto-detect если не задан) |
| `WHISPERX_OUTPUT_FORMAT` | srt | Формат вывода |

## Важно

- Сервис обрабатывает запросы строго по одному (GPU lock)
- Без `API_KEY` в env сервис не запустится
- Модели кешируются в Docker volume `whisperx-cache`
