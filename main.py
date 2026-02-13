from __future__ import annotations

from app import create_app
from app.config import Settings

settings = Settings.from_env()
app = create_app(settings)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )
