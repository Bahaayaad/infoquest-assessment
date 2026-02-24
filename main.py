import logging
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from routes import ingest, chat, health, research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="InfoQuest - Expert Network Search")

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(health.router)

app.include_router(research.router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    errors = [
        {"field": e["loc"][-1], "message": e["msg"].replace("Value error, ", "")}
        for e in exc.errors()
    ]
    logging.getLogger(__name__).warning("Validation error on %s: %s", request.url.path, errors)
    return JSONResponse(status_code=422, content={"errors": errors})