import json
import re
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger

from yandex_cloud_ml_sdk import YCloudML

# sdk = YCloudML(folder_id="b1glafs4lro29a2bgh68", auth="AQVNxywslPY9Wkzlwaul9blrrWc2EROX9eg0nX4S")
sdk = YCloudML(folder_id="b1glafs4lro29a2bgh68", auth="AQVNzGGcyQD2fPYz9gQx0fYjleid1uDgkxw5I-SA")

# Initialize
app = FastAPI()
logger = None

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

def prepare_query(user_input: str) :
    return [
        {
            "role": "system",
            "text": """
            Ты должен отвечать на вопросы, заданные на русском языке, и ответы должны быть на русском языке. Предоставь информацию по заданному вопросу с тремя источниками.
Все ответы формируются и возвращаются в формате JSON со следующими ключами:
id — числовое значение, соответствующее идентификатору запроса (передаётся во входном запросе).
answer — числовое значение, содержащее правильный ответ на вопрос (если вопрос подразумевает выбор из вариантов). Если вопрос не предполагает выбор из вариантов, значение должно быть null.
reasoning — текстовое поле, содержащее объяснение или дополнительную информацию по запросу.
sources — список ссылок на источники информации (если используются). Если источники не требуются, значение должно быть пустым списком [].
            """,
        },
        {
            "role": "user",
            "text": user_input,
        }
    ]

def parse_response(response_text):
    response_text = response_text[3:len(response_text) - 3]
    response = json.loads(response_text)
    return PredictionResponse(
        id=response["id"],
        answer=response["answer"],
        reasoning=response["reasoning"],
        sources=[HttpUrl(re.findall(r'https?://\S+', source)[0]) for source in response["sources"]]
    )

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        # Здесь будет вызов вашей модели
        query = prepare_query(body.query)
        # Вызов модели
        result = sdk.models.completions("yandexgpt").configure(temperature=0.5).run(query)
        # Достаем результат
        answer = result[0]  # Замените на реальный вызов модели
        response = parse_response(answer.text)

        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
