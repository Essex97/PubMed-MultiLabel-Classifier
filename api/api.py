import logging
import uvicorn

from fastapi import FastAPI, APIRouter, Response, status
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)


logging.basicConfig(level=logging.INFO)

app = FastAPI()
router = APIRouter()

model_name_or_path = "Stratos97/biobert-base-cased-PubMed-Mesh"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=None)


@router.get("/health")
async def get_health():
    return {"message": "OK"}


@router.get("/")
async def home():
    return {"message": "Machine Learning service"}


@router.post("/v1/predict")
async def data(input_data: dict, response: Response):
    try:

        # Get the input article (text)
        article = input_data["text"]

        # Classify the given article
        scores = pipe(article)[0]

        # Construct the response
        results = {
            f"article": article,
            "scores": {r['label']: r['score'] for r in scores}
        }
    except Exception as e:
        logging.error("Something went wrong ", e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"STATUS": "Error", "RESPONSE": {}}

    return {"STATUS": "OK", "RESPONSE": results}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("api:app", reload=True, port=6000, host="0.0.0.0")
