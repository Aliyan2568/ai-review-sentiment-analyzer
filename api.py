from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="AI Review Sentiment Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading sentiment model...")

analyzer = pipeline(
    "sentiment-analysis",
    model="./sentiment_model",
    tokenizer="./sentiment_model"
)

print("Model loaded successfully!")


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1, max_length=5000)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    result = analyzer(request.review)[0]

    sentiment = result["label"]
    confidence = result["score"]

    return {
        "review": request.review,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }
