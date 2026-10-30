# AI Review Sentiment Analyzer

A full-stack machine learning application that uses a fine-tuned DistilBERT model to classify product reviews as positive or negative. Features an interactive dashboard for analyzing reviews, viewing confidence scores, and visualizing sentiment distributions.

## Features

- DistilBERT sentiment classification
- Real time predictions through a FastAPI backend
- Dashboard built & designed with HTML, CSS, and JavaScript
- Confidence scores for individual predictions
- Positive and negative sentiment statistics
- Sentiment distribution visualization
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## Tech Stack

- **Language:** Python
- **Machine Learning:** PyTorch
- **NLP:** Hugging Face Transformers
- **Dataset:** Amazon Polarity
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Evaluation:** Scikit-learn

## Model Performance

The model achieved **86.20% accuracy** on a 500-review test sample through one epoch.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Installation

1. Clone the repository:

git clone https://github.com/Aliyan2568/ai-review-sentiment-analyzer.git

2. Navigate to the project:

cd ai-review-sentiment-analyzer

3. Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

4. Install dependencies:

pip install -r requirements.txt

5. Train the model:

python train_model.py

This generates the local `sentiment_model/` directory required by the application.

## Running the Application

Start the FastAPI server:

python -m uvicorn api:app --reload

Open the application in your browser:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

