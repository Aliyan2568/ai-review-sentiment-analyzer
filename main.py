from transformers import pipeline

analyzer = pipeline(
    "sentiment-analysis",
    model="./sentiment_model",
    tokenizer="./sentiment_model"
)

positive_count = 0
negative_count = 0

while True:
    review = input("\nEnter a product review (or 'quit' to exit): ")

    if review.lower() == "quit":
        break

    if not review.strip():
        print("Please enter a review.")
        continue

    result = analyzer(review)[0]

    sentiment = result["label"]
    confidence = result["score"] * 100

    if sentiment == "POSITIVE":
        positive_count += 1
    else:
        negative_count += 1

    print(f"\nSentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nReview Statistics")
    print(f"Positive Reviews: {positive_count}")
    print(f"Negative Reviews: {negative_count}")
