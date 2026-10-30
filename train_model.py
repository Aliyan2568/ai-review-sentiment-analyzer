from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm.auto import tqdm

MODEL_NAME = "distilbert/distilbert-base-uncased"

print("Loading review dataset...")

train_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="train",
    streaming=True
)

test_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="test",
    streaming=True
)

train_samples = list(
    train_stream.shuffle(seed=42, buffer_size=10000).take(2000)
)

test_samples = list(
    test_stream.shuffle(seed=42, buffer_size=10000).take(500)
)

train_data = Dataset.from_list(train_samples)
test_data = Dataset.from_list(test_samples)

print("Dataset loaded!")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_reviews(batch):
    tokens = tokenizer(
        batch["content"],
        truncation=True,
        max_length=128
    )

    tokens["labels"] = batch["label"]

    return tokens


train_data = train_data.map(
    tokenize_reviews,
    batched=True,
    remove_columns=train_data.column_names
)

test_data = test_data.map(
    tokenize_reviews,
    batched=True,
    remove_columns=test_data.column_names
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)

test_loader = DataLoader(
    test_data,
    batch_size=8,
    collate_fn=collator
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

print(f"Training on: {device}")

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 1

print("Starting training...")

model.train()

for epoch in range(epochs):
    total_loss = 0

    progress = tqdm(train_loader)

    for batch in progress:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(loss=loss.item())

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1} Loss: {average_loss:.4f}")



print("Evaluating model...")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        labels = batch["labels"]

        outputs = model(**batch)

        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = sum(
    p == label for p, label in zip(all_predictions, all_labels)
) / len(all_labels)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["NEGATIVE", "POSITIVE"],
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

print("Saving trained model...")

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("Training complete!")
