from datasets import load_dataset
import os

# Create data folder (CSV thô nằm trong data/csv/)
os.makedirs("data/csv", exist_ok=True)

print("Downloading parser dataset...")

parser_data = load_dataset(
    "yashpwr/resume-ner-training-data"
)

print("Parser dataset downloaded successfully.")

# Export parser dataset to CSV
for split, dataset in parser_data.items():
    dataset.to_pandas().to_csv(
        f"data/csv/parser_{split}.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("Parser dataset exported to CSV.")


print("Downloading matching dataset...")

matching_data = load_dataset(
    "med2425/resume-job-fit-merged-v1"
)

print("Matching dataset downloaded successfully.")

# Export matching dataset to CSV
for split, dataset in matching_data.items():
    dataset.to_pandas().to_csv(
        f"data/csv/matching_{split}.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("Matching dataset exported to CSV.")

print("\nAll datasets downloaded and exported successfully!")