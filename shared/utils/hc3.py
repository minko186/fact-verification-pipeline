from datasets import load_dataset
import csv

# Load the train split of HC3
dataset = load_dataset("Hello-SimpleAI/HC3", "all")["train"].select(range(5000))

# Open a CSV file for writing
with open("hc3_train_filtered.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["question", "source", "chatgpt_answer"])

    for row in dataset:
        # chatgpt_answers is a list, but per dataset docs should only contain one string
        chatgpt_answer = row["chatgpt_answers"][0] if row["chatgpt_answers"] else ""
        writer.writerow([row["question"], row["source"], chatgpt_answer])

print("✅ Saved as hc3_train_filtered.csv")
