import os
import json

import re
from datasets import load_dataset
from detoxify import Detoxify
from tqdm import tqdm
import kagglehub
import pdb


# Toggle detoxification step on/off
detoxify = False  # Set to True to enable detoxifying

# Step 1: Load Hugging Face Dataset
print("Loading Hugging Face dataset...")
hf_dataset = load_dataset("ajibawa-2023/Children-Stories-Collection", split="train")
hf_texts = [item["text"] for item in hf_dataset if "text" in item and item["text"].strip()]
print(f"HuggingFace stories count: {len(hf_texts)}")
# print("Sample HuggingFace story:", hf_texts[0] if hf_texts else "No data")


# Step 2: Download Kaggle Dataset via kagglehub
print("Downloading Kaggle dataset...")
kaggle_path = kagglehub.dataset_download("edenbd/children-stories-text-corpus")
print("Path to dataset files:", kaggle_path)
# pdb.set_trace()  # Uncomment if debugging is needed

# Step 3: Read .txt files and split into stories
kaggle_texts = []
for root, _, files in os.walk(kaggle_path):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Split on 2 or more newlines (paragraph breaks)
                chunks = re.split(r'\n{2,}', content)
                # Group every two chunks as title + story text
                i = 0
                while i < len(chunks) - 1:
                    title = chunks[i].strip()
                    text = chunks[i+1].strip()
                    full_story = title + "\n\n" + text
                    if full_story:
                        kaggle_texts.append(full_story)
                    i += 2

print(f"Kaggle stories count: {len(kaggle_texts)}")
# print("Sample Kaggle story:", kaggle_texts[0] if kaggle_texts else "No data")

# Step 4: Combine all texts
all_texts = hf_texts + kaggle_texts
print(f"Total combined stories: {len(all_texts)}")

# Step 5: Detoxify and filter (optional)
clean_texts = []

if detoxify:
    model = Detoxify('original')
    print("Detoxifying and cleaning texts...")
    for text in tqdm(all_texts):
        if not text.strip():
            continue
        score = model.predict(text)
        if score["toxicity"] < 0.2:
            clean_texts.append(text.strip())
else:
    print("Detoxification disabled, copying all texts...")
    clean_texts = [text.strip() for text in all_texts if text.strip()]

# Step 6: Save as JSONL for training
output_file = "clean_kids_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for line in clean_texts:
        json.dump({"text": line}, f)
        f.write("\n")

print(f"\n✅ Cleaned dataset saved to: {output_file}")
print(f"Total clean stories: {len(clean_texts)}")
