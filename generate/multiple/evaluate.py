# Need GPU for this code + install pip evaluate
from datasets import load_dataset, Dataset
import os
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
import torch
import evaluate

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load reference dataset once
reference_dataset = load_dataset("knkarthick/dialogsum", split='test')
reference_df = reference_dataset.to_pandas()

# Function for batch summarization
def batch_summarize(texts, batch_size=8):
    generated_summaries = []  # To store the predictions (generated summaries)
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(inputs['input_ids'].to(device), max_length=256, num_beams=4, early_stopping=True)
        decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        generated_summaries.extend(decoded_summaries)
    return generated_summaries

# Path to disfluency outputs
BASE_DIR = '/content/drive/MyDrive/disfluency_outputs'
directories = os.listdir(BASE_DIR)

# Metrics
rouge_metric = evaluate.load("rouge")
comet_metric = evaluate.load('comet')
meteor_metric = evaluate.load('meteor')

for i, directory in enumerate(directories):
    csv_file = os.listdir(os.path.join(BASE_DIR, directory))

    # Load CSV file as dataset
    dataset = load_dataset('csv', data_files={'train': os.path.join(BASE_DIR, directory, csv_file[0])})

    df = pd.read_csv(os.path.join(BASE_DIR, directory, csv_file[0]))

    # Regroup reconstructed dialogue
    grouped = df.groupby('id', sort=False).apply(
        lambda group: "\n".join(f"{row['speaker']}: {row['sentence']}" for _, row in group.iterrows())
    ).reset_index(name='reconstructed_dialogue')
    dataset = Dataset.from_pandas(grouped)

    if len(dataset) == 1500:
      continue
    print(f"Processing directory... {directory}...")

    # Match reference summaries by id
    matched_reference_df = reference_df[reference_df['id'].isin(grouped['id'])]
    merged_df = pd.merge(grouped, matched_reference_df[['id', 'summary']], on='id', how='inner')

    # Sort by ID to ensure alignment
    merged_df = merged_df.sort_values(by='id').reset_index(drop=True)
    dataset = Dataset.from_pandas(merged_df[['id', 'reconstructed_dialogue']])
    reference_summaries = merged_df['summary'].tolist()

    # Perform batch summarization
    predictions = batch_summarize(dataset['reconstructed_dialogue'])

    # Evaluate with ROUGE
    result = rouge_metric.compute(predictions=predictions, references=reference_summaries)
    print(f"ROUGE-L: {round(100*result['rougeL'], 2)}")
    print(f"ROUGE-1: {round(100*result['rouge1'], 2)}")
    print(f"ROUGE-2: {round(100*result['rouge2'], 2)}")

    # Evaluate with COMET
    comet_score = comet_metric.compute(
        predictions=predictions,
        references=reference_summaries,
        sources=dataset['reconstructed_dialogue']
    )
    print(f"COMET: {round(100*comet_score['mean_score'], 2)}")

    # Evaluate with METEOR
    meteor_result = meteor_metric.compute(predictions=predictions, references=reference_summaries)
    print(f"METEOR: {round(100*meteor_result['meteor'], 2)}")