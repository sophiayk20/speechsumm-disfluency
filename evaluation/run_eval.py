from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
import torch
import evaluate

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

MODE='OTAS'
dataset = load_dataset("sophiayk20/replacement-one-speaker", split=MODE)

# Function for batch summarization
def batch_summarize(texts, batch_size=8):
    generated_summaries = []  # To store the predictions (generated summaries)
    # Tokenize texts in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        # Generate summaries
        summary_ids = model.generate(inputs['input_ids'].to(device), max_length=256, num_beams=4, early_stopping=True)
        decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        generated_summaries.extend(decoded_summaries)
    return generated_summaries

# Perform batch summarization
predictions = batch_summarize(dataset['disfluent_dialogue']) 

# Evaluate using ROUGE-L
rouge_metric = evaluate.load("rouge")

# Compute ROUGE-L
result = rouge_metric.compute(predictions=predictions, references=dataset['summary'])
print(f"ROUGE-L: {round(100*result['rougeL'], 2)}")
print(f"ROUGE-1: {round(100*result['rouge1'], 2)}")
print(f"ROUGE-2: {round(100*result['rouge2'], 2)}")

comet_metric = evaluate.load('comet')
comet_score = comet_metric.compute(predictions=predictions, references=dataset['summary'], sources=dataset['disfluent_dialogue'])
print(comet_score)

meteor = evaluate.load('meteor')
results = meteor.compute(predictions=predictions, references=dataset['summary'])
print(f"METEOR: {round(100*results['meteor'], 2)}")