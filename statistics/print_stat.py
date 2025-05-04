# Generate descriptive statistics of the 'DialogSum' dataset
from datasets import load_dataset
from statistics import mean
from tqdm import tqdm
import re

def split_turns(dialogue):
    return dialogue.strip().split('\n')

def word_count(text):
    return len(re.findall(r'\w+', text))

def extract_utterance(turn):
    return turn.split(':', 1)[-1].strip() if ':' in turn else turn

dataset = load_dataset("knkarthick/dialogsum", split='test')

# Compute statistics
num_dialogues = len(dataset)
turns_per_dialogue = []
words_per_dialogue = []
utterances_per_turn = []
words_per_turn = []
words_per_summary = []

for entry in tqdm(dataset):
    turns = split_turns(entry['dialogue'])
    turns_per_dialogue.append(len(turns))
    
    utterances = [extract_utterance(turn) for turn in turns]
    words_in_dialogue = sum(word_count(utt) for utt in utterances)
    words_per_dialogue.append(words_in_dialogue)
    
    words_per_turn.extend([word_count(utt) for utt in utterances])
    utterances_per_turn.extend([len(re.split(r'[.?!]', utt)) for utt in utterances if utt.strip()])
    
    summary_words = word_count(entry['summary'])
    words_per_summary.append(summary_words)

# Final statistics
print("Number of dialogues:", num_dialogues)
print("Average number of turns per dialogue:", round(mean(turns_per_dialogue), 2))
print("Average number of words per dialogue:", round(mean(words_per_dialogue), 2))
print("Average number of utterances per turn:", round(mean(utterances_per_turn), 2))
print("Average number of words per turn:", round(mean(words_per_turn), 2))
print("Average number of words per summary:", round(mean(words_per_summary), 2))