from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Literal
from itertools import zip_longest
from python_files.disfluency_generation import LARD  # should be in topic-controllable-summarization directory
import re
import os
import shutil
import nltk

nltk.download('wordnet')

# Instantiate LARD object
lard = LARD()

person_ids = ['#Person1#', '#Person2#']
global_turn_flag = {person_id: False for person_id in person_ids}


def process_turn(person_id, person_turns, mode, BASE_FOLDER):
    global global_turn_flag
    running = ""

    with open(f"/{BASE_FOLDER}/replacement_{mode}_stat.txt", "a") as f:
        person_turns_sentences = sent_tokenize(person_turns)

        # This turn flag is unique to this turn of person
        turn_flag = False
        disfluency_count = 0
        speaker_running = []

        for person_sentence in person_turns_sentences:
            word_rep = word_tokenize(person_sentence)
            string_rep = ' '.join(word_rep)  # Expected input by lard.create_replacements

            if mode == "ATOS" and turn_flag:
                speaker_running.append(person_sentence)
                continue
            if mode in ["OTOS", "OTAS"] and global_turn_flag[person_id]:
                speaker_running.append(person_sentence)
                continue

            disfluency = lard.create_replacements(string_rep)

            if disfluency[0]:
                tokens = disfluency[0].split()
                rejoined_sentence = ' '.join(tokens)
                rejoined_sentence = re.sub(r'\s([.,!?;])', r'\1', rejoined_sentence)
                rejoined_sentence = re.sub(r"\sn't", "n't", rejoined_sentence)
                speaker_running.append(rejoined_sentence)

                disfluency_count += 1

                if mode == "ATOS":
                    turn_flag = True
            else:
                speaker_running.append(person_sentence)

        running += f"{person_id}: " + " ".join(speaker_running) + "\n"

        if disfluency_count > 0 and mode in ['OTAS', 'OTOS']:
            global_turn_flag[person_id] = True

        f.write(f"{disfluency_count}\n")

    return running


def generate_replacement_both_speakers(
    instance_id,
    dialogue_dict,
    mode: Literal['ATAS', 'OTAS', 'ATOS', 'OTOS'] = 'ATOS'
):
    """
    Given dialogue_dict, return disfluent dialogue in string form.
    Generates at least 1 replacement disfluency in each turn.
    """
    global global_turn_flag
    global_turn_flag = {person_id: False for person_id in person_ids}

    dialogue_running = ""

    BASE_FOLDER = "/content/drive/MyDrive/ling384/replacements"
    os.makedirs(BASE_FOLDER, exist_ok=True)
    OUTPUT_FOLDER = f"{BASE_FOLDER}/{mode}-output"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for person1_turn, person2_turn in zip_longest(dialogue_dict[person_ids[0]], dialogue_dict[person_ids[1]]):
        if person1_turn:
            ret = process_turn(person_ids[0], person1_turn, mode, BASE_FOLDER)
            dialogue_running += ret + '\n'

        if person2_turn:
            ret = process_turn(person_ids[1], person2_turn, mode, BASE_FOLDER)
            dialogue_running += ret + '\n'

    with open(f"{OUTPUT_FOLDER}/{instance_id}.txt", "w") as f:
        f.write(dialogue_running)

    return dialogue_running


# Load dataset
dataset = load_dataset("knkarthick/dialogsum", split='test')

# Prepare speaker monologues
speaker_monologues = {}
for instance in tqdm(dataset):
    instance_id = instance['id']
    speaker_monologues[instance_id] = {}
    dialogues = instance['dialogue'].split('\n')
    for dialogue in dialogues:
        speaker_identity, sentences = dialogue.split(':', 1)
        sentences = sentences.strip()
        if speaker_identity not in speaker_monologues[instance_id]:
            speaker_monologues[instance_id][speaker_identity] = []
        speaker_monologues[instance_id][speaker_identity].append(sentences)

print(len(speaker_monologues.keys()))  # Should be 1500 (length of test set)

# Main loop for generation and dataset creation
for MODE in ['ATAS', 'ATOS', 'OTAS', 'OTOS']:
    shutil.rmtree('/content/drive/MyDrive/ling384/replacements', ignore_errors=True)

    print(f"Generating for this mode.... {MODE}!!")
    for instance_id in tqdm(speaker_monologues.keys()):
        generate_replacement_both_speakers(instance_id, speaker_monologues[instance_id], mode=MODE)

    disfluent_dialogues = []
    dialogues = []
    ids = []
    summaries = []

    for instance in tqdm(dataset):
        ids.append(instance['id'])
        summaries.append(instance['summary'])
        with open(f"/content/drive/MyDrive/ling384/replacements/{MODE}-output/{instance['id']}.txt", 'r') as f:
            lines = f.readlines()

        text = ''.join(lines).strip()
        disfluent_dialogues.append(text)
        dialogues.append(instance['dialogue'])

    assert len(dialogues) == len(disfluent_dialogues) == len(ids) == len(summaries) == 1500

    data = {
        'id': ids,
        'dialogue': dialogues,
        'disfluent_dialogue': disfluent_dialogues,
        'summary': summaries,
    }

    dataset = dataset.from_dict(data)
    dataset.push_to_hub(f"sophiayk20/replacement-both-speakers", split=MODE)
