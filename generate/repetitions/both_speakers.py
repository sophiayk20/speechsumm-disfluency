from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Literal
from itertools import zip_longest
from python_files.disfluency_generation import LARD # should be in topic-controllable-summarization directory
import re
import os
import shutil
shutil.rmtree('/content/drive/MyDrive/ling384/repetitions') # remove this folder in case it was already created

dataset = load_dataset("knkarthick/dialogsum", split='test')

speaker_monologues = {} # 'train_0': {'#Person1#': ['Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?', 'Yes, well, you haven't had one for 5 years. You should have one every year.'],
                        #             '#Person2#': []}
for instance in tqdm(dataset):
  instance_id = instance['id']
  speaker_monologues[instance_id] = {} # init dict
  dialogues = instance['dialogue'].split('\n')
  for dialogue in dialogues:
    # max split at 1 because 17:05 is in text
    speaker_identity, sentences = dialogue.split(':', 1)
    sentences = sentences.strip() # sentences is a string
    if speaker_identity not in speaker_monologues[instance_id].keys():
      speaker_monologues[instance_id][speaker_identity] = []
    speaker_monologues[instance_id][speaker_identity].append(sentences)

print(len(speaker_monologues.keys()))

# instantiate LARD object
lard = LARD()

person_ids = ['#Person1#', '#Person2#']
global_turn_flag = {person_id:False for person_id in person_ids} 

def process_turn(person_id, person_turns, mode, BASE_FOLDER, repetition_degree=2):
  global person_ids, global_turn_flag

  running = ""

  with open(f"/{BASE_FOLDER}/repetition_{mode}_stat.txt", "a") as f:
    person_turns_sentences = sent_tokenize(person_turns)

    # this turn flag is unique to this turn of person hence not global
    turn_flag=False
    disfluency_count = 0
    speaker_running = []

    for person_sentence in person_turns_sentences:
      word_rep = word_tokenize(person_sentence)
      string_rep = ' '.join(wp for wp in word_rep)

      # if we already created for this turn
      if mode == "ATOS" and turn_flag:
        speaker_running.append(person_sentence)
        continue
      if mode in ["OTOS", "OTAS"] and global_turn_flag[person_id] == True:
        speaker_running.append(person_sentence)
        continue
      
    disfluency = lard.create_repetitions(string_rep, repetition_degree)

    # if disfluency can be generated
    if disfluency[0]:
      tokens = disfluency[0].split()
      rejoined_sentence = ' '.join(tokens)

      # regular expression to handle all punctuation marks
      # This ensures that punctuation is attached directly to the preceding word without extra spaces
      rejoined_sentence = re.sub(r'\s([.,!?;])', r'\1', rejoined_sentence)

      # Handle cases like "n't" contractions to avoid spacing issues
      rejoined_sentence = re.sub(r"\sn't", "n't", rejoined_sentence)
      speaker_running.append(rejoined_sentence)

      disfluency_count += 1

      # if all turn one sentence, should be set to True
      if mode == "ATOS":
        turn_flag=True
    # disfluency cannot be generated, add original sentence
    else:
      speaker_running.append(person_sentence)

    # create new data
    running += f"{person_id}: "
    running += " ".join(sent for sent in speaker_running)
    running += "\n"

    # if we added disfluency in turn, disable future generation (one turn covered)
    if disfluency_count > 0 and mode in ['OTAS', 'OTOS']:
      global_turn_flag[person_id]= True

    f.write(f"{disfluency_count}\n")

  return running

def generate_repetition_both_speakers(instance_id, dialogue_dict, mode=Literal['ATAS', 'OTAS', 'ATOS', 'OTOS'], repetition_degree=2):
  """
    Given dialogue dict, return disfluent dialogue in string form
    Generates at least 1 repetition disfluency in each turn
  """
  person1_num_speaks = len(dialogue_dict['#Person1#'])
  person2_num_speaks = len(dialogue_dict['#Person2#'])
  #print(dialogue_dict)

  global global_turn_flag

  global_turn_flag = {person_id: False for person_id in person_ids}

  dialogue_running = ""

  BASE_FOLDER=f"/content/drive/MyDrive/ling384/repetitions/both_speakers"
  os.makedirs(BASE_FOLDER, exist_ok=True)
  OUTPUT_FOLDER=f"{BASE_FOLDER}/{mode}-output"
  os.makedirs(OUTPUT_FOLDER, exist_ok=True)

  # zip longest is needed because 1 speaker might speak for more times (not even)
  for person1_turn, person2_turn in zip_longest(dialogue_dict[person_ids[0]], dialogue_dict[person_ids[1]]):
    if person1_turn:
      ret = process_turn(person_ids[0], person1_turn, mode, BASE_FOLDER, repetition_degree)
      dialogue_running += ret
      dialogue_running += '\n'

    if person2_turn:
      ret = process_turn(person_ids[1], person2_turn, mode, BASE_FOLDER, repetition_degree)
      dialogue_running += ret
      dialogue_running += '\n'
    
  with open(f"{OUTPUT_FOLDER}/{instance_id}.txt", "w") as f:
    f.write(dialogue_running)

  return dialogue_running

#for MODE in ['ATAS', 'ATOS', 'OTAS', 'OTOS']:
for MODE in ['OTOS']:
  print(f"Generating for this mode.... {MODE}!!")
  for instance_id in tqdm(speaker_monologues.keys()):
    generate_repetition_both_speakers(instance_id, speaker_monologues[instance_id], mode=MODE, repetition_degree=2)
  
  disfluent_dialogues = []
  dialogues= []
  ids = []
  summaries= []

  for instance in tqdm(dataset):
    ids.append(instance['id'])
    summaries.append(instance['summary'])
    with open(f"/content/drive/MyDrive/ling384/repetitions/both_speakers/{MODE}-output/{instance['id']}.txt", 'r') as f:
      lines = f.readlines()

    text = ''.join(lines).strip()
    disfluent_dialogues.append(text)
    dialogues.append(instance['dialogue'])

  assert(len(dialogues) == len(disfluent_dialogues) == len(ids) == len(summaries) == 1500)

  data = {
      'id': ids,
      'dialogue': dialogues,
      'disfluent_dialogue': disfluent_dialogues,
      'summary': summaries,
  }

  dataset = Dataset.from_dict(data)
  dataset.push_to_hub("sophiayk20/repetition-both-speakers", split=MODE)