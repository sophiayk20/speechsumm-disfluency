from datasets import load_dataset
import pandas as pd
import os
from python_files.create_dataset import create_dataset  # Adjust path if needed
import nltk
import re
import shutil

nltk.download('wordnet')

dataset = load_dataset("knkarthick/dialogsum")

# process each example in huggingface dataset
split_rows = []

for idx, example in enumerate(dataset['test']):
    dialogue_id = example['id']
    dialogue = example['dialogue']
    turns = dialogue.strip().split("\n")  # assume turns are newline-separated

    for turn in turns:
        match = re.match(r"(#Person\d+#): (.+)", turn)
        if match:
            speaker = match.group(1)
            sentence = match.group(2)
        else:
            speaker = "Unknown"
            sentence = turn
        split_rows.append({
            "id": dialogue_id,
            "speaker": speaker,
            "sentence": sentence
        })

# convert to dataframe
split_df = pd.DataFrame(split_rows)

# save this CSV for disfluency generation
csv_path = "/content/drive/MyDrive/temp_dialogue_turns.csv"
split_df.to_csv(csv_path, index=False)

# list of experiment configurations [fluent, repetition, restart, replacement]
experiment_percentages = [
    [100, 0, 0, 0],
    [50, 20, 15, 15],
    [50, 15, 20, 15],
    [50, 15, 15, 20],
    [50, 50, 0, 0],
    [50, 0, 50, 0],
    [50, 0, 0, 50],
    [0, 100, 0, 0],
    [0, 0, 100, 0],
    [0, 0, 0, 100]
]

#experiment_percentages = [ [50, 15, 20, 15]]

# Output directory (e.g., for Google Drive)

output_base = "/content/drive/MyDrive/disfluency_outputs"

if os.path.exists(output_base):
    shutil.rmtree(output_base)

os.makedirs(output_base,exist_ok=True)

# fixed parameters
repetition_degrees_percentage = [50, 30, 20]
replacement_types_percentage = [20, 15, 20, 15, 20, 10]

# run experiments in loop
for i, config in enumerate(experiment_percentages):
    fluent, rep, restart, replace = config
    output_dir = f"{output_base}/exp_{i:02d}_f{fluent}_r{rep}_rs{restart}_rp{replace}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning config {i:02d}: [Fluent={fluent}%, Repetition={rep}%, Restart={restart}%, Replacement={replace}%]")
    print(f"Output: {output_dir}")

    # base keyword arguments
    kwargs = {
        "input_file_path": csv_path,
        "column_text": "sentence", # where we should synthesize the tokens
        "output_dir": output_dir,
        "create_all_files": False,
        "concat_files": True
    }

    if fluent == 0:
        kwargs["keep_fluent"] = False
        kwargs["percentages"] = [rep, restart, replace]
        if rep > 0:
            kwargs["repetition_degrees_percentage"] = repetition_degrees_percentage
        if replace > 0:
            kwargs["replacement_types_percentage"] = replacement_types_percentage
    else:
        kwargs["keep_fluent"] = True
        kwargs["percentages_with_fluent"] = config
        if rep > 0:
            kwargs["repetition_degrees_percentage"] = repetition_degrees_percentage
        if replace > 0:
            kwargs["replacement_types_percentage"] = replacement_types_percentage

    print("kwargs being passed to create_dataset:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

    # Call function with only the necessary kwargs
    create_dataset(**kwargs)

print("\n ---- All experiments completed. -----")