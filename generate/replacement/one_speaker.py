def generate_replacement_one_speaker(instance_id, dialogue_dict, mode=Literal['ATAS', 'OTAS', 'ATOS', 'OTOS']):
  """

    Given dialogue_dict, return disfluent dialogue in string form
    Generates at least 1 replacement disfluency in each turn

  """

  global global_turn_flag
  # set global turn flag to false at dialogue level
  global_turn_flag = {person_id:False for person_id in person_ids} 
  random.seed(42)

  dialogue_running = ""

  BASE_FOLDER=f"/content/drive/MyDrive/ling384/replacements/one_speaker"
  os.makedirs(BASE_FOLDER, exist_ok=True)
  OUTPUT_FOLDER=f"{BASE_FOLDER}/{mode}-output"
  os.makedirs(OUTPUT_FOLDER, exist_ok=True)
  
  # for each set of sentences that a pereson says in a single turn
  # person1_turns: 'Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?'
  for person1_turn, person2_turn in zip_longest(dialogue_dict[person_ids[0]], dialogue_dict[person_ids[1]]):
      replacement_index = random.choice([0, 1])

      if person1_turn:
        # if this person replaces
        if replacement_index == 0:
          ret = process_turn(person_ids[0], person1_turn, mode, BASE_FOLDER)
          dialogue_running += ret
          dialogue_running += '\n'
        # else, other person replaces, so keep original sentence
        else:
          dialogue_running += f"{person_ids[0]}: "
          dialogue_running += person1_turn
          dialogue_running += '\n'
      
      if person2_turn:
        # if this person replaces
        if replacement_index == 1:
          ret = process_turn(person_ids[1], person2_turn, mode, BASE_FOLDER)
          dialogue_running += ret
          dialogue_running += '\n'
        else:
          dialogue_running += f"{person_ids[1]}: "
          dialogue_running += person2_turn
          dialogue_running += '\n'

  with open(f"{OUTPUT_FOLDER}/{instance_id}.txt", "w") as f:
    f.write(dialogue_running)
  
  return dialogue_running