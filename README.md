# Speech Summarization of Speech Disfluency in Multi-Speaker Dialogues

- Course Final Project for LING 384/584: Computational Psycholinguistics (Prof. Tom McCoy)
- Yale University (Spring 2025)
- Sophia Kang

## Tech Stack
- Python
- LARD algorithm, NLTK sent_tokenize, word_tokenize
- PyTorch, HuggingFace transformers, datasets, evaluate (ROUGE, METEOR, COMET)
- Compute: T4 GPU via Google Colab

## Required Libraries
- LARD: `git clone https://github.com/tatianapassali/artificial-disfluency-generation.git` and `%cd /content/artificial-disfluency-generation`

## Project Motivation

This paper examines the psycholinguistic phenomenon of speech disfluency. Speech disfluency refers to

## Methodology
- LARD algorithm
- Summarization Model: `facebook/BART-large-CNN`

## Results


## Code
- `artificial-disfluency-generation/python_files/create_dataset.py` will not work with HuggingFace datasets due to deprecated version of dataframe feature when using concat dataframe. I have included an altered version of the file that I have used for experiments in `artificial-disfluency-generation`.