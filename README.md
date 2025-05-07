# Speech Summarization of Speech Disfluency in Multi-Speaker Dialogues

- Course Final Project for LING 384/584: Computational Psycholinguistics (Prof. Tom McCoy)
- Yale University (Spring 2025)
- Sophia Kang

## Tech Stack
- Python
- LARD algorithm, NLTK sent_tokenize, word_tokenize
- PyTorch, HuggingFace transformers, datasets, evaluate (ROUGE, METEOR, COMET)
- Compute: L4 and T4 GPU via Google Colab

## Required Libraries
- LARD: `git clone https://github.com/tatianapassali/artificial-disfluency-generation.git` and `%cd /content/artificial-disfluency-generation`

## Project Motivation

This paper examines the psycholinguistic phenomenon of speech disfluency. Speech disfluency refers to the phenomenon of injected interruptions absent in the normal flow of speech, including repetitions, restarts, and replacements. We examine the effect of introducing disfluency in multi-speaker dialogues in consideration of speakers, turns, and utterances by devising multiple disfluent settings, and use these disfluent settings to derive an understanding of whether the span of disfluency or number of interruption points may be more important in impacting downstream performance. We also experiment with mixed disfluency settings and compare how these result in different performance compared to a single-type disfluency setting.

## Methodology
- LARD algorithm
- Summarization Model: `facebook/BART-large-CNN`
- DialogSum dataset (turn-taking conversation with 2 speakers)

## Main Contributions
- We introduce disfluency in multi-speaker dialogues by varying the number of disfluent speakers, turns, and utterances using the LARD algorithm with three disfluency types - repetitions, replacements, and restarts.
- We demonstrate that the summarization perfromance degrades with increasing disfluency complexity and identify that disfluency span has a stronger impact on ROUGE scores than number of interruption points.
- We reveal that among the three disfluency types examined, replacement disfluency has the most detrimental effect on summarization, both in single- and mixed-disfluency settings.

## Notes on Code
- `artificial-disfluency-generation/python_files/create_dataset.py` will not work with HuggingFace datasets due to deprecated version of dataframe feature when using concat dataframe. I have included an altered version of the file that I have used for experiments in `artificial-disfluency-generation`.
- Code for generating disfluency are all within the `generate` folder. Code for Section 3 is in `repetitions`, `replacement`, and `restarts`, whereas code for Section 4 is in `multiple`.
- Evaluation code is in `evaluation` and other code used for generating relevant statistics are in `statistics`.