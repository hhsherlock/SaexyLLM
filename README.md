# SaexyLLM

This repository contains code for various methods I experimented with to create a dialect-generating large language model (each folder represents a different approach). The primary goal is to train the model using a small dataset due to limited resources.

Methods:
- **Data Crawling** – Collecting data from the internet and transcribed audio files, then cleaning up the sentences.
- **Dictionary Approach** – Calculating sentence perplexity and replacing words using an existing dictionary.
- **LoRA Fine-Tuning** – Fine-tuning the model with LoRA; different parameters were tested.
- **Skip-Gram** – Using the skip-gram method to find equivalent words in standard and dialect German.
- **Transition Matrix** – Modifying the embedding matrix to facilitate dialect generation.

**Evaluation** – Assessing the generated output using BLEU scores and perplexity measurements.