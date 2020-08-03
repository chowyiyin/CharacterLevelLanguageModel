# CharacterLevelLanguageModel

In this project, a machine learning pipeline was implemented to recover a pair of plaintexts from a two-time pad encryption. The pipeline includes the training and use of a character-level language model that handles sequential data and a dynamic programming algorithm. It was developed based on the paper “A Natural Language Approach to Automated Cryptanalysis of Two-time Pads”. The language model prototype uses a neural network instead of an n-gram language model, which was suggested in the paper, to further enhance the effectiveness of the model in recovering plaintexts. 

This repository contains a `.ipynb` file that contains the definition and implementation of a `LSTM Character Level Language Model` using the `Skorch` library and a dynamic programming algorithm to recover the plaintexts. Experiments was run on a subset of the Enron Dataset and results from the latest run can be found in the `results` directory.
