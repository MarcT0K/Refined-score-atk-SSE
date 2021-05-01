Refined Score attack attack
===

This repo contains Python implementation of Refined Score attack presented in: [Insert article ref]


# About
This repository contains everything necessary to simulate the attacks (Score attack, Refined score attack, with clustering, generalized version). This code was used to obtained all the results presented in the paper. Thus, it allows to simulate over Enron or Apache dataset, with or without countermeasure, with varying parameters.

In the folder `score_attacks`, you find 3 elements:

* `src`: folder containing all the utilitary codes to simulate the attacks
* `simulate_attack.py`: Python script to simulate an attack. The user can specify parameters (dataset, countermeasure, vocabulary size, query set size, number of known queries).
* `generate_results.py`: launch scripts that were used to obtain the vast majority of the results presented in the paper. 


# Getting started
To simulate a simple attack, you just have to follow these instructions:
0. You must have at least Python 3.6 and `pip` installed.
1. Download the repository
2. Launch the setup script (install dependencies and download the datasets): `sh setup.sh`.
3. Move to `score_attacks` folder: `cd score_attacks`
4. Launch the simulation script: `python3 simulate_attack.py`

To explore further the attack, you can change the parameters:

```
Score attack simulator

optional arguments:
  -h, --help            show this help message and exit
  --similar-voc-size SIMILAR_VOC_SIZE
                        Size of the vocabulary extracted from similar
                        documents.
  --server-voc-size SERVER_VOC_SIZE
                        Size of the 'queryable' vocabulary.
  --queryset-size QUERYSET_SIZE
                        Number of queries which have been observed.
  --nb-known-queries NB_KNOWN_QUERIES
                        Number of queries known by the attacker. Known
                        Query=(Trapdoor, Corresponding Keyword)
  --countermeasure COUNTERMEASURE
                        Which countermeasure will be applied.
  --attack-dataset ATTACK_DATASET
                        Dataset used for the attack
```

You can find all these information using `python3 simulate_attack.py --help`. For example, you can launch the following command: `python3 simulate_attack.py --queryset-size 500 --nb-known-queries 5`. At the start of each simulation, the parameters are systematically printed. Moreover, we developped some logging so you can easily follow the simulation progress.

# Attack codes
The attack codes are in the file `src/attackers.py`. The rest of the file are only used to setup a simulation.

In `src/attackers.py` are present two classes: `ScoreAttacker` (for the Score and Refined score attacks) and `GeneralizedScoreAttacker` (for their "generalized" version cf. the paper). Both have methods `predict` and `predict_with_refinement` which needs a list of trapdoors as parameter. These methods will return, for each trapdoor, a keyword. You can also specify a `cluster_max_size` in the `predict_with_refinement` so you can perform a refinement with clustering (cf. paper appendices).

If you want to understand how to assess the accuracy of these attackers, you can watch either the `simulate_simulation.py` or the `result_procedures.py` (this second file is more complex but also much more messy).

# Code readability
Before the publication of this repo, we tried to simplify as much as possible our function to only keep what is relevent in the paper. We also added **docstrings** and **typing** so it is easier for a reader to understand what the functions/classes do and how to use them.

# Further questions
Feel free to contact us if ever you have some questions about the code or the attack itself (or if you spotted some bugs in our code).