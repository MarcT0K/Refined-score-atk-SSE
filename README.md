Implementations of Score attacks
===

This repository contains Python implementation of the attacks presented in: 

M. Damie, F. Hahn, A. Peter, *"A Highly Accurate Query-Recovery Attack against Searchable Encryption using Non-Indexed Documents"* (USENIX 2021)


# About
This repository contains everything necessary to simulate the attacks (Score attack, Refined score attack, with clustering, generalized version). This code was used to obtained all the results presented in the paper mentioned above. It allows a simulation with varying parameters (choice of dataset, countermeasures, document set sizes, etc.).

In the folder `score_attacks`, you find 3 elements:

* `src`: folder containing all the utilitary codes to simulate the attacks
* `simulate_attack.py`: Python script to simulate an attack with parameters specified by the user (dataset, countermeasure, vocabulary size, query set size, number of known queries).
* `generate_results.py`: launch scripts that were used to obtain the vast majority of the results presented in the paper.


# Getting started
To simulate a simple attack, you just have to follow these instructions:

0. You must have at least Python 3.6 and `pip` installed. NB: you can use a Python virtual environment if you want to keep a clean Python environment (the setup script will install few dependencies).
1. Download the repository
2. Launch the setup script (install dependencies and download the datasets): `bash setup.sh`. The dataset download and decompression may take a while (~1h) depending on your connection and your computer. **Dataset size**: Apache emails takes 252 MB and Enron dataset takes **2.6 GB!!**
3. Move to `score_attacks` folder: `cd score_attacks`
4. Launch the simulation script: `python3 simulate_attack.py`. One attack simulation (including the keyword extraction) takes few minutes.

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

You can find all these information using `python3 simulate_attack.py --help`. For example, you can launch the following command: `python3 simulate_attack.py --queryset-size 500 --nb-known-queries 5`. At the start of each simulation, the parameters are systematically printed. Moreover, we developped some logging so you can easily follow the simulation progress. At the end, the accuracies with and without refinement are printed.

# Attack code
The attack codes are in the file `src/attackers.py`. The rest of the files are only used to setup a simulation (keyword extraction, email cleaning, query generation, etc.).

In `src/attackers.py` are present two classes: `ScoreAttacker` (for the Score and Refined score attacks) and `GeneralizedScoreAttacker` (for their "generalized" version cf. the paper). Both have methods `predict` and `predict_with_refinement` which needs a list of trapdoors as parameter. These methods will return, for each trapdoor, a keyword. You can also specify a `cluster_max_size` in the `predict_with_refinement` so you can perform a refinement with clustering (cf. paper appendices).

If you want to understand how to assess the accuracy of these attackers, you can watch either the `simulate_attack.py` or the `result_procedures.py` (this second file is much messier).

# Result reproduction
If you want to reproduce most results that are presented in the paper, you have to execute `generate_results.py`. The complete execution with 50 repetitions of each experiment may take several days depending on your machine specifications. If you want to reduce the number of repetitions, you can change the variable `NB_REP` in `src/result_procedures.py`. For convenience purpose, this script also have a file logging (in `results.log`) so you can analyze the logs later.

# Code readability
Before the publication of this repo, we tried to simplify as much as possible our function to only keep what is relevent in the paper. We also added **docstrings** and **typing** so it is easier for a reader to understand what the functions/classes do and how to use them.

# Further questions
Feel free to contact me (marc [dot] damie [at] etu [dot] utc [dot] fr) if you have some questions about the code or the attack itself (or if you spotted some bugs in our code).

# Known issues
If you have the error `setup.sh: 1: set: Illegal option -` when executing `setup.sh`, this may be due to the file formatting. It happens when sending (via scp) a file from a Windows computer to a Linux server (Windows formatter using different hidden characters than Unix formatters). To solve this issue, on the Linux server, you just have to install dos2unix (via apt for Debian and Ubuntu) and launch `dos2unix setup.sh`. Then, the script should have the adequate format and should be executed successfully.

The installation and the code have been tested on Debian and Ubuntu so some issues may occur on other operating systems.
