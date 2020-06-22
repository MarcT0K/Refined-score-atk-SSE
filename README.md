QueRyvolution attack
===

This repo contains Python implementation of QueRyvolution attack presented in: [Insert article ref]


# About
This code aims at simulating the QueRyvolution attack against SSE. Thus, it only represents a PoC and not an attack library. However, if someone would like to develop such a library, they could simply extract the `KeywordTrapdoorMatchMaker` class from `matchmaker.py`. This class is designed so it can be integrated in a real scenario.


# Getting started
Download the repo and launch `setup.sh`.

To run a basic attack:
```
python attack_scenario.py
```

# Result procedures
All the procedures we used to produce our results are in `result_procedures.py`.

