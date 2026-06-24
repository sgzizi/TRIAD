# Anatomy of a Decision: Uncertainty-aware Hierarchical Intent Learning via Flow Matching for Multimodal Recommendation

This repository contains the source code for the paper "Anatomy of a Decision: Uncertainty-aware Hierarchical Intent Learning via Flow Matching for Multimodal Recommendation".

Our implementation is based on the [MMRec: A modern MultiModal Recommendation toolbox](https://github.com/enoche/MMRec). We extend our gratitude to the authors of MMRec for their open-source contribution to the community.

## Core Code

The core of our proposed method is located in the following file:

```
./src/models/uhiflow.py
```

This file contains the implementation of the Uncertainty-aware Hierarchical Intent Learning via Flow Matching model.

## Environment Setup

Our code is developed and tested under the same environment as MMRec. To set up the necessary environment, please install the dependencies listed in the `requirements.txt` file in MMRec.

## Data

The Baby and Sports subsets of our dataset are from the MMRec codebase, and they can be placed in the "./data/" directory for execution.

**Installation:**

You can install all the required packages using pip:

```bash
pip install -r requirements.txt
```

## Running the Code

To run the experiments, navigate to the `src` directory and execute the `main.py` script. You can specify the dataset to be used via command-line arguments.

```bash
cd src
python main.py --dataset <dataset_name>
```

Please refer to the `main.py` script for a full list of available arguments and options, including hyperparameters and other settings.



