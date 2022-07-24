# [Color Overmodification Emerges from Data-Driven Learning and Pragmatic Reasoning](https://escholarship.org/uc/item/9kn7n6qb)

This repository contains the code for all experiments and analyses for the paper:

@inproceedings{fang2022color, title={Color Overmodification Emerges from Data-Driven Learning and Pragmatic Reasoning}, author={Fang, Fei and Sinha, Kunal and Goodman, Noah D. and Potts, Christopher and Kreiss, Elisa}, booktitle={Proceedings of the {A}nnual {M}eeting of the {C}ognitive {S}cience {S}ociety}, year={2022}}

## Getting started

Run the following command to install all prerequisites:

```
conda env create -f environment.yml
conda activate fang2022
```

## Datasets

- The datasets used in the experiments reported in the paper are available in `data/`. 
  - `rf-uniform` is the dataset used in Exp. 1, and corresponds to the uniform condition and high-salience condition in Exp. 2 and Exp. 3, respectively
  - `rf-typicality-unishape` corresponds to the typicality condition in Exp. 2
  - 'single-pixel-color` corresponds to the `low-salience` condition in Exp. 3
  - Each of the above is associated with 5 subfolders in `data/`. The four with suffixes (i.e., one of `both-needed`, `either-okay`, `color-needed`, `shape-needed`) are held-out evalsets; the remaining one is used for training.
- Use `shapeworld.py` to generate datasets 
  - Specify dataset config in `datagen_config.yaml`; see `datagen_config.template` for usage and example
  - Run `python shapeworld.py` to generate train data (in NumPy format) which will be stored by default in `data/`
  - If `visualize` is set to `True` in `datagen_config.yaml`, samples for visualization will be generated and stored by default in `viz_data/`; open up `viz_data/{dataset_name}/index.html` for sample output

## Models

- `models.py` implements all models in the paper
- `models/shapeworld` contains trained models on which the results are based. Models trained on each dataset are stored in their respective folders.

## Reproducing the results

### To train
To train the semantic functions in $L_{Eval}$ and the ensemble of semantic functions in $S_{RSA}$'s internal listener $L_{RSA}$, run the following:
```
python train.py --dataset {rf-uniform|rf-typicality-unishape|single-pixel-color} --l0 --cuda
```
Note that the command above will train the semantic functions in both $L_{Eval}$ and $L_{RSA}$.

To train the literal speaker $S_{Lit}$, run the following:
```
python train.py --dataset {rf-uniform|rf-typicality-unishape|single-pixel-color} --contextual --cuda 
```
Pardon the confusing flag name `contextual`, which refers to the literal speaker $S_{Lit}$.

### To evaluate the speakers

To evaluate the speaker's communication with $L_{Eval}$ on the held-out evalsets, run the following:
```
python outputs.py --eval_dataset {rf-uniform|rf-typicality-unishape|single-pixel-color} --speaker {rsa_ensemble|contextual} --listener val
```
Note that here again, `contextual` refers to the literal speaker $S_{Lit}$.

This will generate a series of `.npy` files in `eval_results/{rsa_ensemble|contextual}` regarding the generated sequences. We ran 5 trials with different random seeds, which correspond to the subfolders `0-4`.

You can then use the Jupyter notebooks `analysis/{dataset_name}/{speaker_type}-utterance-dist.ipynb` to analyze the eval results.

### To analyze the learned semantics

To evaluate the learned semantic functions on the held-out evalsets, run the following:
```
python semantics.py --eval_dataset {rf-uniform|rf-typicality-unishape|single-pixel-color}
```

You can then use the Jupyter notebooks `analysis/{dataset_name}/probe-semantics.ipynb` to analyze the uncertainty in the learned semantics.

## Acknowledgments

The following files in this repository are adapted from https://github.com/juliaiwhite/amortized-rsa:
- `colors.py`
- `data.py`
- `language_model.py`
- `models.py`
- `run.py`
- `shapeworld.py`
- `train.py`
- `util.py`
- `vision.py`