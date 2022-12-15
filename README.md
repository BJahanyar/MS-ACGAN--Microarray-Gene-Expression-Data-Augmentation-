# Genome Generator

## Datasets
* Original datasets, generated data, and selected features files are available in this link:  [Datasets](https://1drv.ms/u/s!AhrVsNlX-OnLi9Fgr25EZaSpmpieGw?e=q7SwVb)

## Installation

* Install pytorch >= 1.8 from original website: [pytorch](https://pytorch.org/)
* Install `requirements.txt`:

```commandline
pip install -r requirements.txt
```

## Train

At the first step, check configs in the config folder. There are some config file:

* `dataset`
* `features`: selected features from mRMR algorithm
* `model`: model configurations
* `train`: train configs and hyperparameters

To train your ACGAN you need to run the following code:

```commandline
python train.py
```

## Classification

To Classify datasets use `classification.py`

```commandline
python classification.py
```

- [x] Logistic Regression classification
- [x] NN classification
- [x] Grid search

## Validation

ACGAN validation with trained classifier

```commandline
python validation
```

## Jupyter notebooks

### `classification.ipynb`

Jupyter notebook to classify dataset with some normalization techniques.

### `ergene_nomalization.ipynb`

Implementation of ergene normalization

### `Feature_selection.ipynb`

mRMR feature selection

### `Plot_with_pca.ipynb`

Feature dimension reduction with pca