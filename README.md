# DeepOPG (Off-the-shelf)

This repository implements [DeepOPG](https://arxiv.org/abs/2103.08290) with mostly off-the-shelf components.

## Prerequisites

- Python 3.9 (strict)
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

## Installation

```bash
$ git clone https://github.com/stmharry/deepopg-off-the-shelf.git && cd deepopg-off-the-shelf
$ conda env create -f environment.yaml && conda activate deepopg
```

## Usage

### Training

To run training, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `CONFIG_FILE`: Path to the config file.
- `DATA_DIR`: Path to the dataset directory.

The model will be saved to `$(ROOT_DIR)/models/$(MODEL_NAME)` by default where `MODEL_NAME` is the name of an auto-generated model directory.

```bash
$ make train ARCH=detectron2 CONFIG_FILE=./configs/mask_rcnn_mvitv2_t_3x.py
```

### Inference

To run inference, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `MODEL_NAME`: Name of the model directory.

The inference results will be saved to `$(ROOT_DIR)/results/$(RESULT_NAME)` by default where `RESULT_NAME` is the name of an auto-generated result directory.

```bash
$ make test ARCH=detectron2 MODEL_NAME=$(INSERT_YOUR_MODEL_NAME)
```
