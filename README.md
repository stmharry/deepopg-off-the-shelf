# DeepOPG (Off-the-shelf)

This repository implements [DeepOPG](https://arxiv.org/abs/2103.08290) with mostly off-the-shelf components.

## Prerequisites

- Python 3.9 (strict)
- Poetry

  ```bash
  $ curl -sSL https://install.python-poetry.org | python3 -
  ```

## Installation

```bash
$ git clone https://github.com/stmharry/deepopg-off-the-shelf.git && cd deepopg-off-the-shelf
$ poetry env use python3.9 && poetry shell && poetry install --with dev
```

## Usage

### Inference

```bash
$ make test ARCH=detectron2 MODEL_DIR=/path/to/model_dir
```

### Training

```bash
$ make train ARCH=detectron2 CONFIG_FILE=./configs/mask_rcnn_mvitv2_t_3x.py
```
