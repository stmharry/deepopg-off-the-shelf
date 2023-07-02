# DeepOPG (Off-the-shelf)

This repository implements [DeepOPG](https://arxiv.org/abs/2103.08290) with mostly off-the-shelf components.

## Prerequisites

- [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

## Installation

```bash
$ git clone --recurse-submodules https://github.com/stmharry/deepopg-off-the-shelf.git && cd deepopg-off-the-shelf
$ conda env create -f environment.yaml && conda activate deepopg
$ make install
```

## Usage

Before running any commands, please set up the follow environment variables:

```bash
$ export ROOT_DIR=/path/to/root
```

The typical workflow is as follows:

1. Train a model or identify a model to use.
2. Run inference on the test set.
3. Postprocess the results.
4. Visualize the results.

### Training

To run training, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `CONFIG_FILE`: Path to the config file.

The model will be saved to `$(ROOT_DIR)/models/$(MODEL_NAME)` by default where `MODEL_NAME` is the name of an auto-generated model directory.

#### Example

```bash
$ make train ARCH=detectron2 CONFIG_NAME=mask_rcnn_mvitv2_t_3x.py
> PYTHONPATH=./detectron2:. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    python scripts/main.py \
        --main-app tools.lazyconfig_train_net:main \
        --config-file ./configs/mask_rcnn_mvitv2_t_3x.py \
        --data-dir /mnt/hdd/PANO/data \
        train.output_dir=/mnt/hdd/PANO/models/2023-07-02-070303
```

### Inference

To run inference, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `DATASET_NAME`: Name of the dataset.
- `MODEL_NAME`: Name of the model directory.

The inference results will be saved to `$(ROOT_DIR)/results/$(RESULT_NAME)` by default where `RESULT_NAME` is the name of an auto-generated result directory.

#### Example

```bash
$ make test ARCH=detectron2 DATASET_NAME=pano_debug MODEL_NAME=2023-05-30-175110
> PYTHONPATH=./detectron2:. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    python scripts/main.py \
        --main-app tools.lazyconfig_train_net:main \
        --config-file /mnt/hdd/PANO/models/2023-05-30-175110/config.yaml \
        --data-dir /mnt/hdd/PANO/data \
        --eval-only \
        train.init_checkpoint=/mnt/hdd/PANO/models/2023-05-30-175110/model_final.pth \
        train.output_dir=/mnt/hdd/PANO/results/2023-07-02-070543 \
        dataloader.test.dataset.names=pano_debug \
        dataloader.evaluator.output_dir=/mnt/hdd/PANO/results/2023-07-02-070543 \
        model.roi_heads.box_predictor.test_score_thresh=0.0 \
        model.roi_heads.box_predictor.test_nms_thresh=0.0 \
        model.roi_heads.box_predictor.test_topk_per_image=500
```

### Postprocess

To run postprocess, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `DATASET_NAME`: Name of the dataset.
- `RESULT_NAME`: Name of the result directory.

#### Example

```bash
$ make postprocess ARCH=detectron2 DATASET_NAME=pano_debug RESULT_NAME=2023-07-01-075136
> PYTHONPATH=./detectron2:. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    python scripts/commands.py \
        --data_dir /mnt/hdd/PANO/data \
        --result_dir /mnt/hdd/PANO/results/2023-07-01-075136 \
        --dataset_name pano_debug \
        --do_postprocess \
        --output_prediction_name instances_predictions.postprocessed.pth
```

### Visualization

To run visualization, one need to specify the following variables:

- `ARCH`: Either `detectron2` or `maskdino`.
- `DATASET_NAME`: Name of the dataset.
- `RESULT_NAME`: Name of the result directory.

#### Example

```bash
$ make visualize ARCH=detectron2 DATASET_NAME=pano_debug RESULT_NAME=2023-07-01-075136
> PYTHONPATH=./detectron2:. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    python scripts/commands.py \
        --data_dir /mnt/hdd/PANO/data \
        --result_dir /mnt/hdd/PANO/results/2023-07-01-075136 \
        --dataset_name pano_debug  \
        --prediction_name instances_predictions.postprocessed.pth \
        --do_visualize \
        --visualizer_dir visualize.postprocessed \
        --nodo_coco \
        --coco_annotator_url http://192.168.0.79:5000/api
```
