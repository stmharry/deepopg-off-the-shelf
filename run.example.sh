#!/bin/bash

### env

export COCO_ANNOTATOR_USERNAME=
export COCO_ANNOTATOR_PASSWORD=
# export CUDA_VISIBLE_DEVICES=""

### executable

# export PYTHON=python
# export PYTHON="python -m pdb"

### target

# export TARGET=train
# export TARGET=test
# export TARGET=postprocess
# export TARGET=visualize
# export TARGET=visualize-coco
# export TARGET=evaluate

### dataset

# export DATASET_NAME=pano_eval
# export DATASET_NAME=pano_ntuh
# export DATASET_NAME=pano_ntuh_debug
# export DATASET_NAME=pano_odontoai_train

### arch

# export ARCH=detectron2
# export ARCH=yolo

### setup

if [ "$TARGET" = "train" ]; then
	# export CONFIG_NAME=

elif [ "$TARGET" = "visualize-gt" ]; then
	# export RESULT_NAME=

else
	if [ "$ARCH" = "detectron2" ]; then
		# export MODEL_NAME=
		# export RESULT_NAME=

	elif [ "$ARCH" = "yolo" ]; then
		# export MODEL_NAME=
		# export RESULT_NAME=

	fi

fi

make $TARGET
